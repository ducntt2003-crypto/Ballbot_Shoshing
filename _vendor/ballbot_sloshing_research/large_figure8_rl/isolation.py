"""TRACK8N_ISOLATED_CONTROLLER_CONTEXT_V1: sequential, complete public MjData copy."""
import copy,types,threading,contextlib
from collections import deque
import mujoco
from .common import *
from ballbot_sloshing_research.track8_l.adapter import Adapter
VERSION='TRACK8Q_CONTEXT_V1_FROM_O_N_V1_1'
EXTERNAL={'model','data','tmp','renderer','pre_sample_hook','sample_hook'}

def encode(v,seen=None):
 if isinstance(v,np.ndarray):return dict(dtype=v.dtype.str,shape=v.shape,sha256=hashlib.sha256(v.tobytes()).hexdigest())
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,(str,int,float,bool)) or v is None:return v
 if isinstance(v,Path):return str(v)
 if isinstance(v,np.random.Generator):return encode(v.bit_generator.state)
 if isinstance(v,(list,tuple,deque)):return [encode(x) for x in v]
 if isinstance(v,dict):return {str(k):encode(a) for k,a in sorted(v.items(),key=lambda p:str(p[0]))}
 if isinstance(v,types.MethodType):return dict(bound_class=type(v.__self__).__name__,method=v.__func__.__qualname__)
 if isinstance(v,types.FunctionType):return dict(function=v.__qualname__,closure=[type(c.cell_contents).__name__ for c in v.__closure__ or []])
 if hasattr(v,'__dict__'):return dict(type=type(v).__module__+'.'+type(v).__name__,state=encode(v.__dict__))
 return str(v)
def python_state(e):return encode({k:v for k,v in e.__dict__.items() if k not in EXTERNAL and not k.startswith('_n_diag')})
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def arrays(d):
 a={k:getattr(d,k).copy() for k in dir(d) if not k.startswith('_') and isinstance(getattr(d,k),np.ndarray) and getattr(d,k).dtype.kind in 'biufc'}
 # Dense Newton, island discovery disabled: excluded fields are not computed by this pipeline.
 a={k:v for k,v in a.items() if 'island' not in k and not k.startswith('efc_AR') and not k.startswith('efc_JT') and k not in ['efc_J_rowadr','efc_J_rownnz','efc_J_rowsuper','efc_J_colind']}
 for k in ['time','ncon','nefc']:a[k]=np.asarray(getattr(d,k))
 a['warning_number']=d.warning.number.copy();a['warning_lastinfo']=d.warning.lastinfo.copy()
 for k in ['pos','frame','dist','geom1','geom2','dim','efc_address']:
  a['contact_'+k]=np.asarray([getattr(c,k) for c in d.contact]).copy()
 return a
def native_state(e):
 a=arrays(e.data);s=int(mujoco.mjtState.mjSTATE_INTEGRATION);x=np.empty(mujoco.mj_stateSize(e.model,s));mujoco.mj_getState(e.model,e.data,x,s);a['integration']=x;return a
def same(a,b):return set(a)==set(b) and all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in a)
def diff(a,b):
 out=[]
 for k in sorted(set(a)|set(b)):
  if k not in a or k not in b or a[k].shape!=b[k].shape:out.append(dict(field=k,shape_changed=True));continue
  if a[k].tobytes()!=b[k].tobytes():out.append(dict(field=k,shape=list(a[k].shape),dtype=str(a[k].dtype),max_abs=float(np.max(abs(a[k].astype(float)-b[k].astype(float)))) if a[k].size else 0.,mismatching_elements=int(np.count_nonzero(a[k]!=b[k]))))
 return out
def checkpoint(e):return dict(native=native_state(e),python=python_state(e))
def unchanged(e,c):return same(native_state(e),c['native']) and python_state(e)==c['python']
def archive(name,e):
 folder=ROOT/'snapshots';folder.mkdir(exist_ok=True);p=folder/(name+'.npz');assert not p.exists();a=native_state(e);np.savez_compressed(p,**a)
 units={'time':'s','ctrl':'Nm originalmotororder','qpos':'m/rad/unit quaternion, modelDOF dependent','qvel':'m/s or rad/s, modelDOF dependent','qacc':'m/s^2 or rad/s^2','qacc_warmstart':'generalized acceleration units','xfrc_applied':'N/Nm, WORLD','mocap_pos':'m,WORLD','mocap_quat':'unit quaternion wxyz'}
 save('snapshots/'+name+'.json',dict(time_s=float(e.data.time),source_object='branchlocalMjData',native_npz_sha256=sha(p),fields=[dict(name=k,shape=list(v.shape),dtype=str(v.dtype),unit=units.get(k,'MuJoCo native field, see model/field convention; no universal scalarunit'),frame='MODEL_NATIVE_FIELD_CONVENTION',bytes_sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in a.items()],python_state=python_state(e)))
def clone(e):
 assert not (e.model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_ISLAND)) and not mujoco.mj_isSparse(e.model) and e.model.opt.solver==int(mujoco.mjtSolver.mjSOL_NEWTON)
 f=copy.copy(e)
 for key,value in e.__dict__.items():
  if key not in EXTERNAL:setattr(f,key,copy.deepcopy(value))
 f.model=e.model;f.tmp=e.tmp;f.data=copy.copy(e.data);f.renderer=None;f.pre_sample_hook=None;f.sample_hook=None
 if 'act' in f.backbone.__dict__:
  fun=e.backbone.__dict__['act'];bound=[c.cell_contents for c in getattr(fun,'__closure__',()) or [] if isinstance(c.cell_contents,types.MethodType)]
  if len(bound)!=1 or bound[0].__self__ is not e.backbone or bound[0].__func__ is not type(e.backbone).act:raise ValueError('Unsupported instance act binding; no arbitrary closure sanitization')
  del f.backbone.__dict__['act']
 # Reject other direct function-valued mutable control ownership, never silently copy it.
 for obj in [f.backbone,f.backbone.inner]:
  if any(isinstance(v,types.FunctionType) for v in obj.__dict__.values()):raise ValueError('Unowned controller function')
 assert same(native_state(e),native_state(f));return f

class ContactSession:
 """Exclusive single-thread process-global collision ownership, not clone-local."""
 active=False
 def __enter__(self):
  if ContactSession.active or threading.current_thread() is not threading.main_thread() or mujoco.get_mjcb_control() is not None:raise RuntimeError('Incompatible callback ownership')
  ContactSession.active=True;self.adapter=Adapter()
  try:self.adapter.__enter__()
  except BaseException:ContactSession.active=False;raise
  return self
 def __exit__(self,*args):
  try:return self.adapter.__exit__(*args)
  finally:ContactSession.active=False
