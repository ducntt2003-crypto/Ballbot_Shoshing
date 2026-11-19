import ctypes,subprocess,shutil
import mujoco
from .common import *
BUILD=ROOT/'observer_build';LIB=None
def compile_observer():
 BUILD.mkdir(exist_ok=False);pkg=Path(mujoco.__file__).parent;src=Path(__file__).with_name('observer.c');out=BUILD/'libtrack8i_observer.so';cmd=[shutil.which('cc'),'-shared','-fPIC','-O2','-std=c99','-Wall','-Wextra','-Werror','-I'+str(pkg/'include'),str(src),'-L'+str(pkg),'-l:libmujoco.so.3.3.2','-Wl,-rpath,'+str(pkg),'-o',str(out)]
 with (BUILD/'build.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
 save('observer_build/build.json',dict(command=cmd,source_sha256=sha(src),binary_sha256=sha(out),engine_binary_sha256=sha(pkg/'libmujoco.so.3.3.2'),engine_rebuilt=False))
def lib():
 global LIB
 if LIB is None:
  LIB=ctypes.CDLL(str(BUILD/'libtrack8i_observer.so'));LIB.track8i_info.argtypes=[ctypes.POINTER(ctypes.c_int)];LIB.track8i_read.argtypes=[ctypes.POINTER(ctypes.c_double)]*7
 return LIB
def raw():
 l=lib();a=(ctypes.c_int*5)();l.track8i_info(a);n,nv,ns,overflow,finished=list(a);st=np.zeros((n,ns));acc=np.zeros((n,nv));f=np.zeros((n,nv));t=np.zeros(n);la=np.zeros(nv);lf=np.zeros(nv);w=np.zeros(3);l.track8i_read(*[x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for x in [st,acc,f,t,la,lf,w]])
 return dict(states=st,cached_acc=acc,cached_constraint=f,times=t,final_acc=la,final_constraint=lf,raw_last_ground_W=w,overflow=overflow,finished=finished)
def root_columns(m):
 return np.array([int(m.jnt_dofadr[j])+k for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and m.body_parentid[m.jnt_bodyid[j]]==0 for k in range(3)]).reshape(-1,3)
def decode(m,r,before_v,after_v):
 assert r['finished'] and not r['overflow'] and len(r['times'])==4,'Incomplete/ambiguous native-stage observation'
 a=np.vstack([r['cached_acc'][1:],r['final_acc']]);f=np.vstack([r['cached_constraint'][1:],r['final_constraint']]);cols=root_columns(m);W=f[:,cols].sum(axis=1);dv=m.opt.timestep*np.array([1,2,2,1])@a/6;gap=after_v-before_v-dv
 r.update(acc=a,constraint=f,ground_W=W,qvel_identity_gap=gap,force_mapping_gap=W[-1]-r['raw_last_ground_W'],root_free_translation_columns=cols);return r
class StepObserver:
 """Wrap Python mj_step only to read caches before env.state() can overwrite them."""
 def __init__(self,e,collect):self.e=e;self.collect=collect;self.calls=0
 def __enter__(self):
  assert mujoco.get_mjcb_control() is None;assert lib().track8i_install()==1;self.original=mujoco.mj_step
  def wrapped(m,d,*args,**kwargs):
   if m is not self.e.model or d is not self.e.data:return self.original(m,d,*args,**kwargs)
   assert not args and not kwargs,'Onlyone native step perobservertransaction'
   before=d.qvel.copy();lib().track8i_begin()
   try:self.original(m,d)
   except BaseException:lib().track8i_cancel();raise
   lib().track8i_finish();r=decode(m,raw(),before,d.qvel.copy());self.calls+=1;self.collect(r)
  self.wrapped=wrapped;mujoco.mj_step=wrapped;return self
 def __exit__(self,*args):
  assert mujoco.mj_step is self.wrapped;mujoco.mj_step=self.original;assert lib().track8i_uninstall()==1;assert mujoco.get_mjcb_control() is None
