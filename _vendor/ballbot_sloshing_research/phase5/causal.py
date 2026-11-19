"""Phase-binned matched-state forks preserving native and controller state exactly."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import copy,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,mujoco
from ballbot_sloshing_research.phase4.reward_experiment import EnergyEnv
from ballbot_sloshing_research.phase4.energy_diagnostics import RefinedAccounting
from ballbot_fast_rl.outer_loop.control import kinematics
from ballbot_fast_rl.robust_control.v3.identify import snapshot,SPEC
from .common import *

def encode(obj):
 if isinstance(obj,np.ndarray):return obj.tolist()
 if isinstance(obj,np.generic):return obj.item()
 if isinstance(obj,(str,int,float,bool)) or obj is None:return obj
 if isinstance(obj,dict):return {str(k):encode(v) for k,v in obj.items()}
 if isinstance(obj,(list,tuple)):return [encode(v) for v in obj]
 if hasattr(obj,'__dict__'):return dict(class_name=type(obj).__module__+'.'+type(obj).__name__,state=encode(obj.__dict__))
 return str(type(obj))

def fork(e):
 f=copy.copy(e)
 for key,value in e.__dict__.items():
  if key not in ['model','data','tmp','renderer','pre_sample_hook','sample_hook']:setattr(f,key,copy.deepcopy(value))
 f.model=e.model;f.tmp=e.tmp;f.data=mujoco.MjData(e.model);mujoco.mj_setState(e.model,f.data,snapshot(e),SPEC);mujoco.mj_forward(e.model,f.data);f.pre_sample_hook=None;f.sample_hook=None;return f

def branch(e,axis=None,sign=0,onset=0.):
 f=fork(e);start_t=float(f.data.time);f.max_steps=f.steps+300;acc=RefinedAccounting(f);physical=[];first_step=None;f.pre_sample_hook=acc.before
 def sample(env):
  acc.after(env);s=kinematics(env);ep=s['p'][:2]-env.reference();p=acc.rows[-1];physical.append(np.r_[env.data.time-start_t,p[1]+p[2],sum(p[3:8]),p[8],sum(p[9:14]),p[14],p[15],p[17:19],p[22:26],ep,s['rpy'][:2],env.data.actuator_force[env.motor_ids]])
 f.sample_hook=sample
 while f.steps<f.max_steps:
  action=np.zeros(3);t=float(f.data.time)-start_t
  if axis is not None and t>=onset-1e-9 and t<onset+.06-1e-9:action[axis]=.1*sign
  o,r,term,trunc,info=f.step(action)
  if first_step is None:first_step=f.data.qpos.copy()
  if term or trunc:break
 a=np.array(physical);s=kinematics(f);strict=bool(info['termination_reason']=='TIME_LIMIT' and np.linalg.norm(s['rpy'][:2])<.12 and abs(a[:,17:20]).max()<9.9);return dict(rows=a,first_qpos=first_step,end_qpos=f.data.qpos.copy(),end_qvel=f.data.qvel.copy(),energy=acc.summary(),strict=strict,reason=info['termination_reason'])

# physical rows: dt,E,Pin,Pd,Win,D,residual,accWx,accWy,qx,qy,vx,vy,epx,epy,roll,pitch,tau0..2
SCHEMA=['dt','E','Pin','Pd','Win','D','residual','accWx','accWy','qx','qy','vx','vy','epx','epy','roll','pitch','tau0','tau1','tau2']

def run(job):
 case,seed=job;begin=time.monotonic();e=EnergyEnv(variant='RP',backbone='V2',scenario=case,duration=5.,frozen=False);e.reset(seed=seed);seen=set();records=[];checks=[]
 while e.data.time<2.6-1e-9 and len(seen)<8:
  q,v=e.fluid_state();A,phi,_=polar(np.r_[q,v]);axis=int(np.argmax(A));angle=(float(phi[axis])+np.pi)%(2*np.pi)-np.pi;b=int(np.floor((angle+np.pi)/(2*np.pi)*8));s=kinematics(e);eligible=bool(e.data.time>=.6-1e-9 and A[axis]>=.001 and np.linalg.norm(q)<=.006 and np.linalg.norm(s['rpy'][:2])<=.02 and abs(e.base_action).max()<=.8);first=None
  if eligible and b not in seen:
   seen.add(b);folder=ROOT/'phase_actions'/case/str(seed)/f'bin{b}';folder.mkdir(parents=True,exist_ok=False);hist=list(e.frame_history);features=np.concatenate([hist[max(0,len(hist)-1-i*25)] for i in range(10)]);native=snapshot(e);metadata=dict(case=case,source_seed=seed,source_time_s=float(e.data.time),phase_bin=b,phase_rad=angle,dominant_axis=axis,latent=np.r_[q,v].tolist(),amplitudes=A.tolist(),observation=e.observation().tolist(),history=features.tolist(),qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),controller=encode(e.backbone),model_hash=e.model_hash)
   base=branch(e);first=base['first_qpos'];np.savez_compressed(folder/'baseline.npz',rows=base['rows'],schema=np.array(SCHEMA),native_initial_state=native)
   effects=[]
   for motor in range(3):
    for onset in [0.,.08]:
     for sign in [-1,1]:
      r=branch(e,motor,sign,onset);a=r['rows'];d=a-base['rows'];assert a.shape==base['rows'].shape;path=folder/f'motor{motor}_sign{sign}_onset{onset}.npz';np.savez_compressed(path,rows=a,delta=d,schema=np.array(SCHEMA));effect=dict(motor=motor,sign=sign,onset_s=onset,duration_s=.06,raw_amplitude=.1,delta_E_end_J=float(d[-1,1]),delta_E_mean_J=float(d[:,1].mean()),delta_work_J=float(d[-1,4]),delta_position_RMS_m=float(np.sqrt(np.mean(np.sum(a[:,13:15]**2,axis=1)))-np.sqrt(np.mean(np.sum(base['rows'][:,13:15]**2,axis=1)))),delta_tilt_RMS_rad=float(np.sqrt(np.mean(a[:,15:17]**2))-np.sqrt(np.mean(base['rows'][:,15:17]**2))),delta_tau_RMS_Nm=float(np.sqrt(np.mean(a[:,17:20]**2))-np.sqrt(np.mean(base['rows'][:,17:20]**2))),strict=r['strict'],reason=r['reason'],path=str(path));effects.append(effect)
   metadata['baseline_path']=str(folder/'baseline.npz');metadata['effects']=effects
   with (folder/'state.json').open('x') as f:json.dump(metadata,f,indent=2,allow_nan=False)
   records.append(metadata)
  _,_,t,u,info=e.step(np.zeros(3))
  if first is not None:
   gap=float(abs(e.data.qpos-first).max());checks.append(gap);assert gap<=1e-10,('Forkcontroller/nativecontinuationmismatch',case,seed,gap)
  if t or u:break
  if time.monotonic()-begin>120:raise TimeoutError('Causalpersource120swallbudget')
 e.close();print('CAUSAL',case,seed,'bins',len(seen),'maxgap',max(checks) if checks else None,flush=True);return records,checks,time.monotonic()-begin

def main():
 start=time.monotonic();records=[];checks=[];times=[];jobs=[(case,seed) for case in ['B1_displacement','B2_velocity'] for seed in CAUSAL_SEEDS]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(run,j) for j in jobs]):r,c,t=f.result();records+=r;checks+=c;times.append(t)
 save('phase_action_dataset.json',dict(states=records,source_trajectories=len(jobs),fork_continuation_max_gap=max(checks),wall_s=time.monotonic()-start,branch_native_steps=sum(len(np.load(s['baseline_path'])['rows'])*(1+len(s['effects'])) for s in records),scope='Same fullnative/controller/history statewithinall13branches; onlyresidualchanged. Acrossphasebinsotherstatesnaturallyvary, sophaseinteractionisconditional, notisolateddo(phase).'))
 print('CAUSAL_DATASET',len(records),'states',flush=True)
if __name__=='__main__':main()
