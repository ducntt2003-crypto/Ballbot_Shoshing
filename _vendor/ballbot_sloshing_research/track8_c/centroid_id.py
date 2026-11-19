"""Augmented native+controller one-macro Jacobian, not FFT pole guessing."""
import copy,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np,mujoco
from .common import *
from .native import restore_snapshot
from ballbot_sloshing_research.phase5.causal import fork
from ballbot_fast_rl.outer_loop.control import MIX
def layout(e):
 excluded=set(range(e.ball_d+3,e.ball_d+6));keep=[i for i in range(e.model.nv) if i not in excluded];spos=np.full(len(keep),.03)
 for j,k in enumerate(keep):
  if k in list(range(e.base_d,e.base_d+3))+list(range(e.ball_d,e.ball_d+3)):spos[j]=.1
  if k in e.slosh_v:spos[j]=.0084
  if k in e.wheel_dofs:spos[j]=1.
 sv=np.ones(e.model.nv);sv[e.base_d:e.base_d+3]=.1;sv[e.base_d+3:e.base_d+6]=.3;sv[e.ball_d:e.ball_d+3]=.1;sv[e.wheel_dofs]=2.;sv[e.slosh_v]=.1
 return keep,np.r_[spos,sv,[.1,.1,.01,.01,.04,.04,.1,.1,.1]]
def coordinates(e,anchor,keep):
 dq=np.zeros(e.model.nv);mujoco.mj_differentiatePos(e.model,dq,1.,anchor,e.data.qpos)
 velocity=e.data.qvel.copy();ball=e.model.body('ball').id;R=e.data.xmat[ball].reshape(3,3);offset=R@e.model.geom_pos[e.ball_geom];omega=R@velocity[e.ball_d+3:e.ball_d+6];velocity[e.ball_d+3:e.ball_d+6]=omega;velocity[e.ball_d:e.ball_d+3]+=np.cross(omega,offset);R0=np.zeros(9);mujoco.mju_quat2Mat(R0,anchor[e.ball_q+3:e.ball_q+7]);center0=anchor[e.ball_q:e.ball_q+3]+R0.reshape(3,3)@e.model.geom_pos[e.ball_geom];dq[e.ball_d:e.ball_d+3]=e.data.geom_xpos[e.ball_geom]-center0
 return np.r_[dq[keep],velocity,e.backbone.integral,e.backbone.inner.integral,e.backbone.tilt,e.base_action]
def perturb(f,j,value,keep):
 n=len(keep);nv=f.model.nv
 if j<n:
  dq=np.zeros(nv);dq[keep[j]]=value;mujoco.mj_integratePos(f.model,f.data.qpos,dq,1.)
 elif j<n+nv:
  k=j-n
  if f.ball_d+3<=k<f.ball_d+6:
   axis=k-f.ball_d-3;R=f.data.xmat[f.model.body('ball').id].reshape(3,3);f.data.qvel[f.ball_d+3:f.ball_d+6]+=R.T[:,axis]*value;dw=np.zeros(3);dw[axis]=value;f.data.qvel[f.ball_d:f.ball_d+3]-=np.cross(dw,R@f.model.geom_pos[f.ball_geom])
  else:f.data.qvel[k]+=value
 else:
  k=j-n-nv
  if k<2:f.backbone.integral[k]+=value
  elif k<4:f.backbone.inner.integral[k-2]+=value
  elif k<6:f.backbone.tilt[k-4]+=value
  else:f.base_action[k-6]+=value
 mujoco.mj_forward(f.model,f.data)
def identify(e,eps=1e-4):
 keep,scale=layout(e);n=len(scale);anchor=e.data.qpos.copy();A=np.zeros((n,n));B=np.zeros((n,2))
 for j in range(n):
  out=[]
  for sign in [-1,1]:
   f=fork(e);perturb(f,j,sign*eps*scale[j],keep);f.step(np.zeros(3));out.append(coordinates(f,anchor,keep)/scale)
  A[:,j]=(out[1]-out[0])/(2*eps)
 for j in range(2):
  out=[]
  for sign in [-1,1]:
   f=fork(e);f.step(-MIX[:,j]*sign*eps/.3);out.append(coordinates(f,anchor,keep)/scale)
  B[:,j]=(out[1]-out[0])/(2*eps)
 return A,B,keep,scale
def eigensummary(A,keep,nv,slosh):
 val,V=np.linalg.eig(A);out=[];indices=[keep.index(int(j)) for j in slosh]+[len(keep)+int(j) for j in slosh]
 for k,v in enumerate(val):
  freq=np.angle(v)/(.01*2*np.pi);rate=np.log(max(abs(v),1e-30))/.01
  if freq>.1 and freq<20:out.append(dict(index=k,frequency_Hz=float(freq),growth_per_s=float(rate),modal_participation=float(np.linalg.norm(V[indices,k])**2/np.linalg.norm(V[:,k])**2)))
 return out
def one(seed):
 out=[]
 for stamp in [5,10,14]:
  e=restore_snapshot(ROOT/'reproduction'/str(seed)/f'snapshot_{stamp}.pkl')
  try:
   start=time.monotonic();A,B,keep,scale=identify(e);nv=e.model.nv;cfg=read(ROOT/'controller_config.json');K=np.zeros((2,len(scale)));K[:,len(keep)+e.slosh_v]=-cfg['c_per_s']*np.linalg.inv(cfg['H'])*scale[len(keep)+e.slosh_v][None,:];v1=A+B@K;path=ROOT/f'centroid_model_{seed}_{stamp}.npz';np.savez_compressed(path,A=A,B=B,keep=keep,scale=scale,anchor=e.data.qpos,slosh_dofs=e.slosh_v)
   out.append(dict(seed=seed,time_s=stamp,path=str(path),baseline_modes=eigensummary(A,keep,nv,e.slosh_v),V1_feedback_modes=eigensummary(v1,keep,nv,e.slosh_v),wall_s=time.monotonic()-start,states=len(scale),native_steps=(2*len(scale)+4)*5))
  finally:e.close()
 return out
def main():
 save('centroid_identification_addendum.json',dict(reason='V1instantaneousworkfeedbackpassedE0-E2butfailedall3circleE3before30s. Instantaneousnegativepowerisnotclosedloopstability. Identifyactualaugmentedcontroller+plantone10msmacroJacobian beforeanyrevision; no gain sweep orphysicschanges.',V1_result='FAILED_E3',native_macro_dt=.01,physical_dt=.002,finite_difference_normalized_step=.0001,seeds=REPRO,times_s=[5,10,14],budget_s=120,features='Correctedchart: retainwheelangles(0.323mmeccentricity); ballangularvelocityWORLD, ballorientationexcludedonlywiththisinvariantvelocity. Allotherqvel + controllerI/tilt/storedaction. OlderrawchartmodelandfailedV2retainedbutnotusedasaccuratepoleproof.',scope='Localtrajectorylinearizations andcausalmacros, notglobalstabilityproof; polesvalidatedacross9states.'))
 with ProcessPoolExecutor(max_workers=3) as pool:groups=list(pool.map(one,REPRO))
 save('centroid_identification.json',dict(records=[r for g in groups for r in g],scope='Fullnativeaugmented10msJacobianfiniteperturbations; cyclicneutralanglesexcluded, otherneutralmodesretained. Not identified solelybyFFT.'));print('CENTROID_MODEL_IDENTIFIED',[(r['seed'],r['time_s'],r['baseline_modes'],r['V1_feedback_modes']) for g in groups for r in g],flush=True)
if __name__=='__main__':main()
