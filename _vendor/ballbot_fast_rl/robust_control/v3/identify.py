"""Native RK4 finite differences and independent tiny-excitation validation; no physics edits."""
import json
import numpy as np,mujoco
from scipy.linalg import eig
from ballbot_fast_rl.robust_control.v2.instrument import AuditWorld,rp_rate
from ballbot_fast_rl.robust_control.v2.variant import ConstraintAwareCascade
from ballbot_fast_rl.outer_loop.control import MIX,kinematics
from .config import ROOT,GAINS
DT=.002
SPEC=int(mujoco.mjtState.mjSTATE_INTEGRATION)
def snapshot(e):
 s=np.zeros(mujoco.mj_stateSize(e.model,SPEC));mujoco.mj_getState(e.model,e.data,s,SPEC);return s
class Flow:
 def __init__(self,e,s):self.e=e;self.m=e.model;self.s=s;self.d=mujoco.MjData(self.m)
 def restore(self,dq=None,dv=None):
  mujoco.mj_setState(self.m,self.d,self.s,SPEC)
  if dq is not None:mujoco.mj_integratePos(self.m,self.d.qpos,dq,1.)
  if dv is not None:self.d.qvel[:]+=dv
  mujoco.mj_forward(self.m,self.d)
 def output(self):
  original=self.e.data;self.e.data=self.d
  try:s=kinematics(self.e);return np.r_[s['rpy'][:2],rp_rate(s)]
  finally:self.e.data=original
 def advance(self,u):
  self.d.ctrl[self.e.motor_ids]=-10*(MIX@u);mujoco.mj_step(self.m,self.d);mujoco.mj_forward(self.m,self.d);return self.output()
def settled(seed):
 w=AuditWorld(dict(sensor=dict(mode='perfect')),seed,2.);c=ConstraintAwareCascade(GAINS);c.reset(w.measurement.control_state(),w.ref[:2]);a=np.zeros(3)
 for _ in range(500):a,_=c.act(w.measurement.control_state());w.step(a)
 return w,(MIX.T@a)/1.5

def main():
 out=ROOT/'identification';out.mkdir(exist_ok=False);w,u=settled(150000);e=w.env;flow=Flow(e,snapshot(e));flow.restore();flow.advance(u);qbase=flow.d.qpos.copy();vbase=flow.d.qvel.copy();nv=e.model.nv;n=2*nv;eps=1e-5
 def delta():
  dq=np.zeros(nv);mujoco.mj_differentiatePos(e.model,dq,1.,qbase,flow.d.qpos);return np.r_[dq,flow.d.qvel-vbase]
 A=np.zeros((n,n));B=np.zeros((n,2));C=np.zeros((4,n))
 for j in range(n):
  xx=[];yy=[]
  for sign in [-1,1]:
   dq=np.zeros(nv);dv=np.zeros(nv);(dq if j<nv else dv)[j%nv]=sign*eps;flow.restore(dq,dv);yy.append(flow.output());flow.advance(u);xx.append(delta())
  A[:,j]=(xx[1]-xx[0])/(2*eps);C[:,j]=(yy[1]-yy[0])/(2*eps)
 for j in range(2):
  xx=[]
  for sign in [-1,1]:
   du=np.zeros(2);du[j]=sign*eps;flow.restore();flow.advance(u+du);xx.append(delta())
  B[:,j]=(xx[1]-xx[0])/(2*eps)
 # Independent80ms cloned-state excitation, reference nominal evolves under same held command.
 validations=[]
 for seed in [150010,150011]:
  v,uv=settled(seed);s=snapshot(v.env);nom=Flow(v.env,s);pert=Flow(v.env,s);nom.restore();pert.restore();xp=np.zeros(n);truth=[];pred=[];rng=np.random.default_rng(seed)
  for k in range(40):
   du=rng.choice([-1,1],2)*.0005;yn=nom.advance(uv);yp=pert.advance(uv+du);xp=A@xp+B@du;truth.append(yp-yn);pred.append(C@xp)
  truth=np.array(truth);pred=np.array(pred);err=float(np.sqrt(np.mean((pred-truth)**2)));scale=float(np.sqrt(np.mean(truth**2)));validations.append(dict(seed=seed,horizon_s=.08,RMSE=err,response_RMS=scale,NRMSE_ratio=err/max(scale,1e-12),max_rate_delta_rad_s=float(abs(truth[:,2:]).max())))
  np.savetxt(out/f'validation_{seed}.csv',np.c_[np.arange(40)*DT,truth,pred],delimiter=',',header='t,thetaR,thetaP,omegaR,omegaP,predThetaR,predThetaP,predOmegaR,predOmegaP',comments='');v.close()
 np.savez(out/'native_RK4_model.npz',A=A,B=B,C=C,nominal_u=u)
 (out/'summary.json').write_text(json.dumps(dict(nstate=n,model_hash=e.model_hash,integrator=int(e.model.opt.integrator),dt=DT,eps=eps,seed=150000,validation=validations,one_step_rate_input_gain=(C[2:]@B).tolist(),limitations='Fixed warmstart, fixed nominal contact linearization; no claimed global validity or exact phase margin. Validation held command for80ms on cloned nativeRK4 states, not a long-horizon robot test.',native_transitionFD='Unsupported forRK4; manually central-differenced native step, never changed integrator.'),indent=2));print(validations,flush=True);w.close()
if __name__=='__main__':main()
