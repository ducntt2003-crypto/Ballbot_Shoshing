"""Relative-modal energy decomposition, no new forces applied to the physics."""
from copy import copy
import numpy as np,mujoco
from ballbot_fast_rl.outer_loop.control import kinematics
from ballbot_fast_rl.robust_control.v3.identify import snapshot,SPEC

POWER_NAMES=['translation','gravity','coriolis','euler','centripetal','damping']
def sample(e):
 mujoco.mj_rnePostConstraint(e.model,e.data);body=e.model.body('fluid_fixed').id;six=np.zeros(6);mujoco.mj_objectAcceleration(e.model,e.data,mujoco.mjtObj.mjOBJ_BODY,body,six,0);R=e.data.xmat[body].reshape(3,3);q,v=e.fluid_state();q3=np.r_[q,0.];v3=np.r_[v,0.];state=kinematics(e);omega=state['omega_B'];alpha=R.T@six[:3];acc=six[3:]+e.model.opt.gravity
 forces=np.array([-(R.T@acc)[:2],(R.T@e.model.opt.gravity)[:2],-2*np.cross(omega,v3)[:2],-np.cross(alpha,q3)[:2],-np.cross(omega,np.cross(omega,q3))[:2]]);p=e.pfluid;powers=np.r_[p.moving*(forces@v),p.damping*float(v@v)];qdd=e.data.qacc[e.slosh_v];identity=p.moving*float(v@qdd)+p.stiffness*float(q@v)-powers[:5].sum()+powers[5]
 return dict(powers=powers,Pin=float(powers[:5].sum()),KE=.5*p.moving*float(v@v),PE=.5*p.stiffness*float(q@q),acceleration_W=acc,relative_forcing=forces.sum(axis=0),identity_W=identity,tilt_norm=float(np.linalg.norm(state['rpy'][:2])))

class Accounting:
 def __init__(self,e):
  self.e=e;self.clone=copy(e);self.clone.data=mujoco.MjData(e.model);self.initial=e.slosh_energy();self.work=np.zeros(6);self.rows=[];self.pre=None;self.first_invalid=None
 def before(self,e,action):
  mujoco.mj_setState(e.model,self.clone.data,snapshot(e),SPEC);self.clone.data.ctrl[e.motor_ids]=-10*np.clip(e.base_action+.03*action,-1,1);mujoco.mj_forward(e.model,self.clone.data);self.pre=sample(self.clone)
 def after(self,e):
  end=sample(e);self.work+=.001*(self.pre['powers']+end['powers']);E=end['KE']+end['PE'];res=E-self.initial-self.work[:5].sum()+self.work[5];q,v=e.fluid_state();self.first_invalid=len(self.rows) if self.first_invalid is None and (np.linalg.norm(q)>.0084 or end['tilt_norm']>.12) else self.first_invalid;self.rows.append(np.r_[e.data.time,end['KE'],end['PE'],end['powers'],self.work,res,end['identity_W'],end['acceleration_W'],end['relative_forcing'],q,v])
 def summary(self):
  a=np.array(self.rows);n=self.first_invalid if self.first_invalid is not None else len(a);b=a[:n];work=b[-1,9:15];scale=max(self.initial,abs(work[:5].sum()),work[5],1e-10)
  return dict(initial_energy_J=self.initial,valid_duration_s=float(b[-1,0]),work_components_J={name:float(work[j]) for j,name in enumerate(POWER_NAMES)},input_work_J=float(work[:5].sum()),damping_loss_J=float(work[5]),delta_energy_J=float(b[-1,1]+b[-1,2]-self.initial),residual_max_J=float(abs(b[:,15]).max()),relative_residual_max=float(abs(b[:,15]).max()/scale),identity_residual_max_W=float(abs(b[:,16]).max()),note='Per2msheld-command endpoint quadrature; no trapezoid across discontinuous command changes. Strict valid prefix; total generalized input includes gravity/rotating frame terms, not motor-only work.')
SCHEMA=['t','KE','PE']+[f'P_{n}' for n in POWER_NAMES]+[f'W_{n}' for n in POWER_NAMES]+['residual','identity','accWx','accWy','accWz','forcingTx','forcingTy','qx','qy','vx','vy']

class RefinedAccounting(Accounting):
 """Simpson in each held2msinterval; midpoint from independent1msmodelcopy."""
 def __init__(self,e):
  super().__init__(e)
  import copy as copying
  self.mid=copying.copy(e);self.mid.model=copying.deepcopy(e.model);self.mid.model.opt.timestep=.001;self.mid.data=mujoco.MjData(self.mid.model);self.midpoint=None
 def before(self,e,action):
  super().before(e,action);mujoco.mj_setState(self.mid.model,self.mid.data,snapshot(e),SPEC);self.mid.data.ctrl[e.motor_ids]=self.clone.data.ctrl[e.motor_ids];mujoco.mj_forward(self.mid.model,self.mid.data);mujoco.mj_step(self.mid.model,self.mid.data);mujoco.mj_forward(self.mid.model,self.mid.data);self.midpoint=sample(self.mid);assert e.model.opt.timestep==.002
 def after(self,e):
  end=sample(e);self.work+=.002/6*(self.pre['powers']+4*self.midpoint['powers']+end['powers']);E=end['KE']+end['PE'];res=E-self.initial-self.work[:5].sum()+self.work[5];q,v=e.fluid_state();self.first_invalid=len(self.rows) if self.first_invalid is None and (np.linalg.norm(q)>.0084 or end['tilt_norm']>.12) else self.first_invalid;self.rows.append(np.r_[e.data.time,end['KE'],end['PE'],end['powers'],self.work,res,end['identity_W'],end['acceleration_W'],end['relative_forcing'],q,v])
 def summary(self):
  d=super().summary();d['note']='Simpson within each held2mscommand using independent1msmidpointcopy; original nativeRK4dt unchanged. Not exact internal MuJoCoRKstagework; compare stored endpoint baseline and local shadow sensitivity.';return d
