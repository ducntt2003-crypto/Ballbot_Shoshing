from collections import deque
import numpy as np,mujoco,gymnasium as gym
from ballbot_fast_rl.position_hold.core import PositionEnv
from ballbot_fast_rl.outer_loop.control import kinematics
from ballbot_fast_rl.robust_control.v3.variant import BandwidthCascade
from ..fluid.reduced_order import ModalParameters,rk4
from ..coupling.model import build_model,reaction_wrench

class BallBotSloshingEnv(PositionEnv):
 """Native passive liquid model, robot-only observations, optional residual PPO.

 Actor receives ideal robot sensors + causal backbone command; never fluid truth.
 O2 stacks robot-only history. Reward may use simulator fluid energy explicitly.
 """
 metadata={'render_modes':['rgb_array'],'render_fps':50}
 def __init__(self,mode='liquid',observation='O1',residual=True,slosh_penalty=.1,duration=3.,volume=.0005,zeta=.02,frequency_scale=1.,initial_slosh=.001,tilt=.015,push=0.,tracking=False,training=False,physics_dt=.002,mount_offset=(0,0,0)):
  self.liquid_mode=mode;self.obs_contract=observation;self.residual=residual;self.slosh_weight=slosh_penalty;self.pfluid=ModalParameters(volume=volume,zeta=zeta,frequency_scale=frequency_scale);self.slosh_reset=initial_slosh;self.track_reference=tracking;self.training_mode=training;self.backbone=BandwidthCascade();self.base_action=np.zeros(3);self.frame_history=deque(maxlen=71);self._building_reset=True
  super().__init__(dict(duration_s=duration,physics_dt=physics_dt,control_dt=.002,tilt_reset_rad=tilt,push_N=push,push_start_s=1.,push_duration_s=.08))
  mode_for_builder='rigid' if mode=='oneway_diagnostic' else mode
  self.model,self.xml_path=build_model(self.tmp.name,self.pfluid,mode_for_builder,physics_dt,mount_offset);self.data=mujoco.MjData(self.model)
  for name,attr in [('base','base')]:setattr(self,attr,self.model.body(name).id)
  self.base_joint=self.model.joint('base_free_joint').id;self.ball_joint=self.model.joint('ball_free_joint').id;self.base_q=int(self.model.jnt_qposadr[self.base_joint]);self.ball_q=int(self.model.jnt_qposadr[self.ball_joint]);self.base_d=int(self.model.jnt_dofadr[self.base_joint]);self.ball_d=int(self.model.jnt_dofadr[self.ball_joint]);self.motor_ids=np.array([self.model.actuator(f'motor_{i}').id for i in range(3)]);self.wheel_ids=np.array([self.model.joint(f'wheel_joint_{i}').id for i in range(3)]);self.wheel_dofs=self.model.jnt_dofadr[self.wheel_ids].astype(int);self.wheel_qpos=self.model.jnt_qposadr[self.wheel_ids].astype(int);self.ball_geom=self.model.geom('the_ball').id;self.ground_geom=self.model.geom('terrain').id;self.tank_id=self.model.body('water_tank').id;self.limits=self.model.actuator_ctrlrange[self.motor_ids].copy();self.torque_scale=self.limits[:,1];self.reference_mass=self.model.body_mass.copy();self.reference_inertia=self.model.body_inertia.copy()
  self.fluid_ids=[self.model.body(n).id for n in ['fluid_fixed','fluid_modal'] if self.model.body(n).id>=0] if mode not in ['empty','rigid','oneway_diagnostic'] else ([self.model.body('fluid_fixed').id] if mode!='empty' else [])
  self.slosh_joints=np.array([self.model.joint(n).id for n in ['slosh_x','slosh_y']]) if mode=='liquid' else np.array([],int);self.slosh_q=self.model.jnt_qposadr[self.slosh_joints].astype(int);self.slosh_v=self.model.jnt_dofadr[self.slosh_joints].astype(int);self.shadow=np.zeros(4);self.obs_dim=23 if observation=='O1' else 184;self.observation_space=gym.spaces.Box(-10,10,(self.obs_dim,),np.float32);self._building_reset=False
  import hashlib
  self.model_hash=hashlib.sha256(self.xml_path.read_bytes()).hexdigest()
 def fluid_state(self):
  if self.liquid_mode=='liquid':return self.data.qpos[self.slosh_q].copy(),self.data.qvel[self.slosh_v].copy()
  if self.liquid_mode=='oneway_diagnostic':return self.shadow[:2].copy(),self.shadow[2:].copy()
  return np.zeros(2),np.zeros(2)
 def slosh_energy(self):
  q,v=self.fluid_state();return self.pfluid.energy(q,v)
 def reference(self):
  t=float(self.data.time)
  return self.origin[:2]+(np.array([.015*np.sin(2*np.pi*.25*t),0.]) if self.track_reference else 0.)
 def _prepare_control(self):
  s=kinematics(self);self.backbone.reference=self.reference().copy();self.base_action=self.backbone.act(s,True)[0];return s
 def frame(self,s=None):
  s=kinematics(self) if s is None else s;ref=self.reference();return np.r_[s['R'].T@[0,0,-1],s['omega_B']/5,(s['p'][:2]-ref)/.1,s['v'][:2]/.3,self.data.qvel[self.wheel_dofs]/10,self.previous_action,(ref-self.origin[:2])/.1,self.base_action,np.sin(s['rpy'][2]),np.cos(s['rpy'][2])].astype(np.float32)
 def observation(self,s=None):
  if self._building_reset:return np.zeros(getattr(self,'obs_dim',23),np.float32)
  if not self.frame_history:return np.zeros(self.obs_dim,np.float32)
  if self.obs_contract=='O1':obs=self.frame_history[-1]
  else:
   hist=list(self.frame_history);obs=np.concatenate([hist[max(0,len(hist)-1-i*10)] for i in range(8)])
  return np.clip(obs,-10,10).astype(np.float32)
 def reset(self,*,seed=None,options=None):
  self._building_reset=True;super().reset(seed=seed,options=options);self._building_reset=False;options=options or {};self.shadow[:]=0
  q=np.asarray(options.get('slosh_q',self.np_random.uniform(-self.slosh_reset,self.slosh_reset,2)));v=np.asarray(options.get('slosh_v',[0.,0.]))
  if self.liquid_mode=='liquid':self.data.qpos[self.slosh_q]=q;self.data.qvel[self.slosh_v]=v
  if self.liquid_mode=='oneway_diagnostic':self.shadow[:]=np.r_[q,v]
  if 'velocity' in options:
   for dof in [self.base_d,self.ball_d]:self.data.qvel[dof:dof+2]+=options['velocity']
  mujoco.mj_forward(self.model,self.data);s=kinematics(self);self.backbone.reset(s,self.reference());self._prepare_control();self.frame_history.clear();self.frame_history.append(self.frame(s));self.previous_action=np.zeros(3);self.last_reward_terms={};return self.observation(),dict(model_hash=self.model_hash,liquid_model=self.pfluid.metadata(),observation_contract=self.obs_contract+'_IDEAL_ROBOT_ONLY',initial_qpos=self.data.qpos.copy(),initial_qvel=self.data.qvel.copy())
 def step(self,action):
  action=np.asarray(action,float)
  if action.shape!=(3,) or not np.isfinite(action).all():raise ValueError('Expected finite3motor commands')
  action=np.clip(action,-1,1);total=np.clip(self.base_action+.03*action,-1,1) if self.residual else action;before=self.previous_action.copy();v0=kinematics(self)['v'].copy()
  _,_,term,trunc,info=super().step(total);s=kinematics(self)
  if self.liquid_mode=='oneway_diagnostic':
   # Diagnostic prescribed-tank model: post-step specific acceleration at modal origin,
   # with rotating-frame Coriolis/Euler/centripetal terms. No shadow load is applied.
   mujoco.mj_rnePostConstraint(self.model,self.data);specific=np.zeros(6);body=self.model.body('fluid_fixed').id;mujoco.mj_objectAcceleration(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,body,specific,0);R=self.data.xmat[body].reshape(3,3);forcing=-(R.T@specific[3:])[:2];alpha=R.T@specific[:3];omega=s['omega_B']
   def shadow_rhs(t,x):
    q3=np.r_[x[:2],0.];v3=np.r_[x[2:],0.];rotating=2*np.cross(omega,v3)+np.cross(alpha,q3)+np.cross(omega,np.cross(omega,q3));return np.r_[x[2:],-self.pfluid.omega**2*x[:2]-2*self.pfluid.zeta*self.pfluid.omega*x[2:]+forcing-rotating[:2]]
   self.shadow=rk4(shadow_rhs,self.data.time-self.dt,self.shadow,self.dt)
  q,v=self.fluid_state();domain=np.linalg.norm(q)>.12*self.pfluid.radius or np.linalg.norm(s['rpy'][:2])>.12
  if domain and not term:term=True;trunc=False;info.update(termination_reason='ROM_DOMAIN',success=False)
  if not np.isfinite(np.r_[q,v]).all():term=True;trunc=False;info.update(termination_reason='INVALID_FLUID',success=False)
  ep=s['p'][:2]-self.reference();E=self.slosh_energy();E0=.5*self.pfluid.stiffness*.003**2
  terms=dict(alive=1.,tilt=.5*float(s['rpy'][:2]@s['rpy'][:2])/.05**2,omega=.1*float(s['omega_B']@s['omega_B'])/.5**2,position=.5*float(ep@ep)/.05**2,velocity=.1*float(s['v'][:2]@s['v'][:2])/.1**2,slosh=self.slosh_weight*E/max(E0,1e-9),effort=.01*float(total@total),action_rate=.002*float((total-before)@(total-before))/.05**2)
  reward=self.dt*(terms['alive']-sum(v for k,v in terms.items() if k!='alive'))-(1. if term else 0.);self.last_reward_terms=terms;self._prepare_control();self.frame_history.append(self.frame(s));info.update(slosh_q=q,slosh_v=v,slosh_energy_J=E,position_error_W=ep,applied_action=total.copy(),reward_terms=terms,ROM_domain_valid=not domain)
  return self.observation(),float(reward),bool(term),bool(trunc),info
 def diagnostics(self):
  s=kinematics(self);q,v=self.fluid_state();W=reaction_wrench(self.model,self.data,self.fluid_ids,self.data.xpos[self.tank_id]) if self.fluid_ids else np.zeros(6);mass=sum(self.model.body_mass[i] for i in self.fluid_ids);com=sum(self.model.body_mass[i]*self.data.xipos[i] for i in self.fluid_ids)/mass if mass else self.data.xpos[self.tank_id].copy()
  return dict(time=float(self.data.time),rpy=s['rpy'],p=s['p'],v=s['v'],ep=s['p'][:2]-self.reference(),q=q,qd=v,energy=self.slosh_energy(),wrench=W,fluid_COM_W=com,total_mass=float(self.model.body_mass.sum()),omega=self.data.qvel[self.wheel_dofs].copy(),tau=self.data.actuator_force[self.motor_ids].copy())
