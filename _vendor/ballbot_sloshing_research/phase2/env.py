from collections import deque
import numpy as np,gymnasium as gym
from ballbot_sloshing_research.envs.ballbot_sloshing_env import BallBotSloshingEnv
from ballbot_fast_rl.robust_control.v2.variant import ConstraintAwareCascade
from ballbot_fast_rl.robust_control.v3.variant import BandwidthCascade
from ballbot_fast_rl.robust_control.config import GAINS
from ballbot_fast_rl.outer_loop.control import kinematics

SCENARIOS=['nominal','B1_displacement','B2_velocity','B3_accelerate_stop','B4_push','B5_tracking']
class Phase2Env(BallBotSloshingEnv):
 def __init__(self,backbone='V2',observation='O2',reward='R2',scenario='nominal',duration=5.,policy_steps=5,frozen=False,level=2):
  self.backbone_name=backbone;self.contract=observation;self.reward_name=reward;self.scenario_config=scenario;self.active_scenario=scenario;self.policy_steps=policy_steps;self.frozen=frozen;self.curriculum_level=level;self.phase_rng=np.random.default_rng(0)
  parent_obs='O1' if observation=='O1' else 'O2';super().__init__(mode='liquid',observation=parent_obs,residual=True,slosh_penalty=0,duration=duration,initial_slosh=0,tilt=.003,physics_dt=.002)
  self.backbone=ConstraintAwareCascade(GAINS) if backbone=='V2' else BandwidthCascade(GAINS);self.frame_history=deque(maxlen=71 if observation=='LEGACY' else 226);self.obs_dim=184 if observation=='LEGACY' else 23 if observation=='O1' else 230;self.observation_space=gym.spaces.Box(-10,10,(self.obs_dim,),np.float32);self.obs_contract='P2_LEGACY_O2_V1' if observation=='LEGACY' else 'P2_'+observation+'_V1';self.policy_dt=.002*policy_steps
 def target(self):
  t=float(self.data.time);p=np.zeros(2);v=np.zeros(2);a=np.zeros(2)
  if self.active_scenario=='B3_accelerate_stop':
   u=np.clip((t-1)/1.5,0,1);p[0]=.02*(10*u**3-15*u**4+6*u**5)
   if 0<u<1:v[0]=.02/1.5*(30*u**2-60*u**3+30*u**4);a[0]=.02/1.5**2*(60*u-180*u**2+120*u**3)
  elif self.active_scenario=='B5_tracking':p[0]=.01*(1-np.cos(np.pi*t));v[0]=.01*np.pi*np.sin(np.pi*t);a[0]=.01*np.pi**2*np.cos(np.pi*t)
  return self.origin[:2]+p,v,a
 def reference(self):return self.target()[0]
 def frame(self,s=None):
  if self.contract=='LEGACY':return BallBotSloshingEnv.frame(self,s)
  s=kinematics(self) if s is None else s;ref,vr,_=self.target();g=s['R'].T@np.array([0.,0.,-1]);g[:2]/=.05
  return np.r_[g,s['omega_B']/.3,(s['p'][:2]-ref)/.02,(s['v'][:2]-vr)/.05,self.data.qvel[self.wheel_dofs]/2,self.previous_action/.1,vr/.05,self.base_action/.1,np.sin(s['rpy'][2]),np.cos(s['rpy'][2])].astype(np.float32)
 def observation(self,s=None):
  if self._building_reset or not self.frame_history:return np.zeros(getattr(self,'obs_dim',23),np.float32)
  hist=list(self.frame_history)
  if self.contract=='O1':out=hist[-1]
  else:
   count,stride=(8,10) if self.contract=='LEGACY' else (10,25);out=np.concatenate([hist[max(0,len(hist)-1-i*stride)] for i in range(count)])
  return np.clip(out,-10,10).astype(np.float32)
 def reset(self,*,seed=None,options=None):
  if seed is not None:self.phase_rng=np.random.default_rng(seed+880000)
  if self.scenario_config=='curriculum':
   choices=['nominal'] if self.curriculum_level==1 else ['nominal','B1_displacement','B2_velocity'] if self.curriculum_level==2 else ['nominal','B1_displacement','B2_velocity','B4_push'] if self.curriculum_level==3 else SCENARIOS;self.active_scenario=str(self.phase_rng.choice(choices))
  else:self.active_scenario=self.scenario_config
  phase=float(self.phase_rng.uniform(0,2*np.pi));direction=np.array([np.cos(phase),np.sin(phase)]);q=np.zeros(2);v=np.zeros(2)
  if self.active_scenario=='nominal':q=.0003*direction
  if self.active_scenario=='B1_displacement':q=.003*direction
  if self.active_scenario=='B2_velocity':v=self.pfluid.omega*.003*direction
  merged=dict(options or {});merged.setdefault('slosh_q',q);merged.setdefault('slosh_v',v);self.cfg.update(push_N=.2 if self.active_scenario=='B4_push' else 0.,push_start_s=1.,push_duration_s=.08)
  obs,info=super().reset(seed=seed,options=merged);self.push_W=np.array([self.cfg['push_N'],0,0]);info.update(scenario=self.active_scenario,initial_slosh_q=self.fluid_state()[0],initial_slosh_v=self.fluid_state()[1],disturbance_schedule=dict(force_W_N=self.push_W.tolist(),start_s=1,duration_s=.08),policy_contract=self.obs_contract,policy_dt=self.policy_dt,backbone=self.backbone_name);return obs,info
 def step(self,action):
  action=np.asarray(action,float)
  if action.shape!=(3,) or not np.isfinite(action).all():raise ValueError('Finite normalized3residual commands required')
  action=np.zeros(3) if self.frozen else np.clip(action,-1,1);totalreward=0.;last=None;terms=[];micro=[]
  for _ in range(self.policy_steps):
   if getattr(self,'pre_sample_hook',None):self.pre_sample_hook(self,action)
   before=self.previous_action.copy();obs,_,term,trunc,info=super().step(action);s=kinematics(self);q,v=self.fluid_state();ref,vr,_=self.target();ep=s['p'][:2]-ref;ev=s['v'][:2]-vr;applied=info['applied_action'];E0=.5*self.pfluid.stiffness*.003**2
   t=dict(alive=1.,tilt=.5*float(s['rpy'][:2]@s['rpy'][:2])/.03**2,omega=.1*float(s['omega_B']@s['omega_B'])/.3**2,position=.5*float(ep@ep)/.02**2,velocity=.1*float(ev@ev)/.05**2,slosh=(.2 if self.reward_name=='R2' else 0)*self.slosh_energy()/E0,effort=.01*float((10*applied)@(10*applied)),variation=.002*float((applied-before)@(applied-before))/.05**2)
   totalreward+=.002*(t['alive']-sum(value for key,value in t.items() if key!='alive'))-(1. if term else 0);terms.append(t);micro.append(dict(t=float(self.data.time),action=applied.copy(),q=q.copy(),qd=v.copy(),energy=self.slosh_energy()));last=(obs,term,trunc,info)
   if getattr(self,'sample_hook',None):self.sample_hook(self)
   if term or trunc:break
  obs,term,trunc,info=last;info.update(phase2_reward_terms={key:float(np.mean([t[key] for t in terms])) for key in terms[0]},micro=micro,scenario=self.active_scenario,reference_velocity=vr.copy(),residual=action.copy());return obs,float(totalreward),term,trunc,info
