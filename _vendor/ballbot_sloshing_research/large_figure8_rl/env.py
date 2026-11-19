"""LARGE8_PRIVILEGED_V4_RESIDUAL_PPO_V1. One process, one native callback owner."""
import copy
import gymnasium as gym
import mujoco
from scipy.spatial.transform import Rotation
from .common import *
from .isolation import ContactSession,clone,native_state,python_state,same
from ballbot_sloshing_research.track8.env import TrackEnv
from ballbot_sloshing_research.track8_c.centroid_controller import CentroidModalFeedback
from ballbot_sloshing_research.track8_i.observer import StepObserver
from ballbot_sloshing_research.track8_l.logger import ActionLogger
from ballbot_sloshing_research.track8_l import adapter as A
from ballbot_fast_rl.outer_loop.control import kinematics

class Guard(Exception):pass

def classify(q,tilt,xy,tau,finite=True,warnings=False,contact=False):
 if contact:return 'CONTACT_INVALID'
 if not finite or warnings:return 'NUMERICAL_INVALID'
 if q>.0084:return 'ROM_INVALID'
 if tilt>.12:return 'BALANCE_INVALID'
 if tau>10+1e-12:return 'ACTUATOR_INVALID'
 if xy>.25:return 'TASK_TRACKING_INVALID'
 return None

def reward_components(x,delta,previous,cfg):
 z=cfg['scales'];w=cfg['weights'];x=np.asarray(x);terms=dict(position=float(np.sum((x[1:3]-x[5:7])**2)/z['position_m']**2),velocity=float(np.sum((x[3:5]-x[7:9])**2)/z['velocity_m_s']**2),balance=float(np.sum(x[11:13]**2)/z['tilt_rad']**2),slosh=float(x[18]/z['energy_J']),effort=float(np.mean(x[19:22]**2)/z['torque_Nm']**2),variation=float(np.mean((np.asarray(delta)-previous)**2)/z['delta']**2));C=sum(w[k]*v for k,v in terms.items());return terms,float(np.exp(-C))

class TrackingPPOEnv(gym.Env):
 metadata={'render_modes':[]}
 def __init__(self,terrain_seed=2244001,duration=30.,capture=False,h=.002):
  self.cfg=read(ROOT/'pilot_config.json');self.reward_cfg=read(ROOT/'reward_config.json');self.action_space=gym.spaces.Box(-1,1,(3,),np.float32);self.observation_space=gym.spaces.Box(-10,10,(31,),np.float32);self.capture=capture;self.session=ContactSession();self.session.__enter__();self.closed=False;self.duration=duration;self.meter_native_steps=0;self.meter_native_cases=0;self.new_native_case=True;self.episode_native_steps=0;self.total_native_steps=0;self.total_env_steps=0;self.invalid_counts={};self.obs=None;self.done=True
  try:self.e=TrackEnv(controller='V2_FFv1',payload='liquid',amplitude=.002,terrain_seed=terrain_seed,duration=duration,A=.75,B=.375,T=30.)
  except BaseException:self.session.__exit__(None,None,None);raise
  self.e.cfg['workspace_m']=1.0
  assert h in [.002,.001,.0005];self.e.model.opt.timestep=h;self.e.physics_dt=h;self.e.cfg['physics_dt']=h;self.e.substeps=round(.002/h);self.h=h
 def observation(self):
  e=self.e;q,qd=e.fluid_state();_,_,a=e.target();w=2*np.pi/30;phase=w*float(e.data.time);return np.clip(np.r_[e.observation(),q/.0084,qd/(e.pfluid.omega*.0084),a/np.array([.5*w*w,w*w]),np.sin(phase),np.cos(phase)],-10,10).astype(np.float32)
 def reset(self,*,seed=None,options=None):
  super().reset(seed=seed);self.episode_native_steps=0;self.new_native_case=True;o,info=self.e.reset(seed=seed,options=options);self.e._n_actor=CentroidModalFeedback();self.e._n_held_delta=np.zeros(3);self.e._o_previous_delta=np.zeros(3);self.previous_delta=np.zeros(3);self.rows=[];self.policy_obs31=[];self.effective_tau_history=[];self.states=[];self.native_controls=[];self.reward_rows=[];self.logger=ActionLogger();self.warning=self.e.data.warning.number.copy();self.done=False;self.episode_return=0.;self.p0=kinematics(self.e)['p'][:2].copy();self.r0=self.e.target()[0].copy();self.e._t_horizon_end_s=self.duration;self.e.pre_sample_hook=self.logger.before;self.e.sample_hook=self.collect;self.obs=self.observation();assert self.observation_space.contains(self.obs);return self.obs.copy(),dict(regime='PRIVILEGED_V4_RESIDUAL_PPO',reference_t_s=float(self.e.data.time),seed=seed)
 def collect(self,e):
  self.logger.after(e);s=kinematics(e);p,v,a=e.target();q,qd=e.fluid_state();self.rows.append(np.r_[e.data.time,s['p'][:2],s['v'][:2],p,v,a,s['rpy'],q,qd,e.slosh_energy(),e.data.ctrl]);
  if self.capture:self.states.append(np.r_[e.data.time,e.data.qpos,e.data.qvel,e.data.qacc,e.data.ctrl,e.data.qacc_warmstart])
 def native(self,r):
  e=self.e
  if self.new_native_case:self.meter_native_cases+=1;self.new_native_case=False
  self.episode_native_steps+=1;self.meter_native_steps+=1;self.total_native_steps+=1;q,qd=e.fluid_state();k=e.base_q;quat=e.data.qpos[k+3:k+7];rpy=Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz');p=e.target()[0];reason=classify(float(np.linalg.norm(q)),float(np.linalg.norm(rpy[:2])),float(np.linalg.norm(e.data.qpos[k:k+2]-p)),float(abs(e.data.ctrl).max()),np.isfinite(np.r_[e.data.qpos,e.data.qvel,e.data.ctrl]).all(),np.any(e.data.warning.number!=self.warning))
  if self.capture:self.native_controls.append(np.r_[e.data.time,e.data.ctrl,q,qd,rpy,len(e.data.contact),float(np.max(abs(r['qvel_identity_gap'])))])
  if np.max(abs(r['qvel_identity_gap']))>1e-8:reason='NUMERICAL_INVALID'
  if reason:raise Guard(reason)
 def fail(self,reason,extra=None):
  self.done=True;self.invalid_counts[reason]=self.invalid_counts.get(reason,0)+1;info=dict(termination_reason=reason,environment_fault=reason in ['CONTACT_INVALID','NUMERICAL_INVALID','INVALID_OBSERVATION'],native_steps=self.episode_native_steps,reward_components={},regime='PRIVILEGED_V4_RESIDUAL_PPO');info.update(extra or {});return self.obs.copy(),-self.reward_cfg['terminal_penalty'],True,False,info
 def step(self,action):
  if self.done:raise gym.error.ResetNeeded('reset required after terminal/truncated')
  z=np.asarray(action,dtype=float)
  if z.shape!=(3,) or not np.isfinite(z).all():return self.fail('INVALID_ACTION')
  if not self.observation_space.contains(self.obs):return self.fail('INVALID_OBSERVATION')
  z=np.clip(z,-1,1);delta=.25*z;e=self.e;n=len(self.rows);before=self.total_native_steps;v4=e._n_actor.action(e);act=np.clip(v4+delta,-1,1);self.policy_obs31.append(self.obs.copy());self.logger.policy(e,self.obs[:23],act);e._n_held_delta=delta.copy();e._o_previous_delta=delta.copy()
  try:
   with StepObserver(e,self.native):_,_,term,trunc,oldinfo=e.step(act)
  except BaseException as err:
   if not isinstance(err,Exception):raise
   reason=str(err) if isinstance(err,Guard) else 'CONTACT_INVALID' if 'UNSUPPORTED' in str(err) else 'NUMERICAL_INVALID';return self.fail(reason,dict(exception=repr(err),native_steps_this_step=self.total_native_steps-before))
  self.total_env_steps+=1;new=np.asarray(self.rows[n:]);terms=[];rates=[]
  for x in new:
   t,r=reward_components(x,delta,self.previous_delta,self.reward_cfg);terms.append(t);rates.append(r)
  reward=float(.002*sum(rates));alloc=dict(alive=.002*len(rates),**dict.fromkeys(self.reward_cfg['weights'],0.))
  for terms_row in terms:
   cum=0.
   for key,val in terms_row.items():
    before_rate=np.exp(-cum);cum+=self.reward_cfg['weights'][key]*val;alloc[key]+=.002*(np.exp(-cum)-before_rate)
  assert abs(sum(alloc.values())-reward)<1e-12
  micro=np.asarray(self.logger.micro[-len(new):]);zero_tau=-10*np.clip(micro[:,1:4]+.03*np.clip(v4,-1,1),-1,1);effective=micro[:,25:28]-zero_tau;self.effective_tau_history.extend(effective.tolist());self.previous_delta=delta.copy();self.obs=self.observation();reason=oldinfo['termination_reason']
  if not np.isfinite(self.obs).all():return self.fail('INVALID_OBSERVATION')
  if term:return self.fail(reason)
  self.done=bool(trunc);self.episode_return+=reward;mean={k:float(np.mean([t[k] for t in terms])) for k in terms[0]};self.reward_rows.append(np.r_[e.data.time,reward,[mean[k] for k in mean],delta,v4])
  info=dict(termination_reason=reason,environment_fault=False,reward_components=mean,native_steps=self.episode_native_steps,native_steps_this_step=self.total_native_steps-before,energy_J=float(new[:,18].mean()),XY_error_m=float(np.linalg.norm(new[-1,1:3]-new[-1,5:7])),torque_peak_Nm=float(abs(new[:,19:22]).max()),q_peak_m=float(np.linalg.norm(new[:,14:16],axis=1).max()),regime='PRIVILEGED_V4_RESIDUAL_PPO',delta=delta.copy(),V4=v4.copy(),reward_decomposition=alloc,effective_delta_torque_RMS_Nm=float(np.sqrt(np.mean(effective**2))),effective_delta_torque_peak_Nm=float(abs(effective).max()),composed_clip_fraction=float(np.any(abs(v4+delta)>1)))
  return self.obs.copy(),reward,False,bool(trunc),info
 def get_state(self):
  keys=['episode_native_steps','previous_delta','policy_obs31','effective_tau_history','rows','states','native_controls','reward_rows','logger','warning','done','episode_return','p0','r0','obs','total_native_steps','total_env_steps','invalid_counts']
  return dict(native=clone(self.e),wrapper={k:copy.deepcopy(getattr(self,k)) for k in keys},np_random=copy.deepcopy(self.np_random.bit_generator.state))
 def set_state(self,s):
  self.new_native_case=True;self.e=clone(s['native'])
  for k,v in s['wrapper'].items():setattr(self,k,copy.deepcopy(v))
  self.np_random.bit_generator.state=copy.deepcopy(s['np_random']);self.e.pre_sample_hook=self.logger.before;self.e.sample_hook=self.collect
 def close(self):
  if not self.closed:
   save('native_sessions/'+str(__import__('os').getpid())+'_'+str(time.time_ns())+'.json',dict(native_steps=self.meter_native_steps,native_cases=self.meter_native_cases,env_steps=self.total_env_steps,invalid_counts=self.invalid_counts,duration_s=self.duration));self.e.pre_sample_hook=None;self.e.sample_hook=None;self.e.close();self.session.__exit__(None,None,None);self.closed=True

 def worker_identity(self):
  import os
  return dict(pid=os.getpid(),terrain_seed=self.e.terrain_seed,native_steps=self.meter_native_steps,episode_steps=self.episode_native_steps,reference=self.e.track_cfg,obs=self.obs.tolist() if self.obs is not None else None,callback_active=ContactSession.active,invalid_counts=self.invalid_counts.copy(),fluid_q=self.e.fluid_state()[0].tolist(),native_time=float(self.e.data.time),controller_integral=self.e.backbone.integral.tolist(),controller_tilt=self.e.backbone.tilt.tolist(),inner_integral=self.e.backbone.inner.integral.tolist(),warmstart=self.e.data.qacc_warmstart.tolist(),native_ctrl=self.e.data.ctrl.tolist(),qpos=self.e.data.qpos.tolist(),qvel=self.e.data.qvel.tolist(),origin=self.e.origin.tolist(),held_delta=self.previous_delta.tolist())
