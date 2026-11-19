"""TRACK8L_CAUSAL_ACTION_LOGGER_V1: pre-step controller values, post-step controls."""
from .common import *
class ActionLogger:
 version='TRACK8L_CAUSAL_ACTION_LOGGER_V1'
 def __init__(self):self.policies=[];self.obs=[];self.policy_qpos=[];self.policy_qvel=[];self.policy_memory=[];self.micro=[];self.pre=None
 def memory(self,e):return np.r_[e.backbone.integral,e.backbone.inner.integral,e.backbone.tilt,e.backbone.reference]
 def policy(self,e,o,a):
  p,v,b=e.target();self.policies.append(np.r_[e.data.time,e.data.time,e.data.time,a,p,v,b]);self.obs.append(np.array(o).copy());self.policy_qpos.append(e.data.qpos.copy());self.policy_qvel.append(e.data.qvel.copy());self.policy_memory.append(self.memory(e).copy())
 def before(self,e,a):
  total=np.clip(e.base_action+.03*a,-1,1);self.pre=np.r_[e.data.time,e.base_action.copy(),a.copy(),total,-10*total,self.memory(e).copy(),e.previous_action.copy()]
 def after(self,e):
  p=self.pre;ctrl=e.data.ctrl[e.motor_ids].copy();tau=e.data.actuator_force[e.motor_ids].copy();assert np.max(abs(ctrl-p[10:13]))<=1e-12;assert np.max(abs(tau-ctrl))<=1e-12;self.micro.append(np.r_[p,e.data.time,ctrl,tau,e.base_action.copy(),self.memory(e).copy()])
 def write(self,folder):
  np.savez_compressed(folder/'action_logger.npz',micro=self.micro,policy=self.policies,observation=np.asarray(self.obs),policy_qpos=self.policy_qpos,policy_qvel=self.policy_qvel,policy_memory=self.policy_memory)
  return dict(version=self.version,native_rows=len(self.micro),policy_rows=len(self.policies),observations_shape=list(np.asarray(self.obs).shape),microschema=dict(application_t=0,pre_backbone=[1,4],held_residual=[4,7],clipped_total=[7,10],expected_ctrl=[10,13],pre_memory=[13,21],previous_applied=[21,24],post_t=24,actual_ctrl=[25,28],actual_motor_tau=[28,31],next_backbone=[31,34],next_memory=[34,42]),policy_schema='observation_t,reference_t,controller_update_t,residual3,p2,v2,a2; observation float32 archived exactly before action; V4 also uses separately saved fullsim/controllerstate (regimeC)',scope='Pre-values are copies from live e, never stale post clone. Next backbone after prepare is separate from applied pre-backbone. Post/application stamps differ by native dt. Fixed simulation time, not execution latency.')
