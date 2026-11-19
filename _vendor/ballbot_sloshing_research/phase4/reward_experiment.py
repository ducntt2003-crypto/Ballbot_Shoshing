"""One primary change: positive generalized energy-input power penalty; no force change."""
import numpy as np
from ballbot_sloshing_research.phase2.env import Phase2Env
from .energy_diagnostics import sample

class EnergyEnv(Phase2Env):
 def __init__(self,variant='R1',**kwargs):
  assert variant in ['R1','RP'];self.variant=variant;self.last_energy_sample=None;self.last_energy_time=None;self.power_weight=.2;self.reward_costs=[]
  super().__init__(observation='O1',reward='R1',**kwargs);self.reward_name=variant;self.power_scale=.5*self.pfluid.stiffness*.003**2*self.pfluid.omega
 def step(self,action):
  external=getattr(self,'sample_hook',None);costs=[];powers=[]
  def collect(env):
   d=sample(env);self.last_energy_sample=d;self.last_energy_time=float(env.data.time);pin=d['Pin'];c=(self.power_weight if self.variant=='RP' else 0)*max(0.,pin)/self.power_scale;costs.append(c);powers.append(pin)
   if external:external(env)
  self.sample_hook=collect
  try:obs,base,t,u,info=super().step(action)
  finally:self.sample_hook=external
  penalty=.002*sum(costs);terms=dict(info['phase2_reward_terms']);terms['power']=float(np.mean(costs));info.update(base_reward=base,power_penalty_integral=penalty,reward_components=terms,positive_generalized_power_W=float(np.mean(np.maximum(powers,0))),signed_generalized_power_W=float(np.mean(powers)),power_scale_W=self.power_scale,reward_version='P4_POSITIVE_MODAL_POWER_V1' if self.variant=='RP' else 'P2_R1_UNCHANGED');return obs,float(base-penalty),t,u,info
 def reset(self,**kwargs):
  self.last_energy_sample=None;self.last_energy_time=None;return super().reset(**kwargs)
