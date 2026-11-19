"""Single V2 correction: output-constraint-aware conditional integration, unchanged gains."""
import numpy as np
from ballbot_fast_rl.outer_loop.control import heading,cap_norm
from .instrument import AuditedCascade,inner_preview
class ConstraintAwareCascade(AuditedCascade):
 version='conditional_constraint_aw_v2'
 def act(self,s,valid=True):
  if self.reference is None:self.reset(s)
  d=self.inspect(s,valid);bad=d['flag_invalid_estimator'] or d['flag_invalid_tilt'] or d['flag_nonfinite']
  if bad:return super().act(s,valid)
  cfg=self.cfg;I0=self.integral.copy();tilt0=self.tilt.copy();B=np.linalg.inv(np.array(cfg['G']))@heading(s['rpy'][2]).T;pd=cfg['kp']*d['ep']+cfg['kd']*d['ev'];trial=d['integral_trial']
  def command(I):
   raw=B@(pd+cfg['ki']*I);bounded,mag=cap_norm(raw,cfg['tilt_limit']);delta,slew=cap_norm(bounded-tilt0,cfg['tilt_rate_limit']*self.inner.dt);return raw,bounded,tilt0+delta,mag,slew
  raw0,_,next0,_,_=command(I0);raw1,_,next1,_,_=command(trial)
  # Reject only increased physical/output constraint violation; inward updates can unwind.
  amplitude_worse=max(0.,np.linalg.norm(raw1)-cfg['tilt_limit'])>max(0.,np.linalg.norm(raw0)-cfg['tilt_limit'])+1e-14
  motor0,_=inner_preview(self.inner,s,next0);motor1,_=inner_preview(self.inner,s,next1)
  from ballbot_fast_rl.outer_loop.control import MIX
  latent_motor_delta=MIX@(self.inner.kp*(raw1-raw0))
  latent_worsens=bool(np.any((abs(motor0)>=1.-1e-12)&(np.sign(motor0)*latent_motor_delta>1e-14)))
  motor_worse=bool(np.any(np.maximum(abs(motor1)-1,0)>np.maximum(abs(motor0)-1,0)+1e-14) or latent_worsens)
  accepted=not(amplitude_worse or motor_worse)
  if accepted:self.integral=trial
  raw,bounded,nexttilt,mag,slew=command(self.integral);raw_motor,innertrial=inner_preview(self.inner,s,nexttilt);inner_before=self.inner.integral.copy();self.tilt=nexttilt;a,innerlog=self.inner.act(s,self.tilt,True)
  d.update(acc_raw=pd+cfg['ki']*self.integral,tilt_raw=raw,tilt_bounded=bounded,slew_demand=float(np.linalg.norm(bounded-tilt0)/self.inner.dt),flag_projection=False,flag_amplitude_reject=bool(amplitude_worse),flag_motor_reject=motor_worse,flag_inner_saturation=bool(np.any(abs(raw_motor)>1)),flag_motor_current=bool(np.any(abs(a)>=.999999)),flag_inner_integral_cap=bool(np.any(innertrial!=inner_before+(self.tilt-s['rpy'][:2])*self.inner.dt)),integral_accepted=accepted,integral_delta=self.integral-I0,raw_motor=raw_motor,inner_integral=self.inner.integral.copy())
  return a,dict(innerlog,**d,integral=self.integral.copy(),desired_tilt=self.tilt.copy(),outer_aw=not accepted,tilt_clamped=mag,slew_limited=slew,a_cmd=pd+cfg['ki']*self.integral)
