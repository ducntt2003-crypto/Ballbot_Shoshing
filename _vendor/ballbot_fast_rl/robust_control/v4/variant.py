"""ONE correction A: measured world-heading P feedback in spare common mode."""
import numpy as np
from ballbot_fast_rl.robust_control.v3.variant import BandwidthCascade
YAW_KP=6.0

def wrap(angle):return float(np.arctan2(np.sin(angle),np.cos(angle)))
class HeadingCascade(BandwidthCascade):
 version='v4_heading_P6_commonmode_headroom'
 def reset(self,state,reference=None):
  super().reset(state,reference)
  self.yaw_reference=None if state is None else float(state['rpy'][2])
 def act(self,s,valid=True):
  a,log=super().act(s,valid)
  if not valid or any(log.get(k,False) for k in ['flag_invalid_estimator','flag_invalid_tilt','flag_nonfinite']):
   return a,dict(log,yaw_error=0.,yaw_raw=0.,yaw_applied=0.,yaw_headroom_limited=False)
  if self.yaw_reference is None:self.yaw_reference=float(s['rpy'][2])
  error=wrap(float(s['rpy'][2])-self.yaw_reference);raw=-YAW_KP*error
  # Preserve V3 differential commands; only limit the new uniform yaw addition.
  lower=float(np.max(-1.-a.astype(float)));upper=float(np.min(1.-a.astype(float)));applied=float(np.clip(raw,lower,upper));actual=np.clip(a.astype(float)+applied,-1,1).astype(np.float32)
  return actual,dict(log,yaw_reference=self.yaw_reference,yaw_error=error,yaw_raw=raw,yaw_applied=applied,yaw_headroom_limited=bool(abs(raw-applied)>1e-12),differential_raw_motor=log.get('raw_motor',a).copy(),raw_motor=log.get('raw_motor',a)+raw)
