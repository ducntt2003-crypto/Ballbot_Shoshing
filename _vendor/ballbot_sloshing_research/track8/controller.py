"""FFv1: add reference velocity/acceleration, retain original gains/constraints.

Integral updates retain REAL position error; acceleration feedforward is not
integrated as a false position offset. No modification of frozen classes.
"""
import numpy as np
from ballbot_fast_rl.robust_control.v2.variant import ConstraintAwareCascade
from ballbot_fast_rl.robust_control.v3.variant import BandwidthCascade
class FeedforwardMixin:
 version='TRACK8_REFERENCE_FF_V1'
 def set_reference(self,p,v,a):self.reference=np.array(p);self.vref=np.array(v);self.aref=np.array(a)
 def inspect(self,s,valid=True):
  d=super().inspect(s,valid)
  if 'ep' in d:
   d['physical_ep']=d['ep'].copy();d['ep']=d['ep']+getattr(self,'aref',np.zeros(2))/self.cfg['kp'];d['ev']=getattr(self,'vref',np.zeros(2))-s['v'][:2]
  return d
class TrackingV2FF(FeedforwardMixin,ConstraintAwareCascade):pass
class TrackingV3FF(FeedforwardMixin,BandwidthCascade):pass
