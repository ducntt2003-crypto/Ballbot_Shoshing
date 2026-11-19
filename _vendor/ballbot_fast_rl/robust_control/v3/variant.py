"""V3 Option B, refinement2: reduce inner rate-feedback bandwidth only.
P/I gains preserve low-frequency restoring action; predictor/allocation/outer/safety unchanged.
"""
from ballbot_fast_rl.robust_control.v2.variant import ConstraintAwareCascade
from .config import GAINS
BETA=.55
class BandwidthCascade(ConstraintAwareCascade):
 version='v3_inner_rate_bandwidth_kd110_refinement2'
 def __init__(self,cfg=None):
  super().__init__(GAINS if cfg is None else cfg)
  self.inner.kd=2*BETA
