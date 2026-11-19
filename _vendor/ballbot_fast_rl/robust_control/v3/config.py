from pathlib import Path
import copy
from ballbot_fast_rl.robust_control.v2.config import CASES as V2CASES,GAINS
ROOT=Path(__file__).resolve().parents[2]/'results/fast_loop_latency_v3'
HISTORY=ROOT.parent/'robust_cascade_v2'
CASES=copy.deepcopy(V2CASES)
# These transport values are held out from correction design/refinement.
for ms in [3,5,8]:CASES['holdout_delay'+str(ms)]=dict(sensor=dict(mode='estimated',delay_s=ms/1000))
CASES['push']=dict(sensor=dict(mode='estimated'),push_N=.5,push_start_s=2.,push_duration_s=.1)
