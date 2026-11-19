import json
from pathlib import Path
from ballbot_fast_rl.robust_control.config import GAINS,ZERO
ROOT=Path(__file__).resolve().parents[2]/'results/robust_cascade_v2'
HISTORY=ROOT.parent/'robust_control_20261030'
CLEAN=dict(ZERO,imu_hz=250,attitude_hz=100,position_hz=50,encoder_hz=100,estimator_hz=100)
CASES={'perfect':dict(sensor=dict(mode='perfect')),'sampled_clean':dict(sensor=CLEAN),'estimated':dict(sensor=dict(mode='estimated'))}
for ms in [2,4,6,10]:CASES[f'delay{ms}']=dict(sensor=dict(mode='estimated',delay_s=ms/1000))
for name,offset in [('COM_plusX',[.002,0,0]),('COM_minusX',[-.002,0,0]),('COM_plusY',[0,.002,0]),('COM_minusY',[0,-.002,0])]:CASES[name]=dict(sensor=dict(mode='estimated'),robot_params={'ballast_offset_m':offset})
CASES['combined']=dict(sensor=dict(mode='estimated',delay_s=.002,gyro_sigma=.002,orientation_sigma=.0015,gyro_bias=[.002,-.002,.001],dropout=.02))
