import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'results/robust_control_20261030'
OLD=ROOT.parent/'outer_loop_position_control'
GAINS=json.loads((OLD/'frozen_controller.json').read_text())['gains']
ZERO=dict(mode='direct',gyro_sigma=0.,gyro_bias=[0,0,0],orientation_sigma=0.,position_sigma=0.,velocity_sigma=0.,encoder_sigma=0.,imu_hz=500,attitude_hz=500,position_hz=500,encoder_hz=500,estimator_hz=500)
EST=dict(mode='estimated')
CASES={
 'perfect':dict(sensor=dict(mode='perfect')),
 'direct_clean':dict(sensor=ZERO),
 'noisy_direct':dict(sensor=dict(mode='direct')),
 'estimated_nominal':dict(sensor=EST),
 'imu_noise':dict(sensor=dict(ZERO,gyro_sigma=.005,orientation_sigma=.002)),
 'gyro_bias':dict(sensor=dict(ZERO,gyro_bias=[.003,-.003,.001])),
 'encoder_noise':dict(sensor=dict(ZERO,encoder_sigma=.10)),
 'delay_small':dict(sensor=dict(EST,delay_s=.006)),
 'delay_large':dict(sensor=dict(EST,delay_s=.050)),
 'combined':dict(sensor=dict(EST,delay_s=.020,gyro_sigma=.002,orientation_sigma=.0015,gyro_bias=[.002,-.002,.001],dropout=.02)),
 'stale_rejection':dict(sensor=dict(EST,delay_s=.150)),
 'coherent_inertia':dict(sensor=EST,robot_params={'robot_mass_scale':1.02}),
 'motor_mismatch':dict(sensor=EST,motor_gains=[1.02,.98,1.01]),
 'friction':dict(sensor=EST,robot_params={'wheel_friction_scale':.95,'ground_friction':.95}),
}
for label,offset in [('COM_plusX',[.002,0,0]),('COM_minusX',[-.002,0,0]),('COM_plusY',[0,.002,0]),('COM_minusY',[0,-.002,0])]:CASES[label]=dict(sensor=EST,robot_params={'ballast_offset_m':offset})
