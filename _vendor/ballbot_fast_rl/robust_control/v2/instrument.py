"""Observational instrumentation of frozen code; no altered baseline equations."""
import numpy as np
from ballbot_fast_rl.outer_loop.control import Cascade,heading,cap_norm,MIX
from ballbot_fast_rl.robust_control.runner import World
from ballbot_fast_rl.robust_control.sensors import SimulatedDevices,PredictionPipeline

def rp_rate(s):
 r,p,_=s['rpy'];wx,wy,wz=s['omega_B'];return np.array([wx+np.sin(r)*np.tan(p)*wy+np.cos(r)*np.tan(p)*wz,np.cos(r)*wy-np.sin(r)*wz])
def inner_preview(inner,s,reference):
 err=reference-s['rpy'][:2];trial=np.clip(inner.integral+err*inner.dt,-inner.limit,inner.limit)
 raw=MIX@(inner.kp*err+inner.ki*trial-inner.kd*rp_rate(s));return raw,trial
class AuditedCascade(Cascade):
 def inspect(self,s,valid=True):
  cfg=self.cfg;finite=np.isfinite(np.r_[s['p'],s['v'],s['rpy'],s['omega_B']]).all();tilt_invalid=np.linalg.norm(s['rpy'][:2])>=.2
  d=dict(flag_invalid_estimator=not valid,flag_invalid_tilt=bool(tilt_invalid),flag_nonfinite=not finite,flag_tilt_magnitude=False,flag_tilt_slew=False,flag_motor_previous=bool(self.inner.last_saturated),flag_integral_cap=False)
  if not valid or not finite or tilt_invalid:return d
  ep=self.reference-s['p'][:2];ev=-s['v'][:2];unbounded=self.integral+ep*self.inner.dt;trial=np.clip(unbounded,-cfg['integral_limit'],cfg['integral_limit']);acc=cfg['kp']*ep+cfg['kd']*ev+cfg['ki']*trial;raw=np.linalg.solve(np.array(cfg['G']),heading(s['rpy'][2]).T@acc);bounded,mag=cap_norm(raw,cfg['tilt_limit']);delta,slew=cap_norm(bounded-self.tilt,cfg['tilt_rate_limit']*self.inner.dt)
  d.update(ep=ep,ev=ev,integral_trial=trial,acc_raw=acc,tilt_raw=raw,tilt_bounded=bounded,tilt_next_trial=self.tilt+delta,slew_demand=float(np.linalg.norm(bounded-self.tilt)/self.inner.dt),flag_tilt_magnitude=mag,flag_tilt_slew=slew,flag_integral_cap=bool(np.any(unbounded!=trial)))
  return d
 def act(self,s,valid=True):
  if self.reference is None:self.reset(s)
  d=self.inspect(s,valid);inner0=self.inner.integral.copy();before=self.integral.copy();a,log=super().act(s,valid)
  invalid=d['flag_invalid_estimator'] or d['flag_invalid_tilt'] or d['flag_nonfinite']
  if not invalid:
   err=log['desired_tilt']-s['rpy'][:2];inner_unbound=inner0+err*self.inner.dt;innertrial=np.clip(inner_unbound,-self.inner.limit,self.inner.limit);raw_trial=MIX@(self.inner.kp*err+self.inner.ki*innertrial-self.inner.kd*rp_rate(s));raw_final=MIX@(self.inner.kp*err+self.inner.ki*self.inner.integral-self.inner.kd*rp_rate(s))
  else:inner_unbound=innertrial=inner0;raw_trial=raw_final=np.zeros(3)
  d.update(flag_inner_saturation=bool(np.any(abs(raw_trial)>1)),flag_motor_current=bool(np.any(abs(a)>=.999999)),flag_inner_integral_cap=bool(np.any(inner_unbound!=innertrial)),integral_accepted=bool(not log['outer_aw']),integral_delta=self.integral-before,raw_motor=raw_final,inner_integral=self.inner.integral.copy())
  return a,dict(log,**d)
class AuditedDevices(SimulatedDevices):
 def __init__(self,*args):super().__init__(*args);self.events=[];self.last_received={}
 def receive(self,now):
  rows=super().receive(now)
  for delivery,seq,kind,stamp,_ in rows:
   self.events.append(dict(sequence=seq,kind=kind,capture_time=stamp,scheduled_delivery=delivery,actual_delivery=now,transport_s=delivery-stamp,delivery_quantization_s=now-delivery));self.last_received[kind]=(stamp,delivery,now)
  return rows
class AuditedPrediction(PredictionPipeline):
 def reset(self):super().reset();self.fused_capture_stamps={}
 def read(self,now):
  old=self.last_delivery;m=super().read(now)
  if self.last_delivery!=old:self.fused_capture_stamps={k:v[0] for k,v in self.last.items()}
  return m
class AuditWorld(World):
 def __init__(self,condition,seed,duration):
  condition=dict(condition,prediction=True);super().__init__(condition,seed,duration)
  # Reset only device RNG/estimator BEFORE any physics; reproduce original first packet exactly.
  if self.devices:
   self.devices=AuditedDevices(self.sensor_cfg,seed+420000);self.pipeline=AuditedPrediction(self.sensor_cfg);self.ref=None;self.measurement=self.measure()
 def timeline(self):
  now=float(self.env.data.time);m=self.measurement;latest=self.pipeline.last if self.pipeline else {};fused=self.pipeline.fused_capture_stamps if self.pipeline else {}
  row=dict(sim_time=now,controller_sim_time=now,actuator_application_sim_time=now,prediction_target=now if m.prediction_timestamp is not None else -1,PV_state_time=m.timestamp,source_timestamp=m.source_timestamp,source_age=m.age,valid=m.valid)
  for kind in ['gyro','attitude','position','encoder']:
   stamp=latest[kind][0] if kind in latest else -1;used=stamp if kind in ['gyro','attitude'] else fused.get(kind,-1)
   row[kind+'_latest_capture']=stamp;row[kind+'_used_capture']=used
   delivered=self.devices.last_received.get(kind) if self.devices else None
   row[kind+'_scheduled_delivery']=delivered[1] if delivered else -1;row[kind+'_actual_delivery']=delivered[2] if delivered else -1
  row['gyro_prediction_horizon']=now-row['gyro_used_capture'] if row['gyro_used_capture']>=0 else -1;return row
