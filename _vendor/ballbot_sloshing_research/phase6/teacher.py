"""Privileged finite-action short-horizon reference, explicitly offline."""
import time
import numpy as np
from ballbot_fast_rl.outer_loop.control import MIX,kinematics
from ballbot_sloshing_research.phase5.causal import fork
class Teacher:
 label='PRIVILEGED MODEL-BASED RESEARCH CONTROLLER'
 def __init__(self):
  self.candidates=[np.zeros(3)]
  for theta in np.arange(8)*np.pi/4:
   a=MIX@np.array([np.cos(theta),np.sin(theta)]);self.candidates.append(.5*a/max(abs(a)))
  self.records=[]
 def predict(self,e,a):
  f=fork(e);f.max_steps=f.steps+50;f.push_W=np.zeros(3);values=[];valid=True;warning0=f.data.warning.number.copy()
  def sample(env):
   nonlocal valid
   s=kinematics(env);q,v=env.fluid_state();ep=s['p'][:2]-env.reference();tau=env.data.actuator_force[env.motor_ids];row=np.r_[env.slosh_energy(),ep,s['rpy'][:2],tau];values.append(row)
   valid=bool(valid and np.isfinite(row).all() and np.linalg.norm(q)<=.0084 and np.linalg.norm(s['rpy'][:2])<=.08 and np.linalg.norm(ep)<=.03 and abs(tau).max()<=10+1e-9)
  f.sample_hook=sample
  for k in range(10):
   _,_,t,u,info=f.step(a if k<5 else np.zeros(3))
   if t:valid=False;break
  v=np.array(values);valid=bool(valid and len(v)==50 and np.array_equal(f.data.warning.number,warning0));Eref=.5*e.pfluid.stiffness*.003**2;pos=float(np.sqrt(np.mean(np.sum(v[:,1:3]**2,axis=1))));tau=float(np.sqrt(np.mean(v[:,5:8]**2)));J=(.5*v[-1,0]+.5*v[:,0].mean())/Eref+pos**2/.02**2+.1*np.mean(np.sum(v[:,3:5]**2,axis=1))/.03**2+.001*np.mean(a*a)
  return dict(cost=float(J),E_end=float(v[-1,0]),E_mean=float(v[:,0].mean()),position_RMS=pos,torque_RMS=tau,valid=valid)
 def act(self,e):
  start=time.monotonic();pred=[self.predict(e,a) for a in self.candidates];b=pred[0];eligible=[i for i,p in enumerate(pred) if p['valid'] and p['position_RMS']<=1.05*b['position_RMS']+.00002 and p['torque_RMS']<=1.10*b['torque_RMS']+.00002];idx=min(eligible,key=lambda i:pred[i]['cost']) if eligible else 0
  if pred[idx]['cost']>=b['cost']:idx=0
  self.records.append(dict(t=float(e.data.time),selected=idx,wall_s=time.monotonic()-start,predictions=pred));return self.candidates[idx].copy()
