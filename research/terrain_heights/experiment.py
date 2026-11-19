"""Versioned higher terrain only: robot, fluid, controllers and limits frozen."""
from pathlib import Path
import sys,json,time,hashlib,xml.etree.ElementTree as ET
import numpy as np
R=Path(__file__).resolve().parent;REPO=R.parents[2];P=json.loads((R/'protocol.json').read_text());SOURCE=REPO/P['source_release'];sys.path.insert(0,str(SOURCE))
from sloshbot.runtime import FigureEnv
from sloshbot.experiment import dump
from ballbot_sloshing_research.track8.evaluate import metrics
from ballbot_fast_rl.outer_loop.control import kinematics
from ballbot_sloshing_research.track8_l import adapter as contact
from stable_baselines3 import PPO
import mujoco

def bump(x,y,cx,cy,r):return np.maximum(0,1-((x-cx)**2+(y-cy)**2)/r**2)**3

def terrain(condition):
 axis=np.linspace(-2,2,129);x,y=np.meshgrid(axis,axis)
 if condition=='MIXED':return .005*bump(x,y,.62,-.30,.30)+.010*bump(x,y,.62,.30,.45)+.020*bump(x,y,-.70,-.20,.60)
 a=P['height_m'][condition];return a*(bump(x,y,.70,.20,.60)+bump(x,y,-.70,-.20,.60))

class HeightEnv(FigureEnv):
 def __init__(self,condition,**kw):
  super().__init__(**kw);self.condition=condition;self.xyz=[];e=self.e;H=terrain(condition);lo=float(H.min());span=max(.004,float(np.ptp(H)));tree=ET.parse(e.xml_path);root=tree.getroot();hf=root.find(".//hfield[@name='track8_collision']");size=hf.get('size').split();size[2]=str(span);hf.set('size',' '.join(size));root.find(".//geom[@name='terrain']").set('pos',f'0 0 {lo}');file=e.xml_path.parent/'height_comparison.xml';tree.write(file);old=e.model;old_hf=old.hfield_data.copy();e.model=mujoco.MjModel.from_xml_path(str(file));e.model.hfield_data[:]=old_hf;hid=e.model.hfield('track8_collision').id;adr=int(e.model.hfield_adr[hid]);e.model.hfield_data[adr:adr+129*129]=(H.ravel()-lo)/span
  self.H=e.model.hfield_data[adr:adr+129*129].astype(float).reshape(129,129)*span+lo;e.heights=self.H.copy()
  for name in ['body_mass','body_inertia','dof_damping','jnt_stiffness','pair_friction','actuator_ctrlrange']:np.testing.assert_array_equal(getattr(old,name),getattr(e.model,name))
  e.model.opt.enableflags=old.opt.enableflags;e.xml_path=file;e.model_hash=hashlib.sha256(file.read_bytes()).hexdigest();e.data=mujoco.MjData(e.model)
 def collect(self,e):
  super().collect(e);s=kinematics(e);self.xyz.append(np.r_[e.data.time,s['p'],s['v'],e.data.geom_xpos[e.ball_geom]])

def run(condition,controller,seed,duration,output):
 out=Path(output);out.mkdir(parents=True,exist_ok=False);env=HeightEnv(condition=condition,controller=controller,terrain_seed=2611811,duration=duration,capture=True);obs,_=env.reset(seed=seed);e=env.e;tick=time.monotonic();model=PPO.load(SOURCE/'pretrained/ppo_500k.zip',device='cpu') if controller=='PPO' else None
 initial=dict(qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),origin=e.origin.tolist());(out/'model.xml').write_text(e.xml_path.read_text().replace(str(SOURCE/'_vendor/ballbotgym/ballbotgym/assets'),'__ASSET_ROOT__'));np.save(out/'hfield.npy',e.model.hfield_data);np.save(out/'world_heights.npy',env.H);error=None;reason='NOT_STARTED'
 try:
  for _ in range(round(duration/.01)):
   action=model.predict(obs,deterministic=True)[0] if model else np.zeros(3);obs,_,term,trunc,info=env.step(action);reason=info['termination_reason']
   if term or trunc:break
   if time.monotonic()-tick>90:reason='WALL_BUDGET';break
 except Exception as exc:error=repr(exc);reason='EXCEPTION'
 finally:
  x=np.array(env.rows).reshape(-1,22);np.savez_compressed(out/'trace.npz',rows=x,native_rows=env.states,xyzv_ball=env.xyz);np.savez_compressed(out/'terminal.npz',t=e.data.time,qpos=e.data.qpos,qvel=e.data.qvel,ctrl=e.data.ctrl);stats=contact.stats();invalid=env.invalid_counts.copy();valid=not invalid and error is None and stats['unsupported']==0;steps=env.meter_native_steps;stop=float(e.data.time);env.close()
 r=dict(condition=condition,controller=controller,seed=seed,planned_s=duration,native_stop_s=stop,last_recorded_s=float(x[-1,0]) if len(x) else 0,native_steps=steps,completed=reason=='TIME_LIMIT' and steps==round(duration/.002),physical_valid=valid,termination=reason,error=error,invalid_counts=invalid,contact_stats=stats,initial=initial,wall_s=time.monotonic()-tick,height_range_m=[float(env.H.min()),float(env.H.max())],metrics=metrics(x,duration,'figure8',.75,.375,30,reason,'liquid') if len(x) else {})
 dump(out/'result.json',r);print(condition,controller,seed,stop,reason,flush=True);return r
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('condition',choices=P['conditions']);p.add_argument('controller',choices=P['controllers']);p.add_argument('seed',type=int);p.add_argument('duration',type=float);p.add_argument('output');a=p.parse_args();run(a.condition,a.controller,a.seed,a.duration,a.output)
