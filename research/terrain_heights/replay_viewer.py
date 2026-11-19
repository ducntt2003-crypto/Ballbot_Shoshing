"""Interactive3D playback of the newly recorded higher-terrain simulations. No new physics."""
from pathlib import Path
import sys,json,time,argparse,numpy as np
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text());SOURCE=R.parents[2]/P['source_release'];sys.path.insert(0,str(SOURCE))
from sloshbot.media import load_model,line
from sloshbot.surface import draw_surface
from sloshbot.cameras import configure_camera
from ballbot_sloshing_research.track8.terrain import sample
import mujoco,mujoco.viewer

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=P['conditions'],default='MIXED');p.add_argument('--controller',choices=P['controllers'],default='V4');p.add_argument('--smoke-seconds',type=float);a=p.parse_args();folder=R/'runs'/f'{a.condition}_{a.controller}_{P["seeds"][0]}';z=np.load(folder/'trace.npz');raw=z['native_rows'];xyz=z['xyzv_ball'];H=np.load(folder/'world_heights.npy');m=load_model(folder);d=mujoco.MjData(m);flags={'pause':False,'reset':False,'cam':2};wire=[];gx=np.linspace(-1.15,1.15,15);gy=np.linspace(-.8,.8,11)
 for y in gy:
  for x1,x2 in zip(gx[:-1],gx[1:]):wire.append(([x1,y,sample(H,x1,y)[0]+.001],[x2,y,sample(H,x2,y)[0]+.001]))
 for x in gx:
  for y1,y2 in zip(gy[:-1],gy[1:]):wire.append(([x,y1,sample(H,x,y1)[0]+.001],[x,y2,sample(H,x,y2)[0]+.001]))
 def key(k):
  if k==32:flags['pause']=not flags['pause']
  elif k in [82,114]:flags['reset']=True
  elif 49<=k<=53:flags['cam']=k-48
 d.qpos[:]=raw[0,1:1+m.nq];d.qvel[:]=raw[0,1+m.nq:1+m.nq+m.nv];d.time=raw[0,0];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
 tick=time.monotonic();play=0.;last=tick;camera=-1;frames=0
 print('Recorded native simulation replay:',a.condition,a.controller,'| Space pause, R restart,1-5 cameras, Esc exit. No new dynamics.',flush=True)
 with mujoco.viewer.launch_passive(m,d,key_callback=key) as viewer:
  while viewer.is_running():
   now=time.monotonic();dt=now-last;last=now
   if a.smoke_seconds and now-tick>=a.smoke_seconds:break
   if flags['reset']:play=0.;flags['reset']=False
   if not flags['pause']:play=min(float(raw[-1,0]),play+dt)
   k=max(0,min(len(raw)-1,int(np.searchsorted(raw[:,0],play,side='right')-1)));row=raw[k];d.qpos[:]=row[1:1+m.nq];d.qvel[:]=row[1+m.nq:1+m.nq+m.nv];d.time=row[0];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d)
   with viewer.lock():
    if flags['cam']!=camera:configure_camera(viewer.cam,m,d,flags['cam']);viewer.cam.distance=2.15 if flags['cam']!=5 else .29;camera=flags['cam']
    scene=viewer.user_scn;scene.ngeom=0;draw_surface(scene,m,d)
    for v,w in wire:line(scene,v,w,[.5,.6,.7,.6])
    path=xyz[:k+1:max(1,(k+1)//180),7:9];trail=np.array([[x,y,sample(H,x,y)[0]+.003] for x,y in path])
    for v,w in zip(trail[:-1],trail[1:]):line(scene,v,w,[.1,1,.2,1])
   if camera in [2,3,4] and frames==0:viewer.cam.lookat[:]=[0,0,.25]
   viewer.sync();frames+=1;time.sleep(.02)
 print(json.dumps(dict(recorded_replay=True,new_physics_steps=0,displayed_sim_time_s=float(d.time),frames=frames,wall_s=time.monotonic()-tick)),flush=True)
if __name__=='__main__':main()
