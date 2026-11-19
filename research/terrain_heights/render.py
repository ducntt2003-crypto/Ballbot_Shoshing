import os
os.environ['MUJOCO_GL']='osmesa';os.environ['LP_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,sys,time,numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text());SOURCE=R.parents[2]/P['source_release'];sys.path.insert(0,str(SOURCE))
import mujoco,cv2
from sloshbot.media import load_model,line
from sloshbot.surface import draw_surface
from ballbot_sloshing_research.track8.terrain import sample

def context(c,method,width=320,height=240):
 folder=R/f'runs/{c}_{method}_{P["seeds"][0]}'
 if not (folder/'result.json').exists():return None
 r=json.loads((folder/'result.json').read_text());z=np.load(folder/'trace.npz');m=load_model(folder);d=mujoco.MjData(m);H=np.load(folder/'world_heights.npy');renderer=mujoco.Renderer(m,height,width,max_geom=1500);wire=[];gx=np.linspace(-1.15,1.15,15);gy=np.linspace(-.8,.8,11)
 for yy in gy:
  for a,b in zip(gx[:-1],gx[1:]):wire.append(([a,yy,sample(H,a,yy)[0]+.001],[b,yy,sample(H,b,yy)[0]+.001]))
 for xx in gx:
  for a,b in zip(gy[:-1],gy[1:]):wire.append(([xx,a,sample(H,xx,a)[0]+.001],[xx,b,sample(H,xx,b)[0]+.001]))
 return dict(c=c,method=method,result=r,raw=z['native_rows'],xyz=z['xyzv_ball'],m=m,d=d,H=H,renderer=renderer,wire=wire,width=width,height=height)

def frame(ctx,t,side=False):
 im=np.zeros((ctx['height'],ctx['width'],3),np.uint8)
 if t>ctx['result']['native_stop_s']+1e-8:return im
 raw=ctx['raw'];m=ctx['m'];d=ctx['d'];k=max(0,min(len(raw)-1,int(np.searchsorted(raw[:,0],t+1e-8,side='right')-1)));row=raw[k];d.qpos[:]=row[1:1+m.nq];d.qvel[:]=row[1+m.nq:1+m.nq+m.nv];d.time=row[0];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d);cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.25];cam.distance=2.35;cam.azimuth=120;cam.elevation=-25
 if side:cam.lookat[:]=[0,0,.28];cam.distance=1.85;cam.azimuth=90;cam.elevation=-12
 ctx['renderer'].update_scene(d,camera=cam);scene=ctx['renderer'].scene;draw_surface(scene,m,d)
 for a,b in ctx['wire']:line(scene,a,b,[.40,.53,.65,.65])
 pts=ctx['xyz'][:k+1:max(1,(k+1)//170),7:9];trail=np.array([[x,y,sample(ctx['H'],x,y)[0]+.003] for x,y in pts])
 for a,b in zip(trail[:-1],trail[1:]):line(scene,a,b,[.1,1,.2,1])
 im=ctx['renderer'].render().copy();ball=d.geom_xpos[m.geom('the_ball').id];h=1000*sample(ctx['H'],*ball[:2])[0];lines=[f'{ctx["c"]} | {ctx["method"]} | native t={d.time:.2f}s',f'Actual ground under ball={h:.1f}mm',f'Run result:{ctx["result"]["termination"]}; {ctx["result"]["metrics"]["ordered_nodes_passed"]}/16']
 for j,txt in enumerate(lines):cv2.putText(im,txt,(5,15+17*j),cv2.FONT_HERSHEY_SIMPLEX,.34 if not side else .48,(255,255,255),1,cv2.LINE_AA)
 return im

def main():
 contexts=[context(c,m) for m in P['controllers'] for c in P['conditions']];mixed=context('MIXED','V4',640,480);writer=cv2.VideoWriter(str(R/'media/height_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(1280,528));assert writer.isOpened();compare=[];detail=[];tick=time.monotonic()
 try:
  for i in range(151):
   assert time.monotonic()-tick<180;t=i/5;tiles=[]
   for index,ctx in enumerate(contexts):
    if ctx is None:
     im=np.zeros((240,320,3),np.uint8);cv2.putText(im,'NOT RUN: safety gate',(8,60),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,120,120),1)
    else:im=frame(ctx,t)
    tiles.append(im)
   canvas=np.zeros((528,1280,3),np.uint8);canvas[48:]=np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]);cv2.putText(canvas,f'Real collision terrain:5 /10 /20mm /MIXED | actual native replay t={t:.1f}s',(7,19),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);cv2.putText(canvas,'Rows:V4,PPO. Green=actual path; wire grid follows true ground. No vertical exaggeration. ROM surface, not CFD.',(7,40),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1);writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR));compare.append(Image.fromarray(canvas).quantize(colors=112))
   if mixed:
    im=frame(mixed,t,side=True);cv2.putText(im,'MIXED: actual5/10/20mm hills | recorded simulation',(7,465),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1);detail.append(Image.fromarray(im).quantize(colors=112))
   if i in [30,75,120]:cv2.imwrite(str(R/f'media/height_comparison_t{t:.0f}s.png'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
 finally:
  writer.release()
  for ctx in contexts+[mixed]:
   if ctx:ctx['renderer'].close()
 compare[0].save(R/'media/height_comparison.gif',save_all=True,append_images=compare[1:],duration=200,loop=0,optimize=True)
 if detail:detail[0].save(R/'media/mixed_5_10_20mm.gif',save_all=True,append_images=detail[1:],duration=200,loop=0,optimize=True)
 with (R/'media/provenance.json').open('x') as f:json.dump(dict(source='first registered seed2611801, all available conditions',frames=151,fps=5,display_includes_endpoint=True,actual_native_span_s=30,backend='CPU OSMesa',render_wall_s=time.monotonic()-tick,new_physics_steps=0,new_training_steps=0,wireframe='Visualization on actual collision surface, not a separate obstacle',surface='ROM-based visualization, not CFD'),f,indent=2)
 print('MEDIA_COMPLETE',time.monotonic()-tick,flush=True)
if __name__=='__main__':main()
