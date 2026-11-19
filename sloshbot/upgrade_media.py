"""Native render of recorded terrain trials; actual green trail + ROM-only surface."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from .bootstrap import ROOT,setup
setup()
from .scene_upgrade import decorate
from .media import load_model
from .surface import draw_surface
import mujoco,cv2,numpy as np,json,time
from PIL import Image

def render(folder,out,view='world'):
    r=json.loads((folder/'result.json').read_text());z=np.load(folder/'trace.npz');raw=z['native_rows'];xyz=z['xyzv'];y=z['telemetry'];heights=np.load(folder/'world_heights.npy');m=load_model(folder);d=mujoco.MjData(m);renderer=mujoco.Renderer(m,480,640,max_geom=1800);cam=mujoco.MjvCamera();writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,480));assert writer.isOpened();indices=list(range(0,len(raw),25))
    if indices[-1]!=len(raw)-1:indices.append(len(raw)-1)
    start=time.monotonic();surface_peak=0.
    try:
        for i in indices:
            assert time.monotonic()-start<120
            row=raw[i];d.qpos[:]=row[1:1+m.nq];d.qvel[:]=row[1+m.nq:1+m.nq+m.nv];d.time=row[0];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d)
            cam.lookat[:]=[.32,0,.30];cam.distance=1.6;cam.azimuth=125;cam.elevation=-23
            if view=='liquid':cam.lookat[:]=d.xpos[m.body('water_tank').id]+[0,0,.035];cam.distance=.29;cam.azimuth=110;cam.elevation=-45
            renderer.update_scene(d,camera=cam);info=decorate(renderer.scene,m,d,heights,xyz[:i+1,1:4],arrow=view!='liquid');surface_peak=max(surface_peak,info['surface_max_T_m']);im=renderer.render().copy();a=y[i]
            lines=[f'SloshBot-RL V2 | {r["level"]} | native state replay',f't={d.time:.3f}s / stop={r["native_stop_s"]:.3f}s | {r["termination"]}',f'XYZ=({xyz[i,1]:.3f},{xyz[i,2]:.3f},{xyz[i,3]:.3f})m; crossing={r["traversal"]["passed"]}',f'Keys forward={a[1]:+.0f} left={a[2]:+.0f} | COMMAND REPLAY',f'ROM q=({a[15]*1000:.2f},{a[16]*1000:.2f})mm E={a[14]:.5g}J','Surface: linear Bessel reconstruction, NOT CFD/SPH','Green=actual trail on collision surface; geometry unexaggerated']
            for j,t in enumerate(lines):cv2.putText(im,t,(6,18+j*17),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA)
            if i==indices[len(indices)//2]:cv2.imwrite(str(out.with_suffix('.png')),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
            writer.write(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
    finally:writer.release();renderer.close()
    return dict(video=str(out.relative_to(ROOT)),source=str(folder.relative_to(ROOT)),frames=len(indices),fps=20,last_recorded_s=float(raw[-1,0]),native_stop_s=r['native_stop_s'],terminal_record_available=(folder/'terminal.npz').exists(),native_steps_generated=0,view=view,surface_peak_T_m=surface_peak,physically_valid=r['physical_valid'])

def gif(src,out,start_s=0,end_s=None):
    cap=cv2.VideoCapture(str(src));fps=cap.get(cv2.CAP_PROP_FPS);frames=[];i=0
    while True:
        ok,im=cap.read()
        if not ok:break
        t=i/fps
        if i%4==0 and t>=start_s and (end_s is None or t<=end_s):frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(im,(384,288)),cv2.COLOR_BGR2RGB)).quantize(colors=96))
        i+=1
    cap.release();assert frames;frames[0].save(out,save_all=True,append_images=frames[1:],duration=200,loop=0,optimize=True)
    return dict(gif=str(out.relative_to(ROOT)),source=str(src.relative_to(ROOT)),start_s=start_s,end_s=end_s,bytes=out.stat().st_size,frames=len(frames),caption='Recorded high-level keyboard script, not a human driving experiment. ROM surface visualization only.')

def main():
    root=ROOT/'upgrade/results';out=ROOT/'upgrade/media';videos=[]
    for level,view,name in [('MILD','world','mild'),('BUMPS','world','bumps'),('BUMPS','liquid','liquid'),('COURSE','world','course_failure')]:
        folder=root/f'{level}_2618001'
        if (folder/'result.json').exists():videos.append(render(folder,out/(name+'.mp4'),view))
    assets=[]
    for src,name,lo,hi in [('mild','mild_rough',0,None),('bumps','smooth_bumps',0,None),('bumps','keyboard_live_trail',0,None),('liquid','liquid_acceleration_braking',1,18),('liquid','turning_with_liquid',17,24)]:assets.append(gif(out/(src+'.mp4'),out/(name+'.gif'),lo,hi))
    with (ROOT/'docs/v2/media_provenance.json').open('x') as f:json.dump(dict(videos=videos,gifs=assets,SPH_comparison_GIF='NOT_GENERATED_STATIC_GATE_FAILED',surface_not_independently_validated=True),f,indent=2)
    print('UPGRADE_NATIVE_MEDIA_COMPLETE',flush=True)
if __name__=='__main__':main()
