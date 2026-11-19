"""Render saved native states only. No integration, fabricated trajectory or CFD."""
from .bootstrap import ROOT,setup
import os
os.environ.setdefault('MUJOCO_GL','egl')
setup()
import mujoco,cv2,numpy as np,json,time
from PIL import Image
from pathlib import Path

def load_model(folder):
    text=(folder/'model.xml').read_text().replace('__ASSET_ROOT__',str(ROOT/'_vendor/ballbotgym/ballbotgym/assets'))
    m=mujoco.MjModel.from_xml_string(text);m.hfield_data[:]=np.load(folder/'hfield.npy');return m

def line(scene,a,b,color):
    if scene.ngeom>=scene.maxgeom:return
    g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,2,np.array(a),np.array(b));scene.ngeom+=1

def render(folder,out):
    r=json.loads((folder/'result.json').read_text());z=np.load(folder/'trace.npz');raw=z['native_rows'];x=z['rows'];y=z['telemetry'] if 'telemetry' in z else np.empty((0,0));m=load_model(folder);d=mujoco.MjData(m);renderer=mujoco.Renderer(m,480,640,max_geom=1800);cam=mujoco.MjvCamera();cam.lookat[:]=[.15,0,.18];cam.distance=2.7;cam.elevation=-25;cam.azimuth=125
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,480));assert writer.isOpened();ids=list(range(0,len(raw),25))
    if ids[-1]!=len(raw)-1:ids.append(len(raw)-1)
    start=time.monotonic()
    try:
        for i in ids:
            assert time.monotonic()-start<120
            row=raw[i];d.qpos[:]=row[1:1+m.nq];d.qvel[:]=row[1+m.nq:1+m.nq+m.nv];d.time=row[0];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d);renderer.update_scene(d,camera=cam)
            if len(y):ref=y[:,25:27]
            else:ref=x[:,5:7]
            stride=max(1,len(ref)//150)
            for a,b in zip(ref[::stride][:-1],ref[::stride][1:]):line(renderer.scene,[*a,.01],[*b,.01],[1,.2,.1,1])
            for a,b in zip(x[:i+1:25,1:3][:-1],x[:i+1:25,1:3][1:]):line(renderer.scene,[*a,.015],[*b,.015],[.2,.5,1,1])
            im=renderer.render().copy();text=[f'SloshBot-RL | {r["controller"]} | saved-state native replay',f't={d.time:.3f}s; termination={r["termination"]}',f'Physical-valid={r["physical_valid"]}; modal ROM, NOT CFD']
            if len(y):
                a=y[min(i,len(y)-1)];text+=[f'Keys: forward={a[1]:+.0f} left={a[2]:+.0f} | prerecorded command replay',f'v_des={a[3]:.3f}m/s yawrate_des={a[4]:.3f}rad/s',f'tilt={np.linalg.norm(a[10:12]):.4f}rad E={a[14]:.5g}J']
            else:text+=[f'Full-task={r["metrics"]["completion"]}; {r["metrics"]["ordered_nodes_passed"]}/16 windows','Red=desired XY, blue=recorded XY; terrain unchanged']
            for k,t in enumerate(text):cv2.putText(im,t,(7,20+18*k),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1,cv2.LINE_AA)
            if i==ids[0]:cv2.imwrite(str(out.with_suffix('.png')),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
            writer.write(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
    finally:writer.release();renderer.close()
    cap=cv2.VideoCapture(str(out));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release();assert frames==len(ids)
    return dict(video=str(out.relative_to(ROOT)),source=str(folder.relative_to(ROOT)),frames=frames,fps=20,actual_last_recorded_sim_s=float(raw[-1,0]),native_stop_s=r.get('native_stop_time_s',r['native_steps']*.002 if not r.get('reused') else r['actual_s']),new_native_steps=0,recorded_replay=True,human_session=False,wall_s=time.monotonic()-start)

def gif(src,out):
    cap=cv2.VideoCapture(str(src));frames=[];i=0
    while True:
        ok,bgr=cap.read()
        if not ok:break
        if i%4==0:
            rgb=cv2.cvtColor(cv2.resize(bgr,(384,288)),cv2.COLOR_BGR2RGB);frames.append(Image.fromarray(rgb).quantize(colors=96))
        i+=1
    cap.release();assert frames;frames[0].save(out,save_all=True,append_images=frames[1:],duration=200,loop=0,optimize=True)
    return dict(gif=str(out.relative_to(ROOT)),frames=len(frames),sampled_fps=5,bytes=out.stat().st_size,source=str(src.relative_to(ROOT)))

def main():
    out=ROOT/'assets/videos';out.mkdir(exist_ok=False);jobs=[('experiment_1_figure8/PPO_TEST_1','ppo_figure8'),('experiment_1_figure8/V4_TEST_1','v4_figure8'),('experiment_2_teleoperation/replay_2618001','keyboard_baseline'),('experiment_2_teleoperation/modal_replay_2618001','keyboard_modal')]
    traverse=ROOT/'experiments/experiment_2_teleoperation/traverse_2618001'
    if (traverse/'result.json').exists():jobs.append(('experiment_2_teleoperation/traverse_2618001','rough_terrain'))
    records=[]
    for folder,name in jobs:records.append(render(ROOT/'experiments'/folder,out/(name+'.mp4')))
    assets=[]
    for video,name in [('ppo_figure8','experiment_1_figure8'),('keyboard_baseline','experiment_2_keyboard'),('rough_terrain' if (out/'rough_terrain.mp4').exists() else 'keyboard_modal','rough_terrain'),('ppo_figure8','ppo_demo')]:assets.append(gif(out/(video+'.mp4'),ROOT/'assets/gifs'/(name+'.gif')))
    with (ROOT/'docs/media_provenance.json').open('x') as f:json.dump(dict(videos=records,gifs=assets,figure8_case='fixed TEST_1',keyboard_case='fixed seed2618001; includes failure',trajectory_line_height_m='visual projection at 0.01/0.015 only; never collision geometry',no_fake_fluid=True),f,indent=2)
    print('NATIVE_MEDIA_COMPLETE',flush=True)
if __name__=='__main__':main()
