"""GLFW press/release keyboard application with timestamped inputs and native logs."""
from .runtime import TeleopEnv,kinematics
from .experiment import dump
import argparse,time,json,csv
from pathlib import Path
import mujoco,glfw,numpy as np

KEYS={glfw.KEY_UP:'UP',glfw.KEY_DOWN:'DOWN',glfw.KEY_LEFT:'LEFT',glfw.KEY_RIGHT:'RIGHT',glfw.KEY_W:'UP',glfw.KEY_S:'DOWN',glfw.KEY_A:'LEFT',glfw.KEY_D:'RIGHT',glfw.KEY_SPACE:'SPACE'}
class KeyboardState:
    def __init__(self):self.codes=set();self.paused=False;self.reset=False;self.exit=False;self.camera=1;self.events=0
    @property
    def keys(self):return {KEYS[k] for k in self.codes if k in KEYS}
    def event(self,key,action):
        self.events+=1
        if action==glfw.RELEASE:self.codes.discard(key);return
        if action not in [glfw.PRESS,glfw.REPEAT]:return
        if key==glfw.KEY_SPACE:self.codes.clear()
        if key in KEYS:self.codes.add(key)
        if action==glfw.PRESS:
            if key==glfw.KEY_P:self.paused=not self.paused;self.codes.clear()
            elif key==glfw.KEY_R:self.reset=True;self.codes.clear()
            elif key==glfw.KEY_ESCAPE:self.exit=True
            elif glfw.KEY_1<=key<=glfw.KEY_4:self.camera=key-glfw.KEY_0
    def focus(self,focused):
        if not focused:self.codes.clear()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=None);p.add_argument('--smoke-seconds',type=float);p.add_argument('--seed',type=int,default=2618001);a=p.parse_args()
    out=Path(a.output or ('runs/keyboard_'+time.strftime('%Y%m%d_%H%M%S')));out.mkdir(parents=True,exist_ok=False)
    if not glfw.init():raise RuntimeError('GLFW initialization failed: run on a graphical desktop with DISPLAY/Wayland access.')
    window=glfw.create_window(1000,700,'SloshBot-RL | arrows/WASD, SPACE stop, P pause, R reset, 1-4 camera, ESC exit',None,None)
    if not window:glfw.terminate();raise RuntimeError('OpenGL window unavailable; headless command replay remains supported.')
    glfw.make_context_current(window);glfw.swap_interval(0);env=TeleopEnv(terrain='playground',duration=3600,capture=False);env.reset(seed=a.seed)
    scene=mujoco.MjvScene(env.e.model,maxgeom=1000);context=mujoco.MjrContext(env.e.model,mujoco.mjtFontScale.mjFONTSCALE_100);option=mujoco.MjvOption();cam=mujoco.MjvCamera();keys=KeyboardState();start=time.monotonic();frames=0;steps=0;episode=0;reason='USER_EXIT';lastframe=start;due=start;lastrows=0;maxlag=0.
    inputs=(out/'keyboard_events.jsonl').open('x');states=(out/'states.csv').open('x',newline='');writer=csv.writer(states);writer.writerow(['episode','t','forward_key','left_key','desired_forward','desired_turn','desired_vx','desired_vy','vx','vy','yawrate','roll','pitch','yaw','desired_yaw','E','qx','qy','qdx','qdy','tau0','tau1','tau2','ground','body_ground','gap','refx','refy'])
    def callback(win,key,scancode,action,mods):
        keys.event(key,action);inputs.write(json.dumps(dict(wall_elapsed_s=time.monotonic()-start,sim_t=float(env.e.data.time),episode=episode,key=key,action=action,keys=sorted(keys.keys)))+'\n');inputs.flush()
    def focus(win,flag):keys.focus(flag)
    glfw.set_key_callback(window,callback);glfw.set_window_focus_callback(window,focus)
    try:
        while not glfw.window_should_close(window) and not keys.exit:
            now=time.monotonic();glfw.poll_events()
            if a.smoke_seconds is not None and now-start>=a.smoke_seconds:reason='BOUNDED_GUI_SMOKE_END';break
            if keys.reset:
                env.reset(seed=a.seed);episode+=1;lastrows=0;keys.reset=False;due=now
            if keys.paused:due=now
            elif now>=due:
                env.command.request_keys(keys.keys);_,_,term,trunc,info=env.step(np.zeros(3));steps+=1;maxlag=max(maxlag,now-due);due+=.01
                for row in env.telemetry[lastrows:]:writer.writerow([episode,*row])
                lastrows=len(env.telemetry);states.flush()
                if len(env.rows)>1000:
                    env.rows.clear();env.logger.micro.clear();env.logger.policies.clear();env.logger.obs.clear();env.logger.policy_qpos.clear();env.logger.policy_qvel.clear();env.logger.policy_memory.clear();env.policy_obs31.clear();env.effective_tau_history.clear();env.reward_rows.clear()
                # Bound in-memory data; CSV remains complete.
                if len(env.telemetry)>1000:env.telemetry.clear();lastrows=0
                if term or trunc:reason=info['termination_reason'];break
            if now-lastframe>=1/30:
                s=kinematics(env.e);cam.lookat[:]=s['p'] if keys.camera==1 else [0,0,.2];cam.distance=1.5 if keys.camera==1 else 4.;cam.azimuth=125 if keys.camera!=4 else 90;cam.elevation=-89 if keys.camera==3 else -8 if keys.camera==4 else -25
                mujoco.mjv_updateScene(env.e.model,env.e.data,option,None,cam,mujoco.mjtCatBit.mjCAT_ALL,scene);w,h=glfw.get_framebuffer_size(window);rect=mujoco.MjrRect(0,0,w,h);mujoco.mjr_render(rect,scene,context)
                text=f't={env.e.data.time:.2f}s  {"PAUSED" if keys.paused else "RUNNING"}  keys={sorted(keys.keys)}\nspeed={np.linalg.norm(s["v"][:2]):.3f}m/s  roll/pitch={s["rpy"][:2]}\nyaw={s["rpy"][2]:.3f}rad  E={env.e.slosh_energy():.6g}J  tau={env.e.data.ctrl}\nModal ROM only; no resolved fluid. Cameras 1 chase / 2 iso / 3 top / 4 side'
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,rect,text,'',context);glfw.swap_buffers(window);frames+=1;lastframe=now
            time.sleep(.001)
    finally:
        elapsed=time.monotonic()-start;dump(out/'session.json',dict(termination=reason,wall_s=elapsed,simulation_s=float(env.e.data.time),frames=frames,rendered_FPS=frames/max(elapsed,1e-9),policy_steps=steps,native_steps=env.meter_native_steps,max_scheduler_lag_s=maxlag,measured_realtime_factor=float(env.e.data.time)/max(elapsed,1e-9),human_input_observed=keys.events>0,notes='Smoke mode verifies window/render/closed-loop only, not human usability or sustained realtime.'))
        inputs.close();states.close();context.free();env.close();glfw.destroy_window(window);glfw.terminate()
    print('GUI_SESSION',out,reason,flush=True)
if __name__=='__main__':main()
