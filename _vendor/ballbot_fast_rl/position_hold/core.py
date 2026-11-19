import json,time
from pathlib import Path
import numpy as np,torch,mujoco
from stable_baselines3 import PPO
from ballbot_fast_rl.contact_validation.study import StudyEnv, CK
from ballbot_fast_rl.env import BallBotFastBalanceEnv
from scipy.spatial.transform import Rotation
from ballbot_fast_rl.contact_validation.diagnostics import contact_sample
from chrono_fsi.phase2.controllers import ReferencePIDAdapter
ROOT=Path(__file__).resolve().parents[1]/'results/position_hold_20261029'
WEIGHTS=dict(alive=1.,tilt=2.,angular_velocity=.1,position=1.5,velocity=1.,torque=.02,action_rate=.01)
REWARD_VERSION='position_hold_integral_v1'
OBS_VERSION='fast_balance_fixed_sensor_v2'
class PositionEnv(StudyEnv):
    def __init__(self,cfg=None):
        super().__init__(dict(cfg or {},reward_weights=WEIGHTS.copy()))
        # Freeze the NOMINAL sensor pose, independent of randomized mass/CAD inertia axes.
        nominal=BallBotFastBalanceEnv()
        self.sensor_p_B=nominal.model.body_ipos[nominal.base].copy()
        self.sensor_R_B=Rotation.from_quat(nominal.model.body_iquat[nominal.base][[1,2,3,0]]).as_matrix()
        nominal.close()
    def state(self):
        s=super().state()
        # XBODY is the regular body origin/orientation; BODY is the inertial frame.
        six=np.empty(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_XBODY,self.base,six,1)
        s['omega']=self.sensor_R_B.T@six[:3]
        s['velocity']=self.sensor_R_B.T@(six[3:]+np.cross(six[:3],self.sensor_p_B))
        return s

def gate(r):
    return bool(r['survival'] and r['roll_RMS_deg']<5 and r['pitch_RMS_deg']<5 and r['saturation_fraction']<=.05 and r['max_displacement_m']<=.30 and r['yaw_max_deg']<=10)
def load(path):
    torch.set_num_threads(1)
    return PPO.load(path,device='cpu')
def rollout(model,controller,seed,cfg,save=None):
    e=PositionEnv(cfg) if controller=='new_position_hold' else StudyEnv(cfg);o,info=e.reset(seed=seed);pid=ReferencePIDAdapter(e.dt,controller=='pid_bumpless');initial=e.state();yaw0=initial['roll_pitch_yaw'][2];values=[];slips=[];terms={};start=time.monotonic()
    for k in range(e.max_steps):
        a=pid.act(e,{'orientation':e.state()['rotvec']}) if controller.startswith('pid_') else model.predict(o,deterministic=True)[0]
        before=o.copy();o,reward,t,u,info=e.step(a);s=info['state'];R=e.data.xmat[e.base].reshape(3,3);pos=e.data.xpos[e.base]-e.origin
        six=np.empty(6);mujoco.mj_objectVelocity(e.model,e.data,mujoco.mjtObj.mjOBJ_BODY,e.base,six,0)
        # The legacy state velocity uses compiled inertial axes; candidate uses fixed
        # sensor axes. Also record origin velocity in W for fair physical metrics.
        jp=np.zeros((3,e.model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(e.model,e.data,jp,jr,e.data.xpos[e.base],e.base)
        velorigin=jp@e.data.qvel
        values.append(np.r_[s['time_s'],s['roll'],s['pitch'],s['yaw'],pos[:2],s['velocity'][:2],six[:3],s['torque'],a,e.data.qvel[e.ball_d+3:e.ball_d+6],velorigin[:2],before,o,six[3:5]])
        for key,val in info['reward_terms'].items():terms[key]=terms.get(key,0)+val
        if k%10==9 or t or u:
            cs,_=contact_sample(e)
            for c in cs:
                if c['pair'].startswith('wheel_'):slips.append([c['time_s'],c['axial_low_mu_slip_m_s'],c['rolling_high_mu_slip_m_s']])
        if t or u:break
    x=np.asarray(values);v=x[:,20:22];p=x[:,4:6];legacy_p=x[:,49:51]*.5;tau=x[:,11:14];yaw=np.unwrap(x[:,3])-yaw0
    rms=lambda y:float(np.sqrt(np.mean(np.asarray(y)**2)))
    late=x[:,0]>=min(1.,float(e.data.time)/2)
    row=dict(controller=controller,seed=seed,horizon_s=cfg['duration_s'],survival=bool(info['success']),survival_s=float(e.data.time),reason=info['termination_reason'],roll_RMS_deg=np.rad2deg(rms(x[:,1])),pitch_RMS_deg=np.rad2deg(rms(x[:,2])),peak_tilt_deg=float(np.rad2deg(max(np.hypot(x[:,1],x[:,2])))),yaw_max_deg=float(np.rad2deg(abs(yaw).max())),max_displacement_m=float(np.linalg.norm(legacy_p,axis=1).max()),max_world_XY_displacement_m=float(np.linalg.norm(p,axis=1).max()),final_position_error_m=float(np.linalg.norm(p[-1])),velocity_XY_RMS_m_s=rms(np.linalg.norm(v,axis=1)),final_velocity_W_origin_m_s=v[-1].tolist(),late_roll_pitch_mean_deg=np.rad2deg(x[late,1:3].mean(axis=0)).tolist(),late_velocity_W_origin_mean_m_s=v[late].mean(axis=0).tolist(),angular_velocity_RMS_rad_s=rms(np.linalg.norm(x[:,8:11],axis=1)),torque_RMS_Nm=rms(tau),torque_peak_Nm=float(abs(tau).max()),saturation_fraction=float(np.mean(np.any(abs(tau)>=9.9,axis=1))),torque_variation_Nm=float(np.sum(abs(np.diff(tau,axis=0)))),torque_mean_Nm=tau[late].mean(axis=0).tolist(),slip_low_RMS_m_s=rms(np.asarray(slips)[:,1]),slip_high_RMS_m_s=rms(np.asarray(slips)[:,2]),reward_terms=terms,wall_s=time.monotonic()-start,model_hash=e.model_hash,mass_kg=float(e.model.body_mass.sum()))
    row['strict_pass']=gate(row)
    if save is not None:
        save=Path(save);save.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(save,trajectory=x,slips=np.asarray(slips),origin_W=e.origin,schema=np.array(['time','roll','pitch','yaw','pW_x','pW_y','vSensor_x','vSensor_y','omegaW_x','omegaW_y','omegaW_z','tau0','tau1','tau2','a0','a1','a2','ball_omegaB_x','ball_omegaB_y','ball_omegaB_z','origin_vW_x','origin_vW_y']+[f'obs_before_{i}' for i in range(18)]+[f'obs_after_{i}' for i in range(18)]+['vW_x','vW_y']))
        save.with_suffix('.json').write_text(json.dumps(row,indent=2))
    e.close();return row

def summarize(rows):
    out=dict(episodes=len(rows),survived=sum(r['survival'] for r in rows),strict=sum(r['strict_pass'] for r in rows),strict_rate=float(np.mean([r['strict_pass'] for r in rows])))
    for k in ['roll_RMS_deg','pitch_RMS_deg','max_displacement_m','final_position_error_m','velocity_XY_RMS_m_s','yaw_max_deg','torque_RMS_Nm','torque_peak_Nm','saturation_fraction','slip_low_RMS_m_s','slip_high_RMS_m_s','survival_s']:
        out[k]=dict(mean=float(np.mean([r[k] for r in rows])),max=float(max(r[k] for r in rows)))
    return out
