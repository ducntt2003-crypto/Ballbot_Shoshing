"""Matched 500Hz plant episodes, complete physical metrics and explicit references."""
import json,csv,time
from pathlib import Path
import numpy as np,torch
from ballbot_fast_rl.position_hold.core import PositionEnv,ROOT as PREVIOUS,CK,load
from ballbot_fast_rl.contact_validation.study import StudyEnv
from ballbot_fast_rl.contact_validation.diagnostics import contact_sample
from chrono_fsi.phase2.controllers import ReferencePIDAdapter
from .control import Cascade,kinematics
from .pilot import ROOT,orient
CANDIDATE=Path(json.loads((PREVIOUS/'selection_fixed_sensor.json').read_text())['candidate_checkpoint'])
SENSOR_POSE=json.loads((PREVIOUS/'candidate_export/metadata.json').read_text())['sensor_pose']
CONTROLLERS=['pid_original','pid_bumpless','original_ppo_legacy','finetuned_ppo_fixed','cascade']
COLS=['time','roll','pitch','yaw','px','py','pz','vx','vy','vz','sensor_vx','sensor_vy','sensor_vz','epx','epy','evx','evy','Ix','Iy','ref_roll','ref_pitch','inner_Ir','inner_Ip','acmd_x','acmd_y','tilt_clamped','slew_limited','outer_aw','motor_saturated','a0','a1','a2','tau0','tau1','tau2','ball_omega_x','ball_omega_y','ball_omega_z']

def episode(controller,seed,cfg,gains,model=None,save=None):
    cfg=dict(cfg);yaw=cfg.pop('yaw',0.);displace=np.array(cfg.pop('initial_displacement',[0,0]),float);velocity=np.array(cfg.pop('initial_velocity',[0,0]),float);options=cfg.pop('reset_options',None)
    e=StudyEnv(cfg) if controller=='original_ppo_legacy' else PositionEnv(cfg);obs,_=e.reset(seed=seed,options=options)
    if yaw:orient(e,yaw)
    for qa,da in [(e.base_q,e.base_d),(e.ball_q,e.ball_d)]:e.data.qpos[qa:qa+2]+=displace;e.data.qvel[da:da+2]+=velocity
    import mujoco
    mujoco.mj_forward(e.model,e.data);obs=e.observation();s0=kinematics(e);target=e.origin[:2].copy();pid=ReferencePIDAdapter(e.dt,controller=='pid_bumpless');cascade=Cascade(gains);cascade.reset(s0,target)
    states=[];slips=[];lat=[];start=time.monotonic();yaw0=s0['rpy'][2]
    if model is None and 'ppo' in controller:model=load(CK if controller=='original_ppo_legacy' else CANDIDATE)
    while e.steps<e.max_steps:
        tick=time.perf_counter();s=kinematics(e)
        log=dict(ep=target-s['p'][:2],ev=-s['v'][:2],integral=np.zeros(2),desired_tilt=np.zeros(2),inner_integral=np.zeros(2),a_cmd=np.zeros(2),tilt_clamped=False,slew_limited=False,outer_aw=False,motor_saturated=False)
        if controller=='cascade':action,log=cascade.act(s)
        elif controller.startswith('pid_'):action=pid.act(e,{'orientation':e.state()['rotvec']})
        else:action=model.predict(obs,deterministic=True)[0]
        lat.append(time.perf_counter()-tick);obs,_,term,trunc,info=e.step(action);s=kinematics(e)
        # Sensor velocity always reported in fixed physical nominal S; not legacy random inertial axes.
        omega=s['omega_B'];sensorp=np.array(SENSOR_POSE['position_B']);vs=np.array(SENSOR_POSE['R_BS']).T@(s['R'].T@s['v']+np.cross(omega,sensorp))
        states.append(np.r_[e.data.time,s['rpy'],s['p']-e.origin,s['v'],vs,target-s['p'][:2],-s['v'][:2],log['integral'],log['desired_tilt'],log['inner_integral'],log['a_cmd'],log['tilt_clamped'],log['slew_limited'],log['outer_aw'],log['motor_saturated'],action,e.data.actuator_force[e.motor_ids],e.data.qvel[e.ball_d+3:e.ball_d+6]])
        if e.steps%10==0 or term or trunc:
            cs,_=contact_sample(e)
            for c in cs:
                if c['pair'].startswith('wheel_'):slips.append([e.data.time,c['axial_low_mu_slip_m_s'],c['rolling_high_mu_slip_m_s']])
        if time.monotonic()-start>120:term=True;info['termination_reason']='WALL_CLOCK_BUDGET';info['success']=False
        if term or trunc:break
    x=np.asarray(states);get=lambda name:x[:,COLS.index(name)];rms=lambda v:float(np.sqrt(np.mean(np.asarray(v)**2)));pos=x[:,4:6];vel=x[:,7:9];err=x[:,13:15];torque=x[:,32:35];late=get('time')>=max(0,float(e.data.time)-2)
    yawtrace=np.unwrap(get('yaw'))-yaw0
    # Preserve exact legacy body-projected workspace/displacement semantics for strict gate.
    from scipy.spatial.transform import Rotation
    rot=Rotation.from_euler('xyz',x[:,1:4]).as_matrix();bodyerror=np.einsum('nji,nj->ni',rot,x[:,4:7]);displacement=np.linalg.norm(bodyerror[:,:2],axis=1)
    r=dict(controller=controller,seed=seed,horizon_s=cfg['duration_s'],survival=bool(info['success']),survival_s=float(e.data.time),reason=info['termination_reason'],roll_RMS_deg=float(np.rad2deg(rms(get('roll')))),pitch_RMS_deg=float(np.rad2deg(rms(get('pitch')))),peak_tilt_deg=float(np.rad2deg(np.max(np.arccos(np.clip(rot[:,2,2],-1,1))))),position_RMSE_m=rms(np.linalg.norm(err,axis=1)),max_displacement_m=float(displacement.max()),max_world_displacement_m=float(np.linalg.norm(pos,axis=1).max()),final_drift_m=float(np.linalg.norm(err[-1])),XY_velocity_RMS_m_s=rms(np.linalg.norm(vel,axis=1)),steady_position_RMSE_m=rms(np.linalg.norm(err[late],axis=1)),steady_position_mean_W_m=err[late].mean(axis=0).tolist(),steady_velocity_RMS_m_s=rms(np.linalg.norm(vel[late],axis=1)),steady_velocity_mean_W_m_s=vel[late].mean(axis=0).tolist(),yaw_drift_max_deg=float(np.rad2deg(abs(yawtrace).max())),torque_RMS_Nm=rms(torque),torque_peak_Nm=float(abs(torque).max()),saturation_fraction=float(np.mean(np.any(abs(torque)>=9.9,axis=1))),torque_variation_Nm=float(np.sum(abs(np.diff(torque,axis=0)))),integral_max_m_s=float(np.max(np.linalg.norm(x[:,17:19],axis=1))),desired_tilt_peak_rad=float(np.max(np.linalg.norm(x[:,19:21],axis=1))),tilt_clamp_fraction=float(get('tilt_clamped').mean()),outer_aw_fraction=float(get('outer_aw').mean()),motor_aw_fraction=float(get('motor_saturated').mean()),latency_p50_ms=float(np.percentile(lat,50)*1000),latency_p99_ms=float(np.percentile(lat,99)*1000),latency_max_ms=float(max(lat)*1000),wall_s=time.monotonic()-start,mass_kg=float(e.model.body_mass.sum()),model_hash=e.model_hash,ballast_offset_m=cfg.get('robot_params',{}).get('ballast_offset_m',[0,0,0]),ballast_mass_kg=3.2)
    r['combined_COM_shift_m']=(3.2/float(e.model.body_mass.sum())*np.array(r['ballast_offset_m'])).tolist()
    r['strict_pass']=bool(r['survival'] and r['roll_RMS_deg']<5 and r['pitch_RMS_deg']<5 and r['saturation_fraction']<=.05 and r['max_displacement_m']<=.3 and r['yaw_drift_max_deg']<=10)
    slip=np.array(slips);r['slip_low_RMS_m_s']=rms(slip[:,1]) if len(slip) else None;r['slip_high_RMS_m_s']=rms(slip[:,2]) if len(slip) else None
    startsettle=cfg.get('push_start_s',0)+cfg.get('push_duration_s',0) if cfg.get('push_N',0)>0 else 1.
    good=(np.linalg.norm(err,axis=1)<.01)&(np.linalg.norm(vel,axis=1)<.01);valid=get('time')>=startsettle;bad=np.flatnonzero(valid&~good);idx=(bad[-1]+1) if len(bad) else np.searchsorted(get('time'),startsettle)
    r['settling_time_s']=float(get('time')[idx]-startsettle) if idx<len(x) and float(e.data.time)-get('time')[idx]>=.5 else None
    if save is not None:
        save=Path(save);save.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(save,trajectory=x,slips=slip,columns=np.array(COLS),origin=e.origin,target=target)
        with save.with_suffix('.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(COLS);w.writerows(x)
        with save.with_suffix('.json').open('x') as f:json.dump(r,f,indent=2)
    e.close();return r

def summary(rows):
    result=dict(episodes=len(rows),survived=sum(r['survival'] for r in rows),strict=sum(r['strict_pass'] for r in rows),strict_rate=float(np.mean([r['strict_pass'] for r in rows])))
    for k,v in rows[0].items():
        if isinstance(v,(float,int)) and k not in ['seed','survival','strict_pass']:
            valid=[r[k] for r in rows if r[k] is not None];result[k]=dict(mean=float(np.mean(valid)),max=float(max(valid)))
    return result
