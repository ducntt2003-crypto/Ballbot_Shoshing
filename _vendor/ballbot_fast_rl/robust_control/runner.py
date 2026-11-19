"""Controllers receive Measurements only. Ground truth confined to devices, task metrics, plant."""
import json,time,csv
from types import SimpleNamespace
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from ballbot_fast_rl.position_hold.core import PositionEnv,CK,load
from ballbot_fast_rl.outer_loop.control import Cascade,kinematics
from ballbot_fast_rl.outer_loop.evaluation import CANDIDATE,SENSOR_POSE
from chrono_fsi.phase2.controllers import ReferencePIDAdapter
from .config import ROOT,GAINS
from .sensors import SimulatedDevices,MeasurementPipeline,PredictionPipeline,Measurement

def old_observation(m,reference,previous):
    pB=m.R.T@(m.p-reference);vB=m.R.T@m.v+np.cross(m.omega_B,np.array(SENSOR_POSE['position_B']))
    return np.clip(np.r_[m.R.T@[0,0,-1],m.omega_B/5,m.wheels/10,pB[:2]/.5,vB[:2],previous,[0,0]],-10,10).astype(np.float32)
class World:
    """Simulation boundary, separate from measurement-only controllers."""
    def __init__(self,condition,seed,duration):
        self.condition=condition;cfg=dict(duration_s=duration)
        if condition.get('robot_params'):cfg['robot_params']=condition['robot_params']
        self.env=PositionEnv(cfg);self.env.reset(seed=seed);self.dt=.002;self.sensor_cfg=condition['sensor'];self.mode=self.sensor_cfg.get('mode','estimated');self.devices=SimulatedDevices(self.sensor_cfg,seed+420000) if self.mode!='perfect' else None;self.pipeline=(PredictionPipeline(self.sensor_cfg) if condition.get('prediction',False) else MeasurementPipeline(self.sensor_cfg)) if self.devices else None;self.ref=None;self.prev=np.zeros(3);self.seq=0
        self.gain=np.array(condition.get('motor_gains',[1,1,1]),float);self.measurement=self.measure()
    def measure(self):
        e=self.env;t=float(e.data.time);truth=kinematics(e)
        if self.mode=='perfect':
            self.seq+=1;m=Measurement(t,t,self.seq,truth['p'].copy(),truth['v'].copy(),truth['R'].copy(),truth['omega_B'].copy(),e.data.qvel[e.wheel_dofs].copy(),True,0.,np.zeros(6))
        else:
            self.devices.sample(t,truth,e.data.qvel[e.wheel_dofs]);self.pipeline.ingest(self.devices.receive(t));m=self.pipeline.read(t)
        if self.ref is None and m.valid:self.ref=m.p.copy()
        self.measurement=m;return m
    def step(self,requested):
        requested=np.clip(np.asarray(requested,float),-1,1);effective=np.clip(self.gain*requested,-1,1)
        # Unknown plant-side gain error, hard physical±10Nm limits retained. No true gain in controller.
        obs,reward,term,trunc,info=self.env.step(effective);self.prev=requested.copy();m=self.measure();return m,term,trunc,info
    def close(self):self.env.close()

def episode(controller,seed,condition,duration=10.,save=None,policy=None,collect=None,render=None):
    condition=dict(condition,prediction=controller=='cascade_predict' or condition.get('prediction',False));world=World(condition,seed,duration);e=world.env;m=world.measurement;c=Cascade(GAINS);pid=ReferencePIDAdapter(.002,False);started=False;rows=[];times=[];start=time.monotonic();yaw0=kinematics(e)['rpy'][2]
    if policy is None and controller in ['original_ppo','position_ppo']:policy=load(CK if controller=='original_ppo' else CANDIDATE)
    while e.steps<e.max_steps:
        tick=time.perf_counter();m=world.measurement;log=dict(integral=c.integral.copy(),desired_tilt=c.tilt.copy(),outer_aw=True,tilt_clamped=False,motor_saturated=False)
        if m.valid and not started:c.reset(m.control_state(),world.ref[:2]);started=True
        if controller in ['imitation','student_ppo']:a=policy.act(m,world.ref,world.prev)
        elif not m.valid:a=np.zeros(3)
        elif controller in ['cascade','cascade_predict']:a,log=c.act(m.control_state(),True)
        elif controller=='pid':a=pid.act(SimpleNamespace(dt=.002,data=SimpleNamespace(time=float(e.data.time))),{'orientation':Rotation.from_matrix(m.R).as_rotvec()})
        elif controller in ['original_ppo','position_ppo']:a=policy.predict(old_observation(m,world.ref,world.prev),deterministic=True)[0]
        else:raise ValueError(controller)
        if collect is not None:collect(m,world.ref,world.prev,a,log)
        times.append(time.perf_counter()-tick);m,term,trunc,info=world.step(a);gt=kinematics(e);ep=gt['p']-e.origin
        estp=m.p-gt['p'];estv=m.v-gt['v'];ori=Rotation.from_matrix(m.R.T@gt['R']).magnitude()
        rows.append(np.r_[e.data.time,gt['rpy'],ep,gt['v'],m.p-e.origin,m.v,m.omega_B,log['integral'],log['desired_tilt'],e.data.actuator_force[e.motor_ids],a,m.valid,m.age,estp,estv,ori,log['outer_aw'],log['tilt_clamped'],log['motor_saturated'],m.source_timestamp,m.timestamp])
        if render is not None:render(world,a,log,info,term or trunc)
        if time.monotonic()-start>120:term=True;info['termination_reason']='WALL_BUDGET';info['success']=False
        if term or trunc:break
    x=np.array(rows);cols=['t','roll','pitch','yaw','px','py','pz','vx','vy','vz','mpx','mpy','mpz','mvx','mvy','mvz','gyro_x','gyro_y','gyro_z','Ix','Iy','ref_roll','ref_pitch','tau0','tau1','tau2','a0','a1','a2','valid','age','err_px','err_py','err_pz','err_vx','err_vy','err_vz','orientation_error','outer_aw','tilt_clamp','motor_sat','source_stamp','estimate_stamp']
    rms=lambda y:float(np.sqrt(np.mean(np.asarray(y)**2)));valid=x[:,29]>.5;late=x[:,0]>=max(0,e.data.time-2);rot=Rotation.from_euler('xyz',x[:,1:4]).as_matrix();bodyerror=np.einsum('nji,nj->ni',rot,x[:,4:7]);displacement=np.linalg.norm(bodyerror[:,:2],axis=1);tau=x[:,23:26];yaw=np.unwrap(x[:,3])-yaw0
    r=dict(controller=controller,seed=seed,duration_s=duration,survival=bool(info['success']),survival_s=float(e.data.time),reason=info['termination_reason'],roll_RMS_deg=np.rad2deg(rms(x[:,1])),pitch_RMS_deg=np.rad2deg(rms(x[:,2])),max_displacement_m=float(displacement.max()),position_RMSE_m=rms(np.linalg.norm(x[:,4:6],axis=1)),final_position_error_m=float(np.linalg.norm(x[-1,4:6])),steady_position_RMSE_m=rms(np.linalg.norm(x[late,4:6],axis=1)),steady_position_mean_W_m=x[late,4:6].mean(axis=0).tolist(),velocity_RMS_m_s=rms(np.linalg.norm(x[:,7:9],axis=1)),steady_velocity_RMS_m_s=rms(np.linalg.norm(x[late,7:9],axis=1)),yaw_drift_max_deg=float(np.rad2deg(abs(yaw).max())),torque_RMS_Nm=rms(tau),torque_peak_Nm=float(abs(tau).max()),saturation_fraction=float(np.mean(np.any(abs(tau)>=9.9,axis=1))),integral_peak_m_s=float(np.linalg.norm(x[:,19:21],axis=1).max()),desired_tilt_peak_rad=float(np.linalg.norm(x[:,21:23],axis=1).max()),outer_aw_fraction=float(x[:,38].mean()),tilt_clamp_fraction=float(x[:,39].mean()),valid_fraction=float(valid.mean()),source_age_max_s=float(x[:,30].max()),sensor_position_RMSE_m=rms(np.linalg.norm(x[valid,31:34],axis=1)) if valid.any() else None,sensor_velocity_RMSE_m_s=rms(np.linalg.norm(x[valid,34:37],axis=1)) if valid.any() else None,orientation_error_RMS_deg=float(np.rad2deg(rms(x[valid,37]))) if valid.any() else None,control_latency_p99_ms=float(np.percentile(times,99)*1000),wall_s=time.monotonic()-start,model_hash=e.model_hash,model_mass_kg=float(e.model.body_mass.sum()))
    r['strict_pass']=bool(r['survival'] and r['roll_RMS_deg']<5 and r['pitch_RMS_deg']<5 and r['max_displacement_m']<=.30 and r['saturation_fraction']<=.05 and r['yaw_drift_max_deg']<=10)
    r['CoM_settling_pass']=bool(r['survival'] and r['steady_position_RMSE_m']<=.02 and r['steady_velocity_RMS_m_s']<=.005)
    if save is not None:
        p=Path(save);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,trajectory=x,columns=np.array(cols));
        with p.with_suffix('.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(cols);w.writerows(x)
        p.with_suffix('.json').write_text(json.dumps(r,indent=2))
    world.close();return r

def summarize(rows):
    result=dict(episodes=len(rows),survived=sum(r['survival'] for r in rows),strict=sum(r['strict_pass'] for r in rows),strict_rate=float(np.mean([r['strict_pass'] for r in rows])),settled=sum(r['CoM_settling_pass'] for r in rows))
    for key in rows[0]:
        if key in ['seed','survival','strict_pass','CoM_settling_pass']:continue
        vv=[r[key] for r in rows if isinstance(r[key],(float,int)) and not isinstance(r[key],bool)]
        if vv:result[key]=dict(mean=float(np.mean(vv)),max=float(max(vv)))
    return result
