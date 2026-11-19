"""Native physical evaluation with time-ordered completion and prefix-only liquid metrics."""
import time
import numpy as np,mujoco,torch
from .common import *
from .env import TrackEnv
from .reference import reference
from .terrain import sample as terrain_sample
from ballbot_fast_rl.outer_loop.control import kinematics
from ballbot_fast_rl.contact_validation.diagnostics import contact_sample,solver_residuals
from ballbot_sloshing_research.phase4.energy_diagnostics import sample as energy_sample
SCHEMA=['t','px','py','vx','vy','refx','refy','refvx','refvy','refax','refay','roll','pitch','yaw','qx','qy','qvx','qvy','E','tau0','tau1','tau2','wheel0','wheel1','wheel2','base0','base1','base2','res0','res1','res2','desired_roll','desired_pitch','accx','accy']
def policies():
 p=[dict(label=l,kind=l,path=None,hold=1) for l in ['PID_balance','V2','V3','V2_FFv1','V3_FFv1']]
 p.append(dict(label='teacher_Phase6',kind='teacher',path=None,hold=5))
 p.append(dict(label='BC_T10',kind='PPO',path=str(P9/'initialization/BC_initial.zip'),hold=1))
 for run in read(P9/'training_summary.json')['runs']:p.append(dict(label='PPO_'+str(run['seed']),kind='PPO',path=run['best_checkpoint'],hold=1))
 p8=PKG/'results/phase8_rl_20261102_stable_v1'
 for run in read(p8/'training_summary.json')['runs']:
  if run['variant']=='reference_KL':p.append(dict(label='KL_'+str(run['seed']),kind='PPO',path=run['best_checkpoint'],hold=1))
 p7=PKG/'results/phase7_20261102_sparse_v1'
 for run in read(p7/'sparse_training.json')['runs']:p.append(dict(label='BC_sparse_'+str(run['seed']),kind='sparse',path=run['checkpoint'],hold=1))
 return p

def metrics(x,horizon,kind,A,B,T,reason,payload):
 t=x[:,0];err=x[:,1:3]-x[:,5:7];norm=np.linalg.norm(err,axis=1);vel=x[:,7:9];tangent=vel/np.maximum(np.linalg.norm(vel,axis=1)[:,None],1e-12);along=np.sum(err*tangent,axis=1);cross=err[:,0]*tangent[:,1]-err[:,1]*tangent[:,0]
 rms=lambda a:float(np.sqrt(np.mean(np.asarray(a)**2)))
 invalid=((np.linalg.norm(x[:,14:16],axis=1)>.0084)|(np.linalg.norm(x[:,11:13],axis=1)>.12)) if payload=='liquid' else np.zeros(len(x),bool)
 ids=np.flatnonzero(invalid);n=int(ids[0]) if len(ids) else len(x);prefix=x[:n];yaw=np.unwrap(x[:,13])-x[0,13]
 survived=bool(reason=='TIME_LIMIT' and t[-1]>=horizon-1e-8);safety=bool(survived and np.linalg.norm(x[:,11:13],axis=1).max()<=.12 and np.max(np.degrees(np.sqrt(np.mean(x[:,11:13]**2,axis=0))))<=5 and abs(yaw).max()<=np.deg2rad(10) and np.mean(np.any(abs(x[:,19:22])>=9.9,axis=1))<=.05 and not len(ids))
 development=A<.2;rmse_limit=.02 if development else .1;max_limit=.05 if development else .25;node_limit=.02 if development else .1
 actual_length=float(np.linalg.norm(np.diff(x[:,1:3],axis=0),axis=1).sum());ref_length=float(np.linalg.norm(np.diff(x[:,5:7],axis=0),axis=1).sum());nodes=[]
 if kind in ['figure8','circle']:
  for j in range(1,17):
   tj=T*j/16;win=abs(t-tj)<=T/32;nodes.append(bool(np.any(win & (norm<=node_limit))))
 ordered=bool(len(nodes)==16 and all(nodes));complete=bool(kind=='figure8' and horizon>=T-1e-9 and safety and rms(norm)<=rmse_limit and norm.max()<=max_limit and ordered and actual_length>=.7*ref_length)
 return dict(actual_s=float(t[-1]),survival=survived,safety=safety,completion=complete,reason=reason,XY_RMSE_m=rms(norm),cross_track_RMSE_m=rms(cross),along_track_RMSE_m=rms(along),max_XY_error_m=float(norm.max()),final_error_m=float(norm[-1]),actual_path_length_m=actual_length,reference_path_length_m=ref_length,reference_progress=float(min(t[-1]/T,1)),ordered_nodes_passed=sum(nodes),ordered_nodes_total=len(nodes),roll_RMS_deg=float(np.degrees(rms(x[:,11]))),pitch_RMS_deg=float(np.degrees(rms(x[:,12]))),tilt_max_deg=float(np.degrees(np.linalg.norm(x[:,11:13],axis=1).max())),yaw_drift_max_deg=float(np.degrees(abs(yaw).max())),speed_RMS_m_s=rms(np.linalg.norm(x[:,3:5],axis=1)),torque_RMS_Nm=rms(x[:,19:22]),torque_peak_Nm=float(abs(x[:,19:22]).max()),saturation_fraction=float(np.mean(np.any(abs(x[:,19:22])>=9.9,axis=1))),E_mean_J=float(prefix[:,18].mean()) if len(prefix) else None,E_peak_J=float(prefix[:,18].max()) if len(prefix) else None,q_RMS_m=rms(np.linalg.norm(prefix[:,14:16],axis=1)) if len(prefix) else None,q_peak_m=float(np.linalg.norm(prefix[:,14:16],axis=1).max()) if len(prefix) else None,q_observed_peak_m=float(np.linalg.norm(x[:,14:16],axis=1).max()),ROM_violation=bool(len(ids)),ROM_first_passage_s=float(t[n]) if len(ids) else None,ROM_valid_duration_s=float(t[n-1]) if n else 0.,liquid_metric_scope='Valid prefix only; rigid zeros are not sloshing suppression successes; no energy ranking of failed tracking.')

def run(job):
 torch.set_num_threads(1);spec=job['policy'];label=spec['label'];cfg={k:job[k] for k in ['payload','amplitude','terrain_seed','kind','A','B','T','duration']};backbone=spec.get('backbone',label if label in ['V2','V3','V2_FFv1','V3_FFv1'] else 'V2');e=TrackEnv(controller=backbone,**cfg);o,initial=e.reset(seed=job['seed'],options=job.get('initial_liquid'))
 model=teacher=pid=None
 if spec['kind']=='PPO':
  from stable_baselines3 import PPO
  meta=read(Path(spec['path']).with_suffix('.contract.json'));assert meta['observation']=='P2_O1_V1' and meta['alpha']==.03 and meta['backbone']=='V2'
  model=PPO.load(spec['path'],device='cpu');assert model.observation_space.shape==(23,) and model.action_space.shape==(3,)
 if spec['kind']=='sparse':
  from ballbot_sloshing_research.phase7.sparse_bc import Student
  model=Student(spec['path'])
 if spec['kind']=='teacher':
  from ballbot_sloshing_research.phase6.teacher import Teacher
  teacher=Teacher()
 if spec['kind']=='PID_balance':
  from chrono_fsi.phase2.controllers import ReferencePIDAdapter
  pid=ReferencePIDAdapter(.002,True);e.residual=False;e.policy_steps=1
 rows=[];qp=[];qv=[];contacts=[];powers=[];latency=[];observations=[];pol=[];start=time.monotonic();deadline=120 if teacher else 90;deadline=min(deadline,job.get('wall_s',deadline));warning=e.data.warning.number.copy();action=np.zeros(3);i=0;previous_velocity=kinematics(e)['v'][:2].copy()
 def collect(env):
  nonlocal previous_velocity
  s=kinematics(env);p,v,a=env.target();q,qd=env.fluid_state();base=env.base_action;desired=env.backbone.tilt;acc=(s['v'][:2]-previous_velocity)/.002;previous_velocity=s['v'][:2].copy()
  rows.append(np.r_[env.data.time,s['p'][:2],s['v'][:2],p,v,a,s['rpy'],q,qd,env.slosh_energy(),env.data.actuator_force[env.motor_ids],env.data.qvel[env.wheel_dofs],base,action,desired,acc]);qp.append(env.data.qpos.copy());qv.append(env.data.qvel.copy())
  if env.steps%5==0:
   cs,ss=contact_sample(env);height,normal=terrain_sample(env.heights,*env.data.geom_xpos[env.ball_geom,:2]);ground=[c for c in cs if c['pair']=='ground'];contacts.append(dict(t=float(env.data.time),height_m=height,analytic_normal=normal.tolist(),mapping_gap=ss['contact_generalized_force_max_abs_gap_native_units'],power_gap_W=ss['contact_power_gap_W'],normal_force_N=sum(c['normal_force_N'] for c in ground),tangential_slip_m_s=max((c['tangential_slip_m_s'] for c in ground),default=0),ground_contact=bool(ss['ground_contact_count']),wheel_contact_count=ss['wheel_contact_count'],normal=[ground[0][f'normal_{k}'] for k in ['x','y','z']] if ground else None,penetration_m=min((c['gap_m'] for c in ground),default=0),solver_iterations=ss['solver_iterations'],solver_gradient=ss['solver_normalized_gradient']))
   if env.payload=='liquid':
    d=energy_sample(env);diag=env.diagnostics();powers.append(np.r_[env.data.time,d['Pin'],d['powers'][-1],d['identity_W'],diag['wrench']])
 e.sample_hook=collect
 try:
  while e.steps<e.max_steps:
   if i%spec['hold']==0:
    tick=time.perf_counter()
    if teacher:action=teacher.act(e)
    elif pid:action=pid.act(e,{'orientation':e.state()['rotvec']})
    elif model:
     action=model.predict(o,deterministic=True)[0] if spec['kind']=='PPO' else model.predict(o)
    else:action=np.zeros(3)
    latency.append(time.perf_counter()-tick)
   observations.append(o.copy());pol.append(np.r_[e.data.time,action]);o,_,term,trunc,info=e.step(action);i+=1
   if term or trunc:break
   if time.monotonic()-start>deadline:info['termination_reason']='WALL_BUDGET';break
  x=np.array(rows);row=metrics(x,cfg['duration'],cfg['kind'],cfg['A'],cfg['B'],cfg['T'],info['termination_reason'],cfg['payload']);row.update(experiment=ID,controller=label,stage=job['stage'],seed=job['seed'],**cfg,wall_s=time.monotonic()-start,physics_dt=.002,policy_dt=.002*e.policy_steps,policy_hold_macro=spec['hold'],model_hash=e.model_hash,heightfield_sha256=__import__('hashlib').sha256(e.model.hfield_data.tobytes()).hexdigest(),total_mass_kg=float(e.model.body_mass.sum()),fluid_mass_kg=float(sum(e.model.body_mass[b] for b in e.fluid_ids)),initial_qpos=initial['initial_qpos'].tolist(),initial_qvel=initial['initial_qvel'].tolist(),inference_mean_ms=float(np.mean(latency)*1000),inference_p99_ms=float(np.percentile(latency,99)*1000),missed_controller_deadlines=int(sum(v>.002*e.policy_steps*spec['hold'] for v in latency)),inference_calls=len(latency),numerical_warnings=int((e.data.warning.number-warning).sum()),reference_contract='TRACK8_ANALYTIC_WORLD_XY_V1',actor_contract=e.obs_contract,checkpoint=spec['path'],checkpoint_sha256=sha(spec['path']) if spec['path'] else None)
  row['backbone']=backbone;row['contract_status']='DECLARED_FF_BACKBONE_TRANSFER' if spec.get('backbone') else 'ORIGINAL_BACKBONE_AND_ACTOR_INTERFACE' if spec['kind'] in ['PPO','sparse','teacher'] else 'VERSIONED_FF_ADAPTER' if 'FFv1' in label else 'BALANCE_ONLY_NOT_TRACKING' if pid else 'ORIGINAL_POSITION_REFERENCE_INTERFACE'
  if row['numerical_warnings']:row['completion']=False;row['safety']=False
  row.update(ground_contact_fraction=float(np.mean([c['ground_contact'] for c in contacts])) if contacts else None,ground_slip_RMS_m_s=float(np.sqrt(np.mean([c['tangential_slip_m_s']**2 for c in contacts]))) if contacts else None,ground_normal_force_peak_N=max((c['normal_force_N'] for c in contacts),default=None),contact_force_mapping_gap_max=max((c['mapping_gap'] for c in contacts),default=None),contact_power_gap_max_W=max((abs(c['power_gap_W']) for c in contacts),default=None),terrain_info=e.terrain_info)
  power=np.array(powers);valid=power[power[:,0]<=row['ROM_valid_duration_s']+1e-9] if len(power) else power
  row['generalized_input_work_endpoint_J']=float(np.trapezoid(valid[:,1],valid[:,0])) if len(valid)>1 else None;row['energy_identity_peak_W']=float(abs(valid[:,3]).max()) if len(valid) else None
  folder=ROOT/'runs'/job['stage']/label/f"{cfg['kind']}_{cfg['payload']}_{cfg['amplitude']}_{cfg['A']}_{job['seed']}";folder.mkdir(parents=True,exist_ok=False)
  np.savez_compressed(folder/'trace.npz',rows=x,schema=np.array(SCHEMA),qpos=np.array(qp),qvel=np.array(qv),observations=np.array(observations),policy=np.array(pol),power_wrench=power)
  row['trace_path']=str(folder/'trace.npz');row['trace_sha256']=sha(folder/'trace.npz');row['xml_path']=str(folder/'model.xml');__import__('shutil').copyfile(e.xml_path,folder/'model.xml');np.save(folder/'heightfield.npy',e.model.hfield_data)
  with (folder/'contacts.json').open('x') as f:json.dump(contacts,f,allow_nan=False)
  if teacher:
   with (folder/'teacher.json').open('x') as f:json.dump(teacher.records,f,allow_nan=False)
  with (folder/'result.json').open('x') as f:json.dump(row,f,indent=2,allow_nan=False)
  print('TRACK8',job['stage'],label,cfg['kind'],cfg['payload'],cfg['amplitude'],job['seed'],row['actual_s'],row['XY_RMSE_m'],row['completion'],row['reason'],flush=True)
  return row
 finally:e.close()
