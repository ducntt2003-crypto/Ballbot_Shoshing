"""One native physical flow; complete accounting and trusted local clone snapshots."""
import time,copy,pickle
import numpy as np,mujoco
from .common import *
from ballbot_sloshing_research.track8.env import TrackEnv
from ballbot_sloshing_research.track8.evaluate import metrics,SCHEMA
from ballbot_sloshing_research.phase4.energy_diagnostics import RefinedAccounting,Accounting,SCHEMA as ENERGY_SCHEMA,sample
from ballbot_fast_rl.contact_validation.diagnostics import contact_sample
from ballbot_sloshing_research.track8.terrain import sample as surface
from ballbot_fast_rl.outer_loop.control import kinematics
from ballbot_fast_rl.robust_control.v3.identify import snapshot,SPEC
class TimedEnv(TrackEnv):
 def __init__(self,**kw):self.backbone_latencies=[];super().__init__(**kw)
 def _prepare_control(self):
  start=time.perf_counter();out=super()._prepare_control();self.backbone_latencies.append(time.perf_counter()-start);return out

def store_snapshot(e,path):
 fields=['steps','previous_action','base_action','origin','backbone','frame_history','no_contact_s','push_W','last_reward_terms']
 value=dict(native=snapshot(e),fields={k:copy.deepcopy(getattr(e,k)) for k in fields},time=float(e.data.time))
 with path.open('xb') as f:pickle.dump(value,f)
def restore_snapshot(path):
 with Path(path).open('rb') as f:s=pickle.load(f)
 e=TrackEnv(controller='V2_FFv1',payload='liquid');e.reset(seed=int(Path(path).parent.name))
 for k,v in s['fields'].items():setattr(e,k,v)
 mujoco.mj_setState(e.model,e.data,s['native'],SPEC);mujoco.mj_forward(e.model,e.data);return e

def rollout(seed,folder,duration=30,kind='figure8',A=.5,B=.25,amplitude=0,terrain_seed=None,controller=None,capture=False,wall=120,refined=True,quadrature='simpson',flat_hfield=False):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);e=TimedEnv(controller='V2_FFv1',payload='liquid',kind=kind,A=A,B=B,T=duration,duration=duration,amplitude=amplitude,terrain_seed=terrain_seed or seed)
 if flat_hfield:
  assert amplitude>0;hid=e.model.hfield('track8_collision').id;adr=int(e.model.hfield_adr[hid]);size=int(e.model.hfield_nrow[hid]*e.model.hfield_ncol[hid]);e.model.hfield_data[adr:adr+size]=.5;e.heights[:]=0;e.terrain_info=dict(e.terrain_info,actual_min_m=0.,actual_max_m=0.,RMS_slope=0.,max_slope=0.,diagnostic='FLAT_HFIELD_ABLATION_NOT_ROUGH_TRIAL')
 o,initial=e.reset(seed=seed);e.backbone_latencies=[]
 account=(RefinedAccounting(e) if quadrature=='simpson' else Accounting(e)) if refined else None;x=[];qp=[];qv=[];wrench=[];positive=[];actions=[];latencies=[];contacts=[];warnings=e.data.warning.number.copy();action=np.zeros(3);vprev=kinematics(e)['v'][:2].copy();start=time.monotonic();seen=[];positive_work=0.
 if account:e.pre_sample_hook=account.before
 def collect(env):
  nonlocal vprev,positive_work
  if account:
   account.after(env);rr=account.rows[-1];endpin=max(0,float(rr[3:8].sum()));positive_work+=(.002/6*(max(0,account.pre['Pin'])+4*max(0,account.midpoint['Pin'])+endpin)) if quadrature=='simpson' else .001*(max(0,account.pre['Pin'])+endpin)
  st=kinematics(env);p,v,a=env.target();q,qd=env.fluid_state();acc=(st['v'][:2]-vprev)/.002;vprev=st['v'][:2].copy();x.append(np.r_[env.data.time,st['p'][:2],st['v'][:2],p,v,a,st['rpy'],q,qd,env.slosh_energy(),env.data.actuator_force[env.motor_ids],env.data.qvel[env.wheel_dofs],env.base_action,action,env.backbone.tilt,acc]);qp.append(env.data.qpos.copy());qv.append(env.data.qvel.copy());positive.append(positive_work)
  if capture or env.steps%5==0:wrench.append(np.r_[env.data.time,env.diagnostics()['wrench']])
  if env.steps%5==0:
   cs,ss=contact_sample(env);ground=[c for c in cs if c['pair']=='ground'];height,normal=surface(env.heights,*env.data.geom_xpos[env.ball_geom,:2]);_,bodycontact,_=env.contacts();contacts.append(dict(t=float(env.data.time),ground_contact=bool(ss['ground_contact_count']),body_ground_contact=bodycontact,normal_force_N=sum(c['normal_force_N'] for c in ground),ground_force_W_N=ss['ground_force_W_N'],slip_m_s=max((c['tangential_slip_m_s'] for c in ground),default=0),height_m=height,normal=normal.tolist(),local_slope=float(np.linalg.norm(normal[:2])/normal[2]),base_origin_clearance_proxy_m=float(env.data.xpos[env.base,2]-surface(env.heights,*env.data.xpos[env.base,:2])[0]),contact_points_W_m=[[c['pos_x_m'],c['pos_y_m'],c['pos_z_m']] for c in ground],mapping_gap=ss['contact_generalized_force_max_abs_gap_native_units'],power_gap_W=ss['contact_power_gap_W'],solver_gradient=ss['solver_normalized_gradient']))
 e.sample_hook=collect
 try:
  while e.steps<e.max_steps:
   tick=time.perf_counter();action=controller.action(e) if controller else np.zeros(3);latencies.append(time.perf_counter()-tick);actions.append(np.r_[e.data.time,action]);o,_,t,u,info=e.step(action)
   if capture:
    for stamp in [5,10,14]:
     if stamp not in seen and abs(e.data.time-stamp)<1e-8:store_snapshot(e,folder/f'snapshot_{stamp}.pkl');seen.append(stamp)
   if t or u:break
   if time.monotonic()-start>wall:info['termination_reason']='WALL_BUDGET';break
  arr=np.array(x);r=metrics(arr,duration,kind,A,B,duration,info['termination_reason'],'liquid');r.update(experiment=ID,controller=controller.name if controller else 'V2_FFv1',seed=seed,payload='liquid',kind=kind,A=A,B=B,T=duration,duration=duration,amplitude=amplitude,terrain_seed=terrain_seed or seed,initial_qpos=initial['initial_qpos'].tolist(),initial_qvel=initial['initial_qvel'].tolist(),model_hash=e.model_hash,numerical_warnings=int((e.data.warning.number-warnings).sum()),native_steps=len(arr),wall_s=time.monotonic()-start,residual_latency_mean_ms=float(np.mean(latencies)*1000),residual_latency_p99_ms=float(np.percentile(latencies,99)*1000),residual_deadlines_missed=int(sum(v>.01 for v in latencies)),privileged_fluid_feedback=controller is not None,snapshots=seen)
  r.update(backbone_latency_mean_ms=float(np.mean(e.backbone_latencies)*1000),backbone_latency_p99_ms=float(np.percentile(e.backbone_latencies,99)*1000),backbone_deadlines_missed=int(sum(v>.002 for v in e.backbone_latencies)),privileged_fluid_feedback=bool(getattr(controller,'privileged',True)) if controller else False,quadrature=quadrature,terrain_info=e.terrain_info,flat_hfield_ablation=flat_hfield,compensation_clip_fraction=float(controller.clipped/controller.calls) if controller and hasattr(controller,'clipped') and controller.calls else None)
  if contacts:r.update(ground_contact_fraction=float(np.mean([c['ground_contact'] for c in contacts])),body_ground_contacts=sum(c['body_ground_contact'] for c in contacts),slip_RMS_m_s=float(np.sqrt(np.mean([c['slip_m_s']**2 for c in contacts]))),normal_force_peak_N=max(c['normal_force_N'] for c in contacts),clearance_proxy_min_m=min(c['base_origin_clearance_proxy_m'] for c in contacts),contact_mapping_gap_max=max(c['mapping_gap'] for c in contacts))
  if r['numerical_warnings']:r['completion']=False;r['safety']=False
  if account:
   r['energy_accounting']=account.summary();n=np.searchsorted(arr[:,0],r['ROM_valid_duration_s']+1e-9,side='right');r['positive_generalized_work_J']=float(positive[n-1]) if n else None
  np.savez_compressed(folder/'trace.npz',rows=arr,schema=np.array(SCHEMA),qpos=np.array(qp),qvel=np.array(qv),native_energy=np.array(account.rows) if account else np.empty((0,26)),energy_schema=np.array(ENERGY_SCHEMA),positive_work=np.array(positive),wrench=np.array(wrench),policy=np.array(actions))
  __import__('shutil').copyfile(e.xml_path,folder/'model.xml');np.save(folder/'heightfield.npy',e.model.hfield_data);r.update(trace_path=str(folder/'trace.npz'),trace_sha256=sha(folder/'trace.npz'),xml_path=str(folder/'model.xml'))
  with (folder/'contacts.json').open('x') as f:json.dump(contacts,f,allow_nan=False)
  with (folder/'result.json').open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
  print('TRACK8C_NATIVE',r['controller'],seed,kind,amplitude,r['actual_s'],r['XY_RMSE_m'],r['q_observed_peak_m'],r['completion'],r['reason'],flush=True);return r
 finally:e.close()
