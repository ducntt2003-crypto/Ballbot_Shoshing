"""Frozen-policy contact-aware evaluation, with explicit longer-horizon gates. No learning."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np,mujoco,torch
from stable_baselines3 import PPO
from ballbot_fast_rl.env import BallBotFastBalanceEnv
from ballbot_fast_rl.contracts import verify
from ballbot_fast_rl.evaluate import aggregate
from chrono_fsi.phase2.controllers import ReferencePIDAdapter
from chrono_fsi.integration.geometry import make_model
from .diagnostics import contact_sample

FAST=Path(__file__).resolve().parents[1]
CK=FAST/'results/20261029_stage1/train_stage2/best_model.zip'

class StudyEnv(BallBotFastBalanceEnv):
    def __init__(self,config):
        cfg=dict(config);params=cfg.pop('robot_params',None);super().__init__(cfg)
        if params:
            path=Path(self.tmp.name)/'robustness_model.xml';m,g=make_model(path,self.mode,params,self.physics_dt)
            assert np.array_equal(m.actuator_trnid,self.model.actuator_trnid)
            assert np.array_equal(m.jnt_qposadr,self.model.jnt_qposadr) and np.array_equal(m.jnt_dofadr,self.model.jnt_dofadr)
            self.model=m;self.geometry=g;self.data=mujoco.MjData(m);self.reference_mass=m.body_mass.copy();self.reference_inertia=m.body_inertia.copy();self.model_hash=hashlib.sha256(path.read_bytes()).hexdigest()


def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))

def contact_metrics(rows,states,duration):
    out=dict(max_force_mapping_gap_native_units=max(s['contact_generalized_force_max_abs_gap_native_units'] for s in states),max_contact_power_gap_W=max(abs(s['contact_power_gap_W']) for s in states),max_solver_normalized_gradient=max(s['solver_normalized_gradient'] for s in states),max_solver_iterations=max(s['solver_iterations'] for s in states))
    t=np.array([s['time_s'] for s in states]);steady=t>=min(1.,duration/2)
    out['three_wheel_contact_fraction_after_startup']=float(np.mean([s['wheel_contact_count']==3 for s in np.array(states,dtype=object)[steady]]))
    out['ground_contact_fraction_after_startup']=float(np.mean([s['ground_contact_count']>=1 for s in np.array(states,dtype=object)[steady]]))
    for pair in ['wheel_0','wheel_1','wheel_2','ground']:
        rr=[r for r in rows if r['pair']==pair];ss=[r for r in rr if r['time_s']>=min(1.,duration/2)]
        if not ss:ss=rr
        if not rr:continue
        forces={round(r['time_s'],9):r['normal_force_N'] for r in rr};active=np.array([float(round(s['time_s'],9) in forces) for s in states]);fn=np.array([forces.get(round(s['time_s'],9),0.) for s in states])
        metrics=dict(normal_force_mean_N=float(np.mean([r['normal_force_N'] for r in ss])),normal_force_RMS_N=rms([r['normal_force_N'] for r in ss]),normal_force_peak_N=float(max(r['normal_force_N'] for r in rr)),tangential_force_RMS_N=rms([r['tangential_force_N'] for r in ss]),low_mu_slip_RMS_m_s=rms([r['axial_low_mu_slip_m_s'] for r in ss]),high_mu_slip_RMS_m_s=rms([r['rolling_high_mu_slip_m_s'] for r in ss]),max_friction_ellipse_utilization=float(max(r['friction_ellipse_utilization'] for r in rr)),mean_friction_power_W=float(np.mean([r['friction_power_W'] for r in ss])),mean_gap_m=float(np.mean([r['gap_m'] for r in ss])),min_gap_m=float(min(r['gap_m'] for r in rr)),sampled_contact_duration_s=float(np.trapz(active,t)),normal_impulse_trapezoid_N_s=float(np.trapz(fn,t)))
        for k,v in metrics.items():out[pair+'_'+k]=v
    return out


def episode(model,controller,seed,cfg,contacts_writer=None,states_writer=None):
    e=StudyEnv(cfg);obs,info=e.reset(seed=seed);pid=ReferencePIDAdapter(e.dt,controller=='pid_bumpless');initial=e.state();yaw0=initial['roll_pitch_yaw'][2];samples=[];contacts=[];contact_states=[];lat=[];start=time.monotonic()
    cs,ss=contact_sample(e);contacts+=cs;contact_states.append(ss)
    for k in range(e.max_steps):
        tick=time.perf_counter()
        action=pid.act(e,{'orientation':e.state()['rotvec']}) if controller in ['pid_original','pid_bumpless'] else model.predict(obs,deterministic=True)[0]
        lat.append(time.perf_counter()-tick);obs,reward,term,trunc,info=e.step(action)
        s=info['state'];samples.append(s)
        # Full-rate representative trajectories; other episodes use registered 100Hz diagnostics.
        if k%5==4 or contacts_writer is not None or term or trunc:
            cs,ss=contact_sample(e);contacts+=cs;contact_states.append(ss)
            if contacts_writer is not None:
                for row in cs:contacts_writer.writerow(dict(seed=seed,controller=controller,**row))
            if states_writer is not None:
                states_writer.writerow(dict(seed=seed,controller=controller,time_s=s['time_s'],roll=s['roll'],pitch=s['pitch'],yaw=s['yaw'],tilt=s['tilt'],displacement_m=float(np.linalg.norm(s['position_error'][:2])),torque0_Nm=s['torque'][0],torque1_Nm=s['torque'][1],torque2_Nm=s['torque'][2],wheel_contacts=ss['wheel_contact_count'],ground_contacts=ss['ground_contact_count'],solver_iterations=ss['solver_iterations'],solver_gradient=ss['solver_normalized_gradient']))
        if term or trunc:break
    roll=np.array([s['roll'] for s in samples]);pitch=np.array([s['pitch'] for s in samples]);yaw=np.unwrap([s['yaw'] for s in samples])-yaw0;torque=np.array([s['torque'] for s in samples]);pos=np.array([s['position_error'][:2] for s in samples]);omega=np.array([s['omega'] for s in samples])
    result=dict(seed=seed,controller=controller,success=bool(info['success']),survival_s=float(e.data.time),reason=info['termination_reason'],roll_RMS_deg=float(np.rad2deg(rms(roll))),pitch_RMS_deg=float(np.rad2deg(rms(pitch))),max_tilt_deg=float(np.rad2deg(max(s['tilt'] for s in samples))),omega_RMS_rad_s=float(np.sqrt(np.mean(np.sum(omega*omega,axis=1)))),final_drift_m=float(np.linalg.norm(pos[-1])),max_displacement_m=float(np.linalg.norm(pos,axis=1).max()),max_abs_yaw_drift_deg=float(np.rad2deg(abs(yaw).max())),final_yaw_drift_deg=float(np.rad2deg(yaw[-1])),torque_RMS_Nm=rms(torque),torque_peak_Nm=float(abs(torque).max()),saturation_fraction=float(np.mean(np.any(abs(torque)>=.99*e.torque_scale,axis=1))),torque_total_variation_Nm=float(np.sum(abs(np.diff(torque,axis=0)))),control_energy_proxy_Nm2_s=float(e.dt*np.sum(torque*torque)),inference_p99_ms=float(np.percentile(lat,99)*1000),wall_s=time.monotonic()-start,model_hash=e.model_hash,mass_kg=float(e.model.body_mass.sum()))
    result['strict_gate_pass']=bool(result['success'] and result['roll_RMS_deg']<5 and result['pitch_RMS_deg']<5 and result['saturation_fraction']<=.05 and result['max_displacement_m']<=.3 and result['max_abs_yaw_drift_deg']<=10)
    result.update(contact_metrics(contacts,contact_states,result['survival_s']))
    if cfg.get('push_N',0)>0:
        end=cfg.get('push_start_s',2.)+cfg.get('push_duration_s',.1);hold=0.;recovery=None
        for s in samples:
            if s['time_s']<end:continue
            good=s['tilt']<np.deg2rad(5) and np.linalg.norm(s['omega'])<.1 and np.linalg.norm(s['velocity'][:2])<.1
            hold=hold+e.dt if good else 0.
            if hold>=.5 and recovery is None:recovery=s['time_s']-end
        result.update(recovery_success=recovery is not None,recovery_time_s=recovery)
    e.close();return result


def summary(rows):
    out=aggregate(rows);out['strict_gate_success_rate']=float(np.mean([r['strict_gate_pass'] for r in rows]));out['yaw_drift_max_deg']=float(max(r['max_abs_yaw_drift_deg'] for r in rows))
    for key in rows[0]:
        if key.startswith(('wheel_','ground_','max_force_mapping','max_contact_power','three_wheel_contact')) and isinstance(rows[0][key],(float,int)):
            values=[r[key] for r in rows];out[key]=dict(mean=float(np.mean(values)),max=float(max(values)))
    if 'recovery_success' in rows[0]:out['recovery_success_rate']=float(np.mean([r['recovery_success'] for r in rows]))
    return out


def run_batch(model,out,cfg,seeds,controllers):
    out.mkdir(exist_ok=False);allrows=[];summaries={}
    for controller in controllers:
        rows=[];contacts_file=out/(controller+'_contacts.csv');states_file=out/(controller+'_states.csv')
        with contacts_file.open('x',newline='') as cf,states_file.open('x',newline='') as sf:
            # Use actual first-contact schema to avoid invented API fields.
            e=StudyEnv(cfg);e.reset(seed=seeds[0]);cs,_=contact_sample(e);e.close()
            cw=csv.DictWriter(cf,['seed','controller']+list(cs[0]));cw.writeheader();sw=csv.DictWriter(sf,['seed','controller','time_s','roll','pitch','yaw','tilt','displacement_m','torque0_Nm','torque1_Nm','torque2_Nm','wheel_contacts','ground_contacts','solver_iterations','solver_gradient']);sw.writeheader()
            for i,seed in enumerate(seeds):
                rows.append(episode(model,controller,seed,cfg,cw if i==0 else None,sw if i==0 else None))
                if (i+1)%10==0:print(out.name,controller,i+1,flush=True)
        allrows+=rows;summaries[controller]=summary(rows)
        (out/(controller+'_episodes.json')).write_text(json.dumps(rows,indent=2));print('SUMMARY',out.name,controller,summaries[controller]['success_rate'],summaries[controller]['strict_gate_success_rate'],summaries[controller]['max_displacement_m']['worst'],flush=True)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    with (out/'episodes.csv').open('x',newline='') as f:
        keys=list(dict.fromkeys(k for r in allrows for k in r));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(allrows)
    return summaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['nominal','robustness'],required=True);a=p.parse_args();torch.set_num_threads(1);c=json.loads((FAST/'configs/stage2_long_balance.json').read_text());verify(CK,c);model=PPO.load(CK,device='cpu');controllers=['pid_original','pid_bumpless','ppo']
    if a.mode=='nominal':
        a.output.mkdir(exist_ok=False)
        first=run_batch(model,a.output/'horizon_5s',dict(c['env'],duration_s=5.),list(range(96000,96050)),controllers)
        second=run_batch(model,a.output/'horizon_10s',dict(c['env'],duration_s=10.),list(range(97000,97020)),controllers)
        passed=second['ppo']['strict_gate_success_rate']>=.9
        (a.output/'longer_horizon_gate.json').write_text(json.dumps(dict(gate10_pass=passed,workspace_hard_m=.75,displacement_goal_m=.3,test20_status='AUTHORIZED_BY_GATE' if passed else 'NOT RUN: 10s strict PPO gate failed',training='NOT RUN: unchanged contact model'),indent=2))
        if passed:run_batch(model,a.output/'horizon_20s',dict(c['env'],duration_s=20.),list(range(99000,99020)),controllers)
    else:
        a.output.mkdir(exist_ok=False)
        cases=[('empty',dict(mode='empty')),('rigid',dict(mode='rigid')),('mass_plus2pct',dict(mode='original',robot_params={'robot_mass_scale':1.02})),('COM_plus2mm',dict(mode='original',robot_params={'ballast_offset_m':[.002,0,0]})),('push_05N',dict(mode='original',push_N=.5,push_start_s=2.,push_duration_s=.1))]
        for name,extra in cases:run_batch(model,a.output/name,dict(c['env'],**extra,duration_s=5.),list(range(98000,98020)),controllers)
if __name__=='__main__':main()
