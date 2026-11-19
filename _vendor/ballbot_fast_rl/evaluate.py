"""Matched-seed rigid-body evaluation; no training or SPH."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from .env import BallBotFastBalanceEnv
from .contracts import verify
from chrono_fsi.phase2.controllers import ReferencePIDAdapter

def episode(policy,seed,config,controller='ppo',trace_path=None):
 e=BallBotFastBalanceEnv(config);obs,info=e.reset(seed=int(seed));pid=ReferencePIDAdapter(e.dt,controller=='pid_bumpless');trace=[];lat=[]
 for i in range(e.max_steps):
  tick=time.perf_counter()
  if controller=='zero':a=np.zeros(3)
  elif controller in ['pid_original','pid_bumpless']:a=pid.act(e,{'orientation':e.state()['rotvec']})
  else:a=policy.predict(obs,deterministic=True)[0]
  lat.append(time.perf_counter()-tick);obs,r,term,trunc,info=e.step(a);s=info['state']
  trace.append(dict(s,action=np.asarray(a).tolist(),reward_terms=info['reward_terms']))
  if term or trunc:break
 t=np.array([s['time_s'] for s in trace]);roll=np.array([s['roll'] for s in trace]);pitch=np.array([s['pitch'] for s in trace]);omega=np.array([s['omega'] for s in trace]);pos=np.array([s['position_error'][:2] for s in trace]);tau=np.array([s['torque'] for s in trace]);v=np.array([s['velocity'] for s in trace]);d=np.linalg.norm(pos,axis=1)
 metrics=dict(seed=int(seed),controller=controller,mass_scale=e.mass_scale,total_mass_kg=float(e.model.body_mass.sum()),push_W_N=e.push_W.tolist(),success=bool(info['success']),survival_s=float(t[-1]),reason=info['termination_reason'],roll_RMS_deg=float(np.rad2deg(np.sqrt(np.mean(roll**2)))),pitch_RMS_deg=float(np.rad2deg(np.sqrt(np.mean(pitch**2)))),max_tilt_deg=float(np.rad2deg(max(s['tilt'] for s in trace))),omega_RMS_rad_s=float(np.sqrt(np.mean(np.sum(omega**2,axis=1)))),final_drift_m=float(d[-1]),max_displacement_m=float(d.max()),torque_RMS_Nm=float(np.sqrt(np.mean(tau*tau))),torque_peak_Nm=float(abs(tau).max()),saturation_fraction=float(np.mean(np.any(abs(tau)>=.99*e.torque_scale,axis=1))),torque_total_variation_Nm=float(np.sum(abs(np.diff(tau,axis=0)))),control_energy_proxy_Nm2_s=float(e.dt*np.sum(tau*tau)),inference_p50_ms=float(np.percentile(lat,50)*1000),inference_p95_ms=float(np.percentile(lat,95)*1000),inference_p99_ms=float(np.percentile(lat,99)*1000),reward_total=float(sum(s['reward_terms']['reward_total'] for s in trace)),episode_terms=info['episode_terms'])
 if trace_path:Path(trace_path).write_text(json.dumps(dict(config=e.cfg,seed=int(seed),controller=controller,metrics=metrics,trace=trace)))
 e.close();return metrics

def aggregate(rows):
 result=dict(episodes=len(rows),success_rate=float(np.mean([r['success'] for r in rows])),fall_rate=float(np.mean([r['reason']=='FALL' for r in rows])))
 n=len(rows);p=result['success_rate'];z=1.96;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;result['success_Wilson_95CI']=[float(center-half),float(center+half)]
 for key in ['survival_s','roll_RMS_deg','pitch_RMS_deg','max_tilt_deg','final_drift_m','max_displacement_m','torque_RMS_Nm','torque_peak_Nm','saturation_fraction','omega_RMS_rad_s','control_energy_proxy_Nm2_s','torque_total_variation_Nm','inference_p99_ms']:
  values=np.array([r[key] for r in rows]);result[key]=dict(mean=float(values.mean()),median=float(np.median(values)),std=float(values.std()),worst=float(values.min() if key=='survival_s' else values.max()))
 return result

def ranking(summary):
 return (summary['success_rate'],summary['survival_s']['mean'],-(summary['roll_RMS_deg']['mean']+summary['pitch_RMS_deg']['mean']),-summary['max_displacement_m']['mean'],-summary['torque_RMS_Nm']['mean'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--episodes',type=int,default=50);p.add_argument('--seed-start',type=int,default=90000);p.add_argument('--initial-checkpoint',type=Path);p.add_argument('--imitation-checkpoint',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
 c=json.loads(a.config.read_text());verify(a.checkpoint,c);model=PPO.load(a.checkpoint,device='cpu');summaries={};allrows=[]
 # Test seeds are touched once, after checkpoint selection on separate validation seeds.
 controllers=['zero','pid_original','pid_bumpless','ppo']+(['ppo_initial'] if a.initial_checkpoint else [])+(['pid_imitation_only'] if a.imitation_checkpoint else [])
 for name in controllers:
  selected=model
  if name=='ppo_initial':verify(a.initial_checkpoint,c);selected=PPO.load(a.initial_checkpoint,device='cpu')
  if name=='pid_imitation_only':verify(a.imitation_checkpoint,c);selected=PPO.load(a.imitation_checkpoint,device='cpu')
  rows=[episode(selected,a.seed_start+i,dict(c['env'],duration_s=5.),name,a.output/f'{name}_seed_{a.seed_start}.json' if i==0 else None) for i in range(a.episodes)]
  summaries[name]=aggregate(rows);summaries[name]['nominal_gate_success_rate']=float(np.mean([r['success'] and r['roll_RMS_deg']<5 and r['pitch_RMS_deg']<5 and r['saturation_fraction']<=.05 and r['max_displacement_m']<=.30 for r in rows]));allrows+=rows;print(name,summaries[name]['success_rate'],summaries[name]['survival_s']['mean'],flush=True)
 (a.output/'episodes.json').write_text(json.dumps(allrows,indent=2));(a.output/'summary.json').write_text(json.dumps(summaries,indent=2))
 with (a.output/'episodes.csv').open('w',newline='') as f:
  fields=[k for k in allrows[0] if k!='episode_terms'];w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(allrows)
if __name__=='__main__':main()
