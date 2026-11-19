"""Bounded write-once evaluation; commands are replayed closed-loop."""
from .runtime import *
from .bootstrap import ROOT
import argparse,csv,json,time
from pathlib import Path
from stable_baselines3 import PPO
from ballbot_sloshing_research.track8.evaluate import metrics
from ballbot_sloshing_research.track8_l import adapter as contact

def dump(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)

def settlement(x,changes,tolerance=.015,yaw_tolerance=.04,hold=.5):
    out=[]
    good=(np.linalg.norm(x[:,7:9]-x[:,5:7],axis=1)<=tolerance)&(abs(x[:,9]-x[:,4])<=yaw_tolerance)&(np.linalg.norm(x[:,10:12],axis=1)<=.06)
    for i,t in enumerate(changes):
        end=changes[i+1] if i+1<len(changes) else float(x[-1,0]);ids=np.flatnonzero((x[:,0]>=t)&(x[:,0]<end));n=round(hold/.002);settled=None
        for k in ids:
            if k+n<=len(x) and x[k+n-1,0]<end and np.all(good[k:k+n]):settled=float(x[k,0]-t);break
        out.append(dict(change_s=t,until_next_change_s=end,recovery_s=settled,censored=settled is None,tolerance_m_s=tolerance,yaw_tolerance_rad_s=yaw_tolerance,hold_s=hold))
    return out

def run(out,controller='V4',seed=2616001,terrain_seed=2617001,duration=30,script=None,terrain='playground',wall_s=90):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);teleop=script is not None;events=json.loads(Path(script).read_text()) if teleop else None
    env=TeleopEnv(terrain=terrain,terrain_seed=terrain_seed,duration=duration,capture=True) if teleop else FigureEnv(controller=controller,terrain_seed=terrain_seed,duration=duration,capture=True)
    clock=time.monotonic();o,_=env.reset(seed=seed);e=env.e;initial=dict(qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),origin=e.origin.tolist())
    text=e.xml_path.read_text().replace(str(ROOT/'_vendor/ballbotgym/ballbotgym/assets/stl_files'),'__ASSET_ROOT__/stl_files').replace(str(ROOT/'_vendor/ballbotgym/ballbotgym/assets/textures'),'__ASSET_ROOT__/textures')
    (out/'model.xml').write_text(text);np.save(out/'hfield.npy',e.model.hfield_data)
    model=PPO.load(ROOT/'pretrained/ppo_500k.zip',device='cpu') if controller=='PPO' else None
    reason='NOT_STARTED';error=None;lat=[];applied=[];idx=0
    try:
        for _ in range(round(duration/.01)):
            if teleop:
                while idx<len(events['events']) and events['events'][idx]['t']<=float(e.data.time)+1e-10:
                    evt=events['events'][idx];env.command.request_keys(evt['keys']);applied.append(dict(evt,actual_sim_t=float(e.data.time)));idx+=1
            tick=time.perf_counter();action=model.predict(o,deterministic=True)[0] if model else np.zeros(3);lat.append(time.perf_counter()-tick)
            o,_,term,trunc,info=env.step(action);reason=info['termination_reason']
            if term or trunc:break
            if time.monotonic()-clock>wall_s:reason='WALL_BUDGET';break
    except Exception as exc:reason='EXCEPTION';error=repr(exc)
    finally:
        x=np.array(env.rows).reshape(-1,22);native=np.array(env.states);stats=contact.stats();invalid=env.invalid_counts.copy();physical=not invalid and error is None and stats['unsupported']==0
        np.savez_compressed(out/'trace.npz',rows=x,native_rows=native,telemetry=np.array(env.telemetry) if teleop else np.empty((0,0)))
        native_steps=env.meter_native_steps;env.close()
    complete=reason=='TIME_LIMIT' and len(x)==round(duration/.002)
    result=dict(controller=controller if not teleop else 'Teleop_FF_Heading',seed=seed,terrain_seed=terrain_seed,planned_s=duration,actual_s=float(x[-1,0]) if len(x) else 0.,completed=complete,physical_valid=physical,termination=reason,error=error,native_steps=native_steps,wall_s=time.monotonic()-clock,initial=initial,total_mass_kg=float(e.model.body_mass.sum()),inference_mean_ms=float(np.mean(lat)*1000),contact_stats=stats,invalid_counts=invalid,terrain=terrain if teleop else 'benchmark',privileged=controller in ['V4','PPO'] and not teleop)
    if len(x):result['metrics']=metrics(x,duration,'figure8',.75,.375,30.,reason,'liquid')
    if teleop and len(env.telemetry):
        y=np.array(env.telemetry);q=np.linalg.norm(y[:,15:17],axis=1);tilt=np.linalg.norm(y[:,10:12],axis=1)
        result['stability']=dict(roll_RMS_rad=float(np.sqrt(np.mean(y[:,10]**2))),pitch_RMS_rad=float(np.sqrt(np.mean(y[:,11]**2))),max_tilt_rad=float(tilt.max()),velocity_RMSE_m_s=float(np.sqrt(np.mean(np.sum((y[:,7:9]-y[:,5:7])**2,axis=1)))),yawrate_RMSE_rad_s=float(np.sqrt(np.mean((y[:,9]-y[:,4])**2))),torque_peak_Nm=float(abs(y[:,19:22]).max()),torque_RMS_Nm=float(np.sqrt(np.mean(y[:,19:22]**2))),q_peak_m=float(q.max()),E_mean_J=float(y[:,14].mean()),E_peak_J=float(y[:,14].max()),ground_contact_fraction=float(y[:,22].mean()),body_ground_count=int(y[:,23].sum()),fall_count=int(tilt.max()>.12),ROM_crossing=bool(q.max()>.0084),recovery=settlement(y,[float(v['t']) for v in events['events'] if v['t']<duration]))
        st=result['stability'];result['stability_pass']=bool(complete and physical and st['max_tilt_rad']<=.12 and st['q_peak_m']<=.0084 and st['torque_peak_Nm']<=10+1e-10 and st['velocity_RMSE_m_s']<=.04 and st['yawrate_RMSE_rad_s']<=.08)
        with (out/'commands.jsonl').open('x') as f:
            for row in applied:f.write(json.dumps(row)+'\n')
        with (out/'telemetry.csv').open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['t','key_forward','key_left','desired_forward_m_s','desired_yawrate_rad_s','desired_vx','desired_vy','actual_vx','actual_vy','actual_yawrate','roll','pitch','yaw','desired_yaw','E_J','qx','qy','qdx','qdy','tau0','tau1','tau2','ground','body_ground','contact_gap','refx','refy']);w.writerows(y)
    dump(out/'result.json',result);print('RESULT',out.name,result['actual_s'],reason,'valid',physical,'pass',result.get('stability_pass',result.get('metrics',{}).get('completion')),flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--controller',default='V4',choices=['V2','V3','V2_FFv1','V3_FFv1','V4','PPO']);p.add_argument('--seed',type=int,default=2616001);p.add_argument('--terrain-seed',type=int,default=2617001);p.add_argument('--duration',type=float,default=30);p.add_argument('--script');p.add_argument('--terrain',choices=['flat','benchmark','playground'],default='playground');p.add_argument('--wall-s',type=float,default=90);a=p.parse_args();run(out=a.output,controller=a.controller,seed=a.seed,terrain_seed=a.terrain_seed,duration=a.duration,script=a.script,terrain=a.terrain,wall_s=a.wall_s)
if __name__=='__main__':main()
