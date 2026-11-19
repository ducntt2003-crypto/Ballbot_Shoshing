"""Write-once native terrain evaluation, true XYZ and measured tank motion."""
from .terrain_upgrade import *
from .experiment import dump,settlement
from pathlib import Path
import argparse,time,json,csv
from ballbot_sloshing_research.track8_l import adapter as contact

def run(out,level,seed,duration=26,wall_s=70):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);env=TerrainEnv(level=level,duration=duration,terrain_seed=2617001,capture=True);env.reset(seed=seed);e=env.e;events=json.loads((Path(__file__).resolve().parents[1]/'configs/terrain_traverse.json').read_text())['events'];idx=0;clock=time.monotonic();reason='NOT_STARTED';error=None
    text=e.xml_path.read_text().replace(str(Path(__file__).resolve().parents[1]/'_vendor/ballbotgym/ballbotgym/assets'),'__ASSET_ROOT__');(out/'model.xml').write_text(text);np.save(out/'hfield.npy',e.model.hfield_data);np.save(out/'world_heights.npy',env.world_heights);np.savez_compressed(out/'initial.npz',qpos=e.data.qpos,qvel=e.data.qvel)
    try:
        for _ in range(round(duration/.01)):
            while idx<len(events) and events[idx]['t']<=float(e.data.time)+1e-9:env.command.request_keys(events[idx]['keys']);idx+=1
            _,_,term,trunc,info=env.step(np.zeros(3));reason=info['termination_reason']
            if term or trunc:break
            if time.monotonic()-clock>wall_s:reason='WALL_BUDGET';break
    except Exception as ex:reason='EXCEPTION';error=repr(ex)
    finally:
        states=np.asarray(env.route_states).reshape(-1,7);y=np.asarray(env.telemetry).reshape(-1,27);raw=np.asarray(env.states);stats=contact.stats();physical=not env.invalid_counts and not error and stats['unsupported']==0;complete=reason=='TIME_LIMIT' and env.meter_native_steps==round(duration/.002)
        np.savez_compressed(out/'trace.npz',rows=env.rows,native_rows=raw,telemetry=y,xyzv=states,tank_motion=env.motion);np.savez_compressed(out/'terminal.npz',t=e.data.time,qpos=e.data.qpos,qvel=e.data.qvel,ctrl=e.data.ctrl)
        r=dict(level=level,seed=seed,controller='Teleop_FF_Heading',physical_valid=physical,completed=complete,native_steps=env.meter_native_steps,native_stop_s=float(e.data.time),last_recorded_s=float(states[-1,0]) if len(states) else 0.,termination=reason,error=error,invalid_counts=env.invalid_counts.copy(),contact_stats=stats,wall_s=time.monotonic()-clock,traversal=traversal(states,level,physical,complete),height_range_m=[float(env.world_heights.min()),float(env.world_heights.max())],geometry_representation='normalized float32 world surface',total_mass_kg=float(e.model.body_mass.sum()))
        if len(y):r['metrics']=dict(roll_RMS_rad=float(np.sqrt(np.mean(y[:,10]**2))),pitch_RMS_rad=float(np.sqrt(np.mean(y[:,11]**2))),tilt_max_rad=float(np.linalg.norm(y[:,10:12],axis=1).max()),velocity_RMSE_m_s=float(np.sqrt(np.mean(np.sum((y[:,7:9]-y[:,5:7])**2,axis=1)))),q_observed_peak_m=float(np.linalg.norm(y[:,15:17],axis=1).max()),ROM_guard_triggered='ROM_INVALID' in env.invalid_counts,E_mean_J=float(y[:,14].mean()),E_peak_J=float(y[:,14].max()),torque_RMS_Nm=float(np.sqrt(np.mean(y[:,19:22]**2))),torque_peak_Nm=float(abs(y[:,19:22]).max()),ground_contact_fraction=float(y[:,22].mean()),recovery=settlement(y,[v['t'] for v in events if v['t']<duration]))
        env.close();dump(out/'result.json',r)
    print('TERRAIN',level,seed,r['native_stop_s'],reason,r['traversal']['passed'],flush=True);return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--level',choices=LEVELS,required=True);p.add_argument('--seed',type=int,default=2618001);p.add_argument('--duration',type=float,default=26);a=p.parse_args();run(a.output,a.level,a.seed,a.duration)
