from pathlib import Path
import sys,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from sloshbot.terrain_upgrade import LEVELS
from sloshbot.motion_replay import ContainerReplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=ROOT/'upgrade/figures';runs=[];summaries=[]
for p in sorted((ROOT/'upgrade/results').glob('*_261*/result.json')):
    r=json.loads(p.read_text());z=np.load(p.parent/'trace.npz');x=z['xyzv'];y=z['telemetry'];r['folder']=p.parent;r['xyz']=x;r['telemetry']=y;runs.append(r)
    summaries.append(dict(level=r['level'],seed=r['seed'],native_steps=r['native_steps'],actual_s=r['native_stop_s'],route_pass=r['traversal']['passed'],physical_valid=r['physical_valid'],termination=r['termination'],displacement_m=r['traversal']['horizontal_displacement_m'],distance3D_m=r['traversal']['cumulative_3D_distance_m'],elevation_range_m=r['traversal']['elevation_range_m'],wall_s=r['wall_s'],**{k:v for k,v in r['metrics'].items() if k!='recovery'}))
with (ROOT/'upgrade/results/traversal.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
for name,curves,unit in [('XY',None,'Y (m)'),('height',lambda x,y:x[:,3],'Base origin Z (m)'),('tilt',lambda x,y:np.linalg.norm(y[:,10:12],axis=1),'Tilt (rad)'),('modal_energy',lambda x,y:y[:,14],'Modal energy (J)'),('torque',lambda x,y:np.max(abs(y[:,19:22]),axis=1),'Peak motor torque (Nm)'),('velocity',lambda x,y:np.linalg.norm(y[:,7:9]-y[:,5:7],axis=1),'Velocity error (m/s)')]:
    fig,ax=plt.subplots(figsize=(7,4))
    for r in runs:
        if r['seed']!=2618001:continue
        x=r['xyz'];y=r['telemetry'];ax.plot(x[:,1] if curves is None else x[:,0],x[:,2] if curves is None else curves(x,y),label=r['level']+' '+r['termination'],lw=.9)
    ax.set(xlabel='X (m)' if curves is None else 'Time (s)',ylabel=unit,title='Actual first registered context; failures retained');ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=180)
    plt.close(fig)
# Actual container packet exported for a later, gated independent fluid replay.
folder=ROOT/'upgrade/results/MILD_2618001';z=np.load(folder/'trace.npz');motion=z['tank_motion'];replay=ContainerReplay(motion);node=replay.at(float(motion[len(motion)//2,0]));omega=np.array([replay.at(float(t))['omega_W_rad_s'] for t in motion[::10,0]]);gap=float(np.sqrt(np.mean((omega-motion[::10,13:16])**2)))
np.savez_compressed(ROOT/'research/high_fidelity/recorded_tank_motion.npz',motion=motion)
with (ROOT/'research/high_fidelity/recorded_tank_motion.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['t_s','px','py','pz']+[f'R{i}{j}' for i in range(3) for j in range(3)]+['omegaWx','omegaWy','omegaWz','vx','vy','vz']);w.writerows(motion)
with (ROOT/'docs/v2/container_motion_replay.json').open('x') as f:json.dump(dict(source='upgrade/results/MILD_2618001/trace.npz',samples=len(motion),start_s=float(motion[0,0]),end_s=float(motion[-1,0]),linear_velocity_point='WORLD tank body origin, not COM',position_interpolation='C1 CubicHermite, measured position and linear velocity',orientation_interpolation='C2 RotationSpline, body rates rotated into WORLD',spline_vs_measured_omega_RMS_rad_s=gap,used_in_independent_fluid=False,case6_status='BLOCKED_STATIC_GATE'),f,indent=2)
summary={level:dict(attempted=sum(r['level']==level for r in runs),passed=sum(r['level']==level and r['traversal']['passed'] for r in runs)) for level in LEVELS}
with (ROOT/'docs/v2/terrain_summary.json').open('x') as f:json.dump(dict(levels=summary,trials=summaries,new_native_route_steps=sum(r['native_steps'] for r in runs)),f,indent=2)
print('TERRAIN_ANALYSIS_COMPLETE',summary,flush=True)
