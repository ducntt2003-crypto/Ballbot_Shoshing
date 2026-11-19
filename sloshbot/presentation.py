"""Offline scientific plots and native state replay, never physics integration."""
from .bootstrap import ROOT
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,csv,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25})

def table(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)

def save(fig,folder,name):
    fig.tight_layout()
    for ext in ['png','pdf','svg']:
        p=folder/(name+'.'+ext);assert not p.exists();fig.savefig(p,dpi=180,bbox_inches='tight')
    plt.close(fig)

def zones(x):
    radius=np.linalg.norm(x[:,1:3],axis=1);out=['flat_initialization' if r<.25 else 'transition' if r<.5 else 'mild_rough' if y<0 else 'smooth_uneven' for r,y in zip(radius,x[:,2])]
    return sorted(set(out))

def main():
    f1=ROOT/'assets/plots/experiment_1';f2=ROOT/'assets/plots/experiment_2';d1=ROOT/'experiments/experiment_1_figure8';d2=ROOT/'experiments/experiment_2_teleoperation'
    runs=[]
    for p in sorted(d1.glob('*/result.json')):
        r=json.loads(p.read_text());r['folder']=p.parent;r['x']=np.load(p.parent/'trace.npz')['rows'];runs.append(r)
    rows=[]
    for r in runs:
        m=r['metrics'];rows.append(dict(controller=r['controller'],seed=r['seed'],privileged=r['privileged'],duration_s=r['actual_s'],native_stop_time_s=r.get('native_stop_time_s',r['actual_s']+(0.002 if not r['physical_valid'] else 0)),physical_valid=r['physical_valid'],fulltask=m['completion'],windows=m['ordered_nodes_passed'],RMSE_m=m['XY_RMSE_m'],max_error_m=m['max_XY_error_m'],roll_RMS_deg=m['roll_RMS_deg'],pitch_RMS_deg=m['pitch_RMS_deg'],tilt_max_deg=m['tilt_max_deg'],yaw_drift_deg=m['yaw_drift_max_deg'],E_mean_J=m['E_mean_J'],E_peak_J=m['E_peak_J'],q_peak_m=m['q_peak_m'],torque_RMS_Nm=m['torque_RMS_Nm'],torque_peak_Nm=m['torque_peak_Nm'],saturation_fraction=m['saturation_fraction'],termination=r['termination'],reused=r.get('reused',False),wall_s=r['wall_s'],metric_scope='observed valid prefix; no failed-prefix energy ranking'))
    table(d1/'comparison.csv',rows)
    reps=[r for r in runs if r['seed']==2616001]
    specs=[('trajectory',None,'X (m)','Y (m)'),('tracking_error',lambda x:np.linalg.norm(x[:,1:3]-x[:,5:7],axis=1),'Time (s)','XY error (m)'),('balance',lambda x:np.linalg.norm(x[:,11:13],axis=1),'Time (s)','Tilt norm (rad)'),('modal_energy',lambda x:x[:,18]*1e6,'Time (s)','Modal energy (microjoules)'),('torque',lambda x:np.max(abs(x[:,19:22]),axis=1),'Time (s)','Maximum motor torque (Nm)')]
    for name,fun,xlabel,ylabel in specs:
        fig,ax=plt.subplots(figsize=(7,4))
        if fun is None:
            x=reps[0]['x'];ax.plot(x[:,5],x[:,6],'k--',label='Desired analytical reference');ax.set_aspect('equal')
        for r in reps:
            x=r['x'];label=r['controller']+(' [privileged]' if r['privileged'] else '')+(' (terminated)' if not r['completed'] else '')
            ax.plot(x[:,0] if fun else x[:,1],fun(x) if fun else x[:,2],label=label,lw=1)
        ax.set(xlabel=xlabel,ylabel=ylabel,title='Experiment 1: fixed TEST_1; failed runs are truncated');ax.legend(fontsize=7);save(fig,f1,name)
    fig,ax=plt.subplots(figsize=(7,4));labels=sorted({r['controller'] for r in runs});ax.bar(labels,[sum(r['metrics']['completion'] for r in runs if r['controller']==l)/3 for l in labels]);ax.set(ylim=(0,1),ylabel='Full-task pass fraction',title='Original 16-window acceptance; three contexts per controller');save(fig,f1,'completion')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(labels,[np.mean([r['metrics']['ordered_nodes_passed'] for r in runs if r['controller']==l]) for l in labels]);ax.axhline(16,c='k',ls='--');ax.set(ylim=(0,17),ylabel='Mean passed windows',title='Ordered windows; full-duration survival is a separate metric');save(fig,f1,'windows')
    tel=[]
    for p in sorted(d2.glob('*/result.json')):
        r=json.loads(p.read_text());z=np.load(p.parent/'trace.npz');y=z['telemetry'];x=z['rows'];st=r['stability'];tel.append(dict(run=p.parent.name,controller=r['controller'],seed=r['seed'],duration_s=r['actual_s'],native_stop_s=r.get('native_stop_time_s',r['actual_s']+(.002 if not r['physical_valid'] else 0)),stability_pass=r['stability_pass'],physical_valid=r['physical_valid'],termination=r['termination'],ROM_guard_triggered='ROM_INVALID' in r['invalid_counts'],observed_q_peak_m=st['q_peak_m'],zones=','.join(zones(x)),max_radius_m=float(np.linalg.norm(x[:,1:3],axis=1).max()),**{k:v for k,v in st.items() if k!='recovery'}))
    table(d2/'stability.csv',tel)
    # Always select the preregistered first context, never the best-looking episode.
    chosen=d2/'modal_replay_2618001'
    if not (chosen/'result.json').exists():chosen=d2/'replay_2618001'
    y=np.load(chosen/'trace.npz')['telemetry'];r=json.loads((chosen/'result.json').read_text())
    panels=[('command_timeline',[(y[:,1],'Forward key'),(y[:,2],'Left key')],'Key direction'),('velocity',[(y[:,5],'Desired vx'),(y[:,7],'Actual vx'),(y[:,6],'Desired vy'),(y[:,8],'Actual vy')],'World velocity (m/s)'),('yaw_rate',[(y[:,4],'Desired'),(y[:,9],'Actual')],'Yaw rate (rad/s)'),('roll_pitch',[(y[:,10],'Roll'),(y[:,11],'Pitch')],'Angle (rad)'),('max_tilt',[(np.linalg.norm(y[:,10:12],axis=1),'Tilt norm')],'Tilt (rad)'),('modal_energy',[(y[:,14]*1e6,'Two-mode energy')],'Energy (microjoules)'),('motor_torque',[(y[:,19+i],f'Motor {i}') for i in range(3)],'Torque (Nm)')]
    for name,curves,unit in panels:
        fig,ax=plt.subplots(figsize=(7,3.5))
        for a,label in curves:ax.plot(y[:,0],a,label=label,lw=.8)
        ax.set(xlabel='Time (s)',ylabel=unit,title=f'Experiment 2: {chosen.name} | {r["termination"]}');ax.legend(fontsize=7);save(fig,f2,name)
    fig,ax=plt.subplots(figsize=(6,5));from .runtime import playground
    z=playground();im=ax.imshow(z,extent=[-2,2,-2,2],origin='lower',cmap='terrain');fig.colorbar(im,ax=ax,label='Height (m), no exaggeration of numeric scale')
    for name in ['replay_2618001','modal_replay_2618001','modal_traverse_2618001']:
        p=d2/name/'trace.npz'
        if p.exists():x=np.load(p)['rows'];ax.plot(x[:,1],x[:,2],label=name)
    ax.set(xlabel='X (m)',ylabel='Y (m)',title='Actual paths on the 4 x 4 m collision playground');ax.legend(fontsize=7);save(fig,f2,'terrain_path')
    rec=[]
    for p in d2.glob('*/result.json'):
        a=json.loads(p.read_text())
        for v in a['stability']['recovery']:rec.append(dict(run=p.parent.name,**v))
    table(d2/'recovery.csv',rec)
    fig,ax=plt.subplots(figsize=(7,3.5));rr=r['stability']['recovery'];ax.bar([v['change_s'] for v in rr],[v['recovery_s'] if v['recovery_s'] is not None else 0 for v in rr],width=.5)
    for v in rr:
        if v['censored']:ax.text(v['change_s'],.02,'censored',rotation=90,ha='center')
    ax.set(xlabel='Command change time (s)',ylabel='Settling time (s)',title='0.015 m/s + 0.04 rad/s + tilt<=0.06 rad, held 0.5 s');save(fig,f2,'recovery')
    summary={'experiment_1':{l:dict(trials=3,fulltask=sum(r['metrics']['completion'] for r in runs if r['controller']==l),full_duration=sum(r['completed'] for r in runs if r['controller']==l),windows=[r['metrics']['ordered_nodes_passed'] for r in runs if r['controller']==l]) for l in labels},'experiment_2':tel,'new_native_evaluation_steps':sum(r.get('native_steps',0) for r in runs)+sum(json.loads(p.read_text()).get('native_steps',0) for p in d2.glob('*/result.json')),'no_new_training':True,'scope':'No population CI/algorithm superiority from one policy lineage and three contexts.'}
    with (ROOT/'docs/research_summary.json').open('x') as f:json.dump(summary,f,indent=2)
    print('PLOTS_AND_TABLES_COMPLETE',flush=True)
if __name__=='__main__':main()
