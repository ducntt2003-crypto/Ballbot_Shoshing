from pathlib import Path
import json,csv,sys,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text());sys.path.insert(0,str(R));from experiment import terrain
from ballbot_sloshing_research.track8.terrain import sample
rows=[];reps={}
for f in sorted((R/'runs').glob('*_261*/result.json')):
 a=json.loads(f.read_text());z=np.load(f.parent/'trace.npz');x=z['rows'];s=z['xyzv_ball'];H=np.load(f.parent/'world_heights.npy');h=np.array([sample(H,*xy)[0] for xy in s[:,7:9]]);m=a['metrics'];flat=(s[:,0]>1)&(h<.0001);base=float(np.median(s[flat,9])) if np.any(flat) else float(s[0,9]);crest=h>=.95*h.max();lift=float(np.median(s[crest,9])-base);row=dict(condition=a['condition'],controller=a['controller'],seed=a['seed'],planned_s=a['planned_s'],native_stop_s=a['native_stop_s'],completed30=a['completed'],physical_valid=a['physical_valid'],full_task=m['completion'],windows=m['ordered_nodes_passed'],RMSE_m=m['XY_RMSE_m'],max_tracking_error_m=m['max_XY_error_m'],tilt_max_rad=np.radians(m['tilt_max_deg']),q_observed_peak_m=m['q_observed_peak_m'],E_mean_J=m['E_mean_J'],torque_RMS_Nm=m['torque_RMS_Nm'],torque_peak_Nm=m['torque_peak_Nm'],saturation_fraction=m['saturation_fraction'],terrain_max_m=float(H.max()),max_terrain_encountered_m=float(h.max()),crest_ball_lift_after_settlement_m=lift,XY_excursion_m=float(np.linalg.norm(s[:,1:3]-s[0,1:3],axis=1).max()),cumulative_distance_m=float(np.linalg.norm(np.diff(s[:,1:4],axis=0),axis=1).sum()),termination=a['termination'],wall_s=a['wall_s'])
 rows.append(row)
 if a['seed']==P['seeds'][0]:reps[(a['condition'],a['controller'])]=(x,s,h,f.parent)
with (R/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for c in P['conditions']:
 summary[c]={}
 for method in P['controllers']:
  v=[a for a in rows if a['condition']==c and a['controller']==method];full=[a for a in v if a['completed30'] and a['physical_valid']]
  summary[c][method]=dict(attempted=len(v),physical30s=sum(a['completed30'] and a['physical_valid'] for a in v),fulltask=sum(a['full_task'] for a in v),mean_full30_RMSE_m=float(np.mean([a['RMSE_m'] for a in full])) if full else None,mean_full30_E_J=float(np.mean([a['E_mean_J'] for a in full])) if full else None,windows=[a['windows'] for a in v],max_tilt_rad=max([a['tilt_max_rad'] for a in v],default=0),actual_height_encountered_m=max([a['max_terrain_encountered_m'] for a in v],default=0),ball_lift_max_m=max([a['crest_ball_lift_after_settlement_m'] for a in v],default=0))
with (R/'analysis.json').open('x') as f:json.dump(dict(summary=summary,trials=rows),f,indent=2)
colors={'H05':'#2b94ad','H10':'#dd9633','H20':'#c64959','MIXED':'#7355b7'}
def save(fig,name):
 fig.tight_layout()
 for e in ['png','pdf','svg']:fig.savefig(R/'figures'/(name+'.'+e),dpi=180,bbox_inches='tight')
 plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for method,style in [('V4','o-'),('PPO','s--')]:
 xs=[];ys=[];es=[]
 for c in ['H05','H10','H20']:
  a=summary[c][method]
  if a['mean_full30_RMSE_m'] is not None:xs.append(P['height_m'][c]*1000);ys.append(a['mean_full30_RMSE_m']*1000);es.append(a['mean_full30_E_J']*1e6)
 axes[0].plot(xs,ys,style,label=method);axes[1].plot(xs,es,style,label=method)
for ax in axes:ax.set_xlabel('Nominal bump height (mm), same footprint');ax.legend();ax.grid(alpha=.25)
axes[0].set_ylabel('Mean full30s XY RMSE (mm)');axes[1].set_ylabel('Mean full30s modal energy (microjoules)');fig.suptitle('Actual height-scale comparison,3paired contexts; MIXED is a separate geometry');save(fig,'03_performance_by_height')
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
for ax,c in zip(axes.ravel(),P['conditions']):
 for method in P['controllers']:
  if (c,method) in reps:
   x,s,h,_=reps[c,method];ax.plot(s[:,0],h*1000,label=method)
 ax.set(title=c,xlabel='Native time (s)',ylabel='Actual terrain under sphere (mm)');ax.legend();ax.grid(alpha=.25)
fig.suptitle('Terrain actually encountered by the native sphere; not only a map outside its path');save(fig,'04_actual_height_encountered')
for name,col,unit,limit in [('05_tracking_error',None,'XY tracking error (m)',.25),('06_tilt',slice(11,13),'Tilt norm (rad)',.12),('07_sloshing',slice(14,16),'Modal displacement norm (mm)',8.4)]:
 fig,ax=plt.subplots(figsize=(9,4))
 for c in P['conditions']:
  if (c,'V4') not in reps:continue
  x,s,h,_=reps[c,'V4'];y=np.linalg.norm(x[:,1:3]-x[:,5:7],axis=1) if col is None else np.linalg.norm(x[:,col],axis=1)*(1000 if name=='07_sloshing' else 1);ax.plot(x[:,0],y,label=c,color=colors[c])
 ax.axhline(limit,c='k',ls=':',label='Original guard');ax.set(xlabel='Native time (s)',ylabel=unit,title='V4 on different real terrain heights, fixed first seed');ax.legend();ax.grid(alpha=.25);save(fig,name)
fig,axes=plt.subplots(1,2,figsize=(11,4));positions=np.arange(4);width=.35
for j,method in enumerate(P['controllers']):
 axes[0].bar(positions+(j-.5)*width,[summary[c][method]['physical30s'] for c in P['conditions']],width,label=method);axes[1].bar(positions+(j-.5)*width,[summary[c][method]['fulltask'] for c in P['conditions']],width,label=method)
for ax in axes:ax.set_xticks(positions,P['conditions']);ax.set_ylim(0,3.5);ax.legend();ax.grid(axis='y',alpha=.25)
axes[0].set_title('Physically complete30s /3');axes[1].set_title('Original full-task passes /3');fig.suptitle('Traversing a raised surface is distinct from meeting every tracking gate');save(fig,'08_completion')
print('ANALYSIS_COMPLETE',summary,flush=True)
