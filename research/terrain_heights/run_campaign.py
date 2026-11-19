from pathlib import Path
import json,subprocess,sys,time
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text());sys.path.insert(0,str(R));from experiment import terrain
import numpy as np
start=time.monotonic();geometry={}
for c in P['conditions']:
 H=terrain(c);d=4/128;g1=np.hypot((H[:-1,1:]-H[:-1,:-1])/d,(H[1:,1:]-H[:-1,1:])/d);g2=np.hypot((H[1:,1:]-H[1:,:-1])/d,(H[1:,:-1]-H[:-1,:-1])/d);slope=float(np.arctan(max(g1.max(),g2.max())));geometry[c]=dict(max_height_m=float(H.max()),min_height_m=float(H.min()),max_slope_rad=slope,max_slope_deg=float(np.degrees(slope)),geometry_pass=slope<=.06,initial_origin_height_m=float(H[64,64]));np.save(R/'figures'/(c+'_heightfield.npy'),H)
with (R/'geometry.json').open('x') as f:json.dump(geometry,f,indent=2)
opened=[];blocked=[]
for condition in P['conditions']:
 if not geometry[condition]['geometry_pass']:blocked.append(dict(condition=condition,reason='GEOMETRY_SLOPE_GATE'));break
 jobs=[('V4',P['seeds'][0],3,'pilot')]+[(c,s,30,str(s)) for c in P['controllers'] for s in P['seeds']]
 for controller,seed,duration,suffix in jobs:
  if time.monotonic()-start>850:blocked.append(dict(condition=condition,reason='AGGREGATE_WALL'));break
  name=f'{condition}_{controller}_{suffix}';folder=R/'runs'/name;cmd=[sys.executable,'-B',str(R/'experiment.py'),condition,controller,str(seed),str(duration),str(folder)]
  with (R/'logs'/(name+'.log')).open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=105)
  assert p.returncode==0,(name,p.returncode)
  r=json.loads((folder/'result.json').read_text());print(name,r['native_stop_s'],r['termination'],flush=True)
  assert sum(json.loads(p.read_text())['native_steps'] for p in (R/'runs').glob('*/result.json'))<=400000
  if suffix=='pilot' and not(r['completed'] and r['physical_valid']):blocked.append(dict(condition=condition,reason='PILOT_PHYSICAL_GATE'));break
 if blocked:break
 opened.append(condition)
 counts={c:sum(json.loads(p.read_text())['completed'] and json.loads(p.read_text())['physical_valid'] for p in (R/'runs').glob(f'{condition}_{c}_261*/result.json')) for c in P['controllers']}
 if min(counts.values())<2:blocked.append(dict(condition=condition,reason='FEWER_THAN_TWO_FULL_PHYSICAL_ROUTES',counts=counts));break
with (R/'campaign_status.json').open('x') as f:json.dump(dict(opened_conditions=opened,blocked=blocked,remaining_not_run=[c for c in P['conditions'] if c not in opened],wall_s=time.monotonic()-start),f,indent=2)
print('DONE',opened,blocked,flush=True)
