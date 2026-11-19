from pathlib import Path
import sys,json,subprocess,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sloshbot.terrain_upgrade import heights,LEVELS
import numpy as np
p=json.loads((ROOT/'upgrade/configs/protocol.json').read_text());start=time.monotonic();geo={}
for level in LEVELS:
 z=heights(level);h=4/128;ax=(z[:-1,1:]-z[:-1,:-1])/h;ay=(z[1:,1:]-z[:-1,1:])/h;bx=(z[1:,1:]-z[1:,:-1])/h;by=(z[1:,:-1]-z[:-1,:-1])/h;sl=np.maximum(np.hypot(ax,ay),np.hypot(bx,by));slope=float(sl.max());geo[level]=dict(height_min_m=float(z.min()),height_max_m=float(z.max()),max_gradient=slope,max_slope_rad=float(np.arctan(slope)),grid_spacing_m=h,normal_z_min=float(1/np.sqrt(1+slope*slope)),geometry_pass=bool(np.arctan(slope)<=.03),height_sha256=hashlib.sha256(z.tobytes()).hexdigest())
with (ROOT/'upgrade/results/geometry.json').open('x') as f:json.dump(geo,f,indent=2)
for level in LEVELS:
 if not geo[level]['geometry_pass']:print('GEOMETRY_GATE_BLOCKED',level,geo[level],flush=True);continue
 for seed,dur,label in [(2618001,2,'pilot')]+[(s,26,str(s)) for s in p['seeds']]:
  out=ROOT/f'upgrade/results/{level}_{label}'
  if (out/'result.json').exists():continue
  if time.monotonic()-start>1000:raise RuntimeError('Aggregate native wall ceiling')
  cmd=[sys.executable,'-B','-m','sloshbot.upgrade_evaluate','--output',str(out.relative_to(ROOT)),'--level',level,'--seed',str(seed),'--duration',str(dur)]
  with (ROOT/f'upgrade/results/{level}_{label}.log').open('x') as f:q=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=85)
  if q.returncode:raise RuntimeError((level,label,q.returncode))
  r=json.loads((out/'result.json').read_text());print(level,label,r['native_stop_s'],r['termination'],r['traversal']['passed'],flush=True)
  assert sum(json.loads(n.read_text())['native_steps'] for n in (ROOT/'upgrade/results').glob('*/result.json'))<=300000
  if label=='pilot' and (not r['physical_valid'] or not r['completed']):break
 if level=='FLAT':
  count=sum(json.loads(n.read_text())['physical_valid'] and json.loads(n.read_text())['completed'] for n in (ROOT/'upgrade/results').glob('FLAT_261*/result.json'))
  if count<2:print('FLAT_ROUTE_GATE_BLOCKS_HARDER_LEVELS',flush=True);break
