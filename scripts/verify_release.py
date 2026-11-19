"""Read-only research-publication audit: no training or native integration."""
from pathlib import Path
import hashlib,json,re,csv,sys,runpy,gzip
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def included(p):
 if not p.is_file():return False
 r=p.relative_to(ROOT)
 return r.parts[0] not in ['.git','.cache','runs'] and not any(v=='__pycache__' or v.endswith('.egg-info') for v in r.parts) and p.name not in ['artifact_manifest.json','MUJOCO_LOG.TXT']
def verify():
 manifest=json.loads((ROOT/'docs/artifact_manifest.json').read_text())['files']
 actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if included(p)}
 assert actual==set(manifest),(actual-set(manifest),set(manifest)-actual)
 for n,h in manifest.items():assert sha(ROOT/n)==h,n
 old=json.loads((ROOT/'docs/publication_history/v2/docs/artifact_manifest.json').read_text())['files']
 changed=['README.md','.gitignore','docs/ALGORITHMS.md','docs/REPRODUCIBILITY.md','scripts/verify_release.py']
 for n,h in old.items():
  p=ROOT/'docs/publication_history/v2'/n if n in changed else ROOT/n
  assert sha(p)==h,n
 print('PASS publication manifest and 929 protected V2 file hashes',flush=True)
 assert sha(ROOT/'pretrained/ppo_500k.zip')=='651989a98dd3ab04294adddd4a0bc0fb63e09b96749e9a980c5041e51bbd5db7'
 rows=list(csv.DictReader((ROOT/'research/terrain_heights/comparison.csv').open()));assert len(rows)==24
 for r in rows:
  folder=ROOT/f"research/terrain_heights/runs/{r['condition']}_{r['controller']}_{r['seed']}"
  z=np.load(folder/'trace.npz');x=z['rows'];assert x.shape==(15000,22) and np.isfinite(x).all()
  assert abs(x[-1,0]-30)<1e-8
  rmse=np.sqrt(np.mean(np.sum((x[:,1:3]-x[:,5:7])**2,axis=1)))
  assert abs(rmse-float(r['RMSE_m']))<1e-12
  assert abs(x[:,18].mean()-float(r['E_mean_J']))<1e-12
  assert abs(np.sqrt(np.mean(x[:,19:22]**2))-float(r['torque_RMS_Nm']))<1e-12
  assert r['full_task']=='False' and r['completed30']=='True' and r['physical_valid']=='True'
 for condition,v,p in [('H05',71.80,71.77),('H10',81.35,81.32),('H20',113.87,113.85),('MIXED',100.07,100.03)]:
  for controller,value in [('V4',v),('PPO',p)]:
   a=[float(x['RMSE_m'])*1000 for x in rows if x['condition']==condition and x['controller']==controller]
   assert len(a)==3 and round(np.mean(a),2)==value
 baseline=list(csv.DictReader((ROOT/'experiments/experiment_1_figure8/comparison.csv').open()));assert len(baseline)==18
 for r in baseline:assert r['fulltask']=='False'
 # Check each regenerated controller summary against preserved raw rows.
 for s in csv.DictReader((ROOT/'docs/controller_summary.csv').open()):
  a=[r for r in baseline if r['controller']==s['controller']];assert len(a)==int(s['n'])==3
  assert abs(float(s['RMSE_mm'])-1000*np.mean([float(x['RMSE_m']) for x in a]))<1e-10
 print('PASS 24 raw native traces, terrain means and 18 baseline/controller rows',flush=True)
 # New publication documents; historical snapshots retain their original path context.
 docs=[ROOT/'README.md']+[ROOT/'docs'/n for n in ['SYSTEM_MODEL.md','ALGORITHMS.md','REINFORCEMENT_LEARNING.md','EXPERIMENTS.md','RESULTS.md','REPRODUCIBILITY.md','TELEOPERATION.md']]
 for f in docs:
  for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',f.read_text()):
   if '://' in target or target.startswith('#'):continue
   assert (f.parent/target.split('#')[0]).exists(),(f,target)
 keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',(ROOT/'paper/references.bib').read_text()));used=set()
 for f in (ROOT/'paper').rglob('*.tex'):
  text=f.read_text()
  for n in re.findall(r'\\input\{([^}]+)\}',text):assert (ROOT/'paper'/(n+'.tex')).exists(),n
  for n in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):assert (ROOT/'paper'/n).exists(),n
  for ns in re.findall(r'\\cite\w*\{([^}]+)\}',text):used.update(n.strip() for n in ns.split(','))
 assert used<=keys
 assert json.loads((ROOT/'docs/publication_validation/receipt.json').read_text())['regression_tests']==41
 print('PASS new README/document links and manuscript references',flush=True)
 runpy.run_path(str(ROOT/'scripts/verify_upgrade.py'),run_name='__main__')
 print('PUBLICATION_VERIFY_PASS: no simulation or training',flush=True)
if __name__=='__main__':verify()
