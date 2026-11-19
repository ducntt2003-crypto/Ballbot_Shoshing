"""Read-only release verification: no training, physics steps, rendering or downloads."""
from pathlib import Path
import json,hashlib,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def included(p):
 return p.is_file() and not any(x in ['.cache','__pycache__'] or x.endswith('.egg-info') for x in p.relative_to(ROOT).parts) and p.name not in ['artifact_manifest.json','MUJOCO_LOG.TXT']
def verify():
 manifest=json.loads((ROOT/'docs/artifact_manifest.json').read_text());files=manifest['files']
 actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if included(p)}
 assert actual==set(files),(actual-set(files),set(files)-actual)
 for n,h in files.items():assert sha(ROOT/n)==h,n
 print('PASS exact file set and SHA256',len(files),flush=True)
 assert sha(ROOT/'pretrained/ppo_500k.zip')=='651989a98dd3ab04294adddd4a0bc0fb63e09b96749e9a980c5041e51bbd5db7'
 assert json.loads((ROOT/'docs/historical_integrity.json').read_text())['passed']
 protocol=json.loads((ROOT/'docs/protocol_sha256.json').read_text())
 for n,h in protocol.items():
  p=ROOT/n
  if n in ['sloshbot/runtime.py','sloshbot/experiment.py']:p=ROOT/'docs'/('baseline_'+Path(n).name)
  assert sha(p)==h,n
 add=json.loads((ROOT/'configs/modal_feedback_addendum.json').read_text())
 for n,h in add['source_after'].items():
  p=ROOT/n if n!='sloshbot/runtime.py' else ROOT/'docs/pre_height_normalization_runtime.py'
  assert sha(p)==h,n
 fix=json.loads((ROOT/'configs/terrain_representation_fix.json').read_text());assert sha(ROOT/fix['source_before_snapshot'])==fix['source_before_sha256'];assert sha(ROOT/'docs/height_normalization_initial_runtime.py')==fix['source_after_sha256']
 qfix=json.loads((ROOT/'configs/heightfield_quantization_addendum.json').read_text());assert sha(ROOT/qfix['source_before_snapshot'])==qfix['source_before_sha256'];assert sha(ROOT/'sloshbot/runtime.py')==qfix['source_after_sha256']
 print('PASS frozen checkpoint, historical receipt, original and versioned contracts',flush=True)
 import mujoco
 expected=['PPO','V2','V3','V2_FFv1','V3_FFv1','V4'];trials=[]
 for i in range(1,4):
  base=None;model0=None;h0=None
  for name in expected:
   folder=ROOT/f'experiments/experiment_1_figure8/{name}_TEST_{i}';r=json.loads((folder/'result.json').read_text());trials.append(r)
   a=np.r_[r['initial']['qpos'],r['initial']['qvel']];h=np.load(folder/'hfield.npy')
   text=(folder/'model.xml').read_text().replace('__ASSET_ROOT__',str(ROOT/'_vendor/ballbotgym/ballbotgym/assets'));m=mujoco.MjModel.from_xml_string(text)
   if base is None:base=a;model0=m;h0=h
   else:
    np.testing.assert_array_equal(base,a);np.testing.assert_array_equal(h0,h)
    for field in ['body_mass','body_inertia','body_ipos','body_iquat','jnt_stiffness','dof_damping','geom_friction','pair_friction','actuator_ctrlrange','actuator_gear']:np.testing.assert_array_equal(getattr(model0,field),getattr(m,field))
   assert not r['metrics']['completion']
 print('PASS 18 trials share matched initial states/terrain/physical arrays; full-task failures retained',flush=True)
 rows=json.loads((ROOT/'docs/research_summary.json').read_text())['experiment_2']
 assert len([r for r in rows if r['run'].startswith('traverse_') and r['stability_pass']])==3
 assert len([r for r in rows if r['run'].startswith('replay_') and r['stability_pass']])==2
 assert any(r['ROM_guard_triggered'] for r in rows if r['run']=='replay_2618001')
 assert not (ROOT/'experiments/experiment_2_teleoperation/modal_replay_2618003').exists()
 print('PASS command-dependent outcomes and modal-transfer gate',flush=True)
 for p in ROOT.rglob('*.md'):
  for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text(errors='replace')):
   if '://' in target or target.startswith('#'):continue
   target=target.split('#')[0]
   if target:assert (p.parent/target).exists(),(p,target)
 keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',(ROOT/'paper/references.bib').read_text()));used=set()
 for p in (ROOT/'paper').rglob('*.tex'):
  text=p.read_text()
  for n in re.findall(r'\\input\{([^}]+)\}',text):assert (ROOT/'paper'/(n+'.tex')).exists(),(p,n)
  for n in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):assert (ROOT/'paper'/n).exists(),(p,n)
  for ns in re.findall(r'\\cite\w*\{([^}]+)\}',text):used.update(n.strip() for n in ns.split(','))
 assert used<=keys,used-keys
 print('PASS Markdown links, manuscript inputs/figures/citation keys',flush=True)
 import cv2
 for row in json.loads((ROOT/'docs/media_provenance.json').read_text())['videos']:
  c=cv2.VideoCapture(str(ROOT/row['video']));assert c.isOpened();assert int(c.get(cv2.CAP_PROP_FRAME_COUNT))==row['frames'];c.release()
 assert json.loads((ROOT/'docs/test_receipt.json').read_text())['passed']
 print(json.dumps(dict(passed=True,manifest_files=len(files),new_physics_steps=0,new_training_transitions=0,tex_compilation='NOT_AVAILABLE',github_publication='NOT_PERFORMED')),flush=True)
if __name__=='__main__':
 verify()
 import runpy
 runpy.run_path(str(ROOT/'scripts/verify_upgrade.py'),run_name='__main__')
