from pathlib import Path
import json,sys,subprocess
r=Path(__file__).resolve().parents[1]
jobs=[('modal_pilot',2618001,3,'keyboard_script')]+[(f'modal_replay_{s}',s,16,'keyboard_script') for s in [2618001,2618002,2618003]]+[(f'modal_traverse_{s}',s,26,'terrain_traverse') for s in [2618001,2618002,2618003]]
for name,seed,duration,script in jobs:
 out=r/'experiments/experiment_2_teleoperation'/name
 if (out/'result.json').exists():
  previous=json.loads((out/'result.json').read_text())
  if not previous['physical_valid'] or not previous['completed']:
   print('STOP_AT_RETAINED_FAILED_GATE',out.name,flush=True);break
  continue
 assert not out.exists()
 cmd=[sys.executable,'-B','-m','sloshbot.experiment','--output',str(out.relative_to(r)),'--script',f'configs/{script}.json','--duration',str(duration),'--terrain','playground','--seed',str(seed),'--wall-s','60','--modal-feedback']
 with (r/'experiments'/f'{name}.log').open('x') as f:q=subprocess.run(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=70)
 assert q.returncode==0
 z=json.loads((out/'result.json').read_text());print(name,z['actual_s'],z['termination'],z['stability_pass'],flush=True)
 total=sum(json.loads(p.read_text()).get('native_steps',0) for p in (r/'experiments').glob('*/*/result.json'))
 assert total+2200<=400000
 if not z['physical_valid'] or not z['completed']:print('MODAL_GATE_STOP_NO_FURTHER_TRIALS',flush=True);break
