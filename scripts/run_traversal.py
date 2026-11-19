from pathlib import Path
import json,sys,subprocess
r=Path(__file__).resolve().parents[1]
for name in ['flat_pilot_iofix','benchmark_pilot','playground_pilot']:
 p=r/'experiments/experiment_2_teleoperation'/name/'result.json'
 assert p.exists() and json.loads(p.read_text())['physical_valid'],name
for seed in [2618001,2618002,2618003]:
 out=r/f'experiments/experiment_2_teleoperation/traverse_{seed}'
 if (out/'result.json').exists():
  previous=json.loads((out/'result.json').read_text())
  if not previous['physical_valid'] or not previous['completed']:
   print('STOP_AT_RETAINED_FAILED_GATE',out.name,flush=True);break
  continue
 assert not out.exists()
 cmd=[sys.executable,'-B','-m','sloshbot.experiment','--output',str(out.relative_to(r)),'--script','configs/terrain_traverse.json','--duration','26','--terrain','playground','--seed',str(seed),'--wall-s','60']
 with (r/'experiments'/f'traverse_{seed}.log').open('x') as f:q=subprocess.run(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=70)
 assert q.returncode==0
 result=json.loads((out/'result.json').read_text());print(out.name,result['actual_s'],result['termination'],flush=True)
 total=sum(json.loads(p.read_text()).get('native_steps',0) for p in (r/'experiments').glob('*/*/result.json'))
 assert total+2200<=400000
 if not result['physical_valid']:break
