"""Run only missing registered evaluations; never train or overwrite outputs."""
from pathlib import Path
import subprocess,sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 p=json.loads((ROOT/'configs/protocol.json').read_text());start=time.monotonic();ledger=ROOT/'experiments/execution.jsonl';planned=[]
 for c in p['experiment_1']['new_controllers']:
  for i,(seed,terrain) in enumerate(p['experiment_1']['cases'],1):planned.append((f'experiments/experiment_1_figure8/{c}_TEST_{i}', ['--controller',c,'--seed',str(seed),'--terrain-seed',str(terrain),'--duration','30','--wall-s','90'],100))
 for terr in ['benchmark','playground']:planned.append((f'experiments/experiment_2_teleoperation/{terr}_pilot',['--script','configs/keyboard_script.json','--duration','3','--terrain',terr,'--seed','2618001','--wall-s','60'],70))
 for seed in p['experiment_2']['seeds']:planned.append((f'experiments/experiment_2_teleoperation/replay_{seed}',['--script','configs/keyboard_script.json','--duration','16','--terrain','playground','--seed',str(seed),'--wall-s','60'],70))
 for out,args,timeout in planned:
  target=ROOT/out
  if (target/'result.json').exists():print('SKIP_COMPLETED',out,flush=True);continue
  if target.exists():print('INCOMPLETE_DIRECTORY_DO_NOT_OVERWRITE',out,flush=True);continue
  if time.monotonic()-start>p['wall_aggregate_native_ceiling_s']:break
  if 'teleoperation' in out:
   prior=[json.loads(q.read_text()) for q in (ROOT/'experiments/experiment_2_teleoperation').glob('*pilot*/result.json')]
   if any(not q['physical_valid'] or not q['completed'] for q in prior):print('TELEOP_GATE_BLOCKED',flush=True);break
  cmd=[sys.executable,'-B','-m','sloshbot.experiment','--output',out,*args];t=time.monotonic();log=ROOT/'experiments'/(target.name+'.log')
  try:
   with log.open('x') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
   rec=dict(output=out,command=cmd,returncode=r.returncode,wall_s=time.monotonic()-t)
  except subprocess.TimeoutExpired:rec=dict(output=out,timeout=True,wall_s=time.monotonic()-t)
  with ledger.open('a') as f:f.write(json.dumps(rec)+'\n')
  print('FINISHED',out,rec,flush=True)
  results=[json.loads(q.read_text()) for q in (ROOT/'experiments').glob('*/*/result.json')]
  assert sum(q.get('native_steps',0) for q in results)<=p['native_step_ceiling']
  if rec.get('returncode',-1)!=0:print('AFFECTED_RUN_FAILED_INSPECT_LOG',log,flush=True)
if __name__=='__main__':main()
