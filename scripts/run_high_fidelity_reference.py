"""Optional isolated Chrono static rerun. No robot coupling, no solver installation."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--worker',type=Path,required=True);p.add_argument('--activation',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
if not a.execute:print('Dry run: three20ms static cases only; specify --execute to create a NEW result root.');sys.exit(0)
assert a.worker.is_file() and not a.output.exists();a.output.mkdir(parents=True)
probe="import ctypes,sys; c=ctypes.CDLL('libcuda.so.1'); r=c.cuInit(0); n=ctypes.c_int(); r=r or c.cuDeviceGetCount(ctypes.byref(n)); print('CUDA',r,n.value);sys.exit(0 if r==0 and n.value>0 else 78)"
def command(args):
 return ['bash','-c','source "$1"\nshift\nexec "$@"','sloshbot',str(a.activation.resolve()),*args] if a.activation else args
with (a.output/'gpu_probe.log').open('x') as f:result=subprocess.run(command([sys.executable,'-B','-c',probe]),stdout=f,stderr=subprocess.STDOUT,timeout=15)
if result.returncode:raise SystemExit('GPU BLOCKED: stop; no Chrono kernels attempted. See gpu_probe.log.')
for case in ['dx8_dt50','dx6_dt50','dx6_dt25']:
 with (a.output/(case+'.log')).open('x') as f:result=subprocess.run(command([str(a.worker.resolve()),str(ROOT/'research/high_fidelity'/(case+'.json')),str((a.output/case).resolve())]),stdout=f,stderr=subprocess.STDOUT,timeout=70)
 if result.returncode:raise SystemExit('Affected native run failed; no further run launched.')
print('Static candidate screens completed; do not infer physical validation or open dynamic cases automatically.')
