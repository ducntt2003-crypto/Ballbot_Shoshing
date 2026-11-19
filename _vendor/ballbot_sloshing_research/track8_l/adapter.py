"""Public process-local collision dispatch, original engine binary unchanged."""
import ctypes,subprocess,shutil
import mujoco
from .common import *
from .exact_oracle import solve
BUILD=ROOT/'adapter_build';LIB=None;CALLBACK=None;SELECTOR_RECORDS=[]
CB=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
def compile_all():
 BUILD.mkdir(exist_ok=False);pkg=Path(mujoco.__file__).parent;src=Path(__file__).with_name('exact_adapter.c');out=BUILD/'libtrack8l_exact.so';cmd=[shutil.which('cc'),'-shared','-fPIC','-O2','-std=c99','-Wall','-Wextra','-Werror','-I'+str(pkg/'include'),str(src),'-L'+str(pkg),'-l:libmujoco.so.3.3.2','-Wl,-rpath,'+str(pkg),'-lm','-o',str(out)]
 with (BUILD/'build.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
 save('adapter_build/provenance.json',dict(version='TRACK8L_EXACT_GLOBAL_POINT_V1',command=cmd,source_sha256=sha(src),binary_sha256=sha(out),engine_binary_sha256=sha(pkg/'libmujoco.so.3.3.2'),engine_rebuilt=False))
def selector(n,pp,tt,qq):
 tick=time.perf_counter()
 try:
  if n>512:raise ValueError('Exact oracle bound512')
  p=np.ctypeslib.as_array(pp,shape=(3,)).copy();tri=np.ctypeslib.as_array(tt,shape=(n*9,)).reshape(n,3,3).copy();r=solve(tri,p);accepted=r['unique']
  if accepted:
   for k in range(3):qq[k]=float(r['q'][k])
  SELECTOR_RECORDS.append(dict(triangles=n,unique=accepted,global_points=r['global_points'],index=r['index'],wall_s=time.perf_counter()-tick,squared_distance_exact=str(r['d2'])));return r['index'] if accepted else -1
 except BaseException as err:SELECTOR_RECORDS.append(dict(error=repr(err),wall_s=time.perf_counter()-tick));return -1

def lib():
 global LIB,CALLBACK
 if LIB is None:
  LIB=ctypes.CDLL(str(BUILD/'libtrack8l_exact.so'));LIB.track8l_counts.argtypes=[ctypes.POINTER(ctypes.c_ulong)];LIB.track8l_exact_counts.argtypes=[ctypes.POINTER(ctypes.c_ulong)];LIB.track8l_event.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)];LIB.track8l_last.argtypes=[ctypes.POINTER(ctypes.c_double)];LIB.track8l_classifier.argtypes=[ctypes.POINTER(ctypes.c_double)];LIB.track8l_set_selector.argtypes=[CB];CALLBACK=CB(selector);LIB.track8l_set_selector(CALLBACK)
 return LIB
def phase(n):lib().track8l_phase(n)
def stats():
 a=(ctypes.c_ulong*12)();lib().track8l_counts(a);b=(ctypes.c_ulong*3)();lib().track8l_exact_counts(b);return dict(zip(['calls','accepted','max_search_triangles','max_active_triangles','unsupported','boundary','search_cap','deep','multipatch','ambiguous','forced_abort','numeric_uncertain'],map(int,a)),exact_calls=int(b[0]),exact_resolved=int(b[1]),exact_rejected=int(b[2]),selector_records=SELECTOR_RECORDS.copy())
def event():
 a=(ctypes.c_double*5)();b=(ctypes.c_double*512)();n=lib().track8l_event(a,b);return dict(time_s=a[0],reason=int(a[1]),phase=int(a[2]),native_eval=int(a[3]),native_state=list(b)[:min(n,512)],native_state_size=n)
class Adapter:
 def __init__(self,query=False,force_eval=0):self.query=query;self.force_eval=force_eval
 def __enter__(self):
  assert lib().track8l_install()==1;lib().track8l_reset();SELECTOR_RECORDS.clear();lib().track8l_mode(0 if self.query else 1);lib().track8l_phase(0);lib().track8l_force(self.force_eval);return self
 def __exit__(self,*args):lib().track8l_force(0);lib().track8l_phase(0);assert lib().track8l_uninstall()==1
if __name__=='__main__':compile_all()
