from pathlib import Path
import sys,unittest,json,time
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
import mujoco
count=0;old=mujoco.mj_step

def counted(m,d,*args,**kw):
    global count
    count+=int(kw.get('nstep',args[0] if args else 1));return old(m,d,*args,**kw)
mujoco.mj_step=counted
start=time.monotonic();suite=unittest.defaultTestLoader.discover(str(r/'tests'));result=unittest.TextTestRunner(verbosity=2).run(suite)
summary=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),native_steps=count,wall_s=time.monotonic()-start,command='python -B scripts/test_essential.py')
# Receipts belong to the caller's new directory; stdout is safe for repeat verification.
print('ESSENTIAL_TEST_RECEIPT',json.dumps(summary),flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
