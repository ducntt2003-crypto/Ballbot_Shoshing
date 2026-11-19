import os,json,hashlib,random
from pathlib import Path
from contextlib import contextmanager
import numpy as np,torch
REPO=Path(__file__).resolve().parents[2]
PKG=REPO/'ballbot_sloshing_research'
P3=PKG/'results/phase3_20261102_v1'
ROOT=Path(os.environ.get('BALLBOT_SLOSH_PHASE4_RESULTS',PKG/'results/phase4_20261102_energy_v1')).resolve()
TRAIN_SEEDS=[710100,720100,730100]
DEV=[420200,420201,420202]
VAL=[420300,420301,420302]
FINAL=[420400,420401,420402,420403,420404]
def read(p):return json.loads(Path(p).read_text())
def save(name,obj):
 s=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False);p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(s)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
@contextmanager
def isolated_rng():
 """Evaluation must not reseed training exploration via PPO.load/setup_model."""
 py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state()
 try:yield
 finally:random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(ts)
def rng_state(envs=None):
 ns=np.random.get_state();d=dict(python=random.getstate(),numpy_global=[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]],torch_cpu=torch.get_rng_state().tolist())
 if envs is not None:d['envs']=[dict(numpy=e.unwrapped.np_random.bit_generator.state,phase=e.unwrapped.phase_rng.bit_generator.state,qpos=e.unwrapped.data.qpos.tolist(),qvel=e.unwrapped.data.qvel.tolist()) for e in envs]
 return d
