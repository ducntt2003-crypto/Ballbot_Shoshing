import os,json,hashlib
from pathlib import Path
import numpy as np,torch
from ballbot_sloshing_research.phase4.common import isolated_rng,rng_state
REPO=Path(__file__).resolve().parents[2];PKG=REPO/'ballbot_sloshing_research';P4=PKG/'results/phase4_20261102_energy_v1'
ROOT=Path(os.environ.get('BALLBOT_SLOSH_PHASE5_RESULTS',PKG/'results/phase5_20261102_phase_v1')).resolve()
EST_TRAIN=list(range(510200,510206));EST_VAL=[510300,510301,510302];EST_TEST=list(range(510400,510405));CAUSAL_SEEDS=list(range(520500,520506))
TRAIN_SEEDS=[810100,820100,830100];DEV=[530200,530201,530202];VAL=[530300,530301,530302];FINAL=list(range(530400,530405))
OMEGA=2*np.pi*2.12873573801632;QREF=.003;VREF=OMEGA*QREF;LATENT_SCALE=np.array([QREF,QREF,VREF,VREF]);PHASE_MIN=.0005

def read(p):return json.loads(Path(p).read_text())
def save(name,obj):
 p=ROOT/name;s=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(s)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
 return h.hexdigest()
def polar(z,omega=OMEGA):
 z=np.asarray(z);q=z[...,:2];v=z[...,2:]/omega;A=np.hypot(q,v);phase=np.arctan2(-v,q);return A,phase,A>=PHASE_MIN
def wrap(angle):return np.arctan2(np.sin(angle),np.cos(angle))
def history_features(observations):
 o=np.asarray(observations);return np.concatenate([o[np.maximum(0,np.arange(len(o))-5*i)] for i in range(10)],axis=1)
