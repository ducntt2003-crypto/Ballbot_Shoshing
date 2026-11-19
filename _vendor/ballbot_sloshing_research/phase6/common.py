import os,json,hashlib
from pathlib import Path
import numpy as np
from ballbot_sloshing_research.phase5.common import REPO,PKG,P4,sha,read,polar,history_features,isolated_rng,OMEGA,LATENT_SCALE
P5=PKG/'results/phase5_20261102_phase_v1'
ROOT=Path(os.environ.get('BALLBOT_SLOSH_PHASE6_RESULTS',PKG/'results/phase6_20261102_feasibility_v1')).resolve()
ID=ROOT.name
DEV=[610100,610101,610102];VAL=[610300,610301,610302];FINAL=list(range(610400,610405));TRAIN=[910100,920100,930100]
CASES=['B1_displacement','B2_velocity','B3_accelerate_stop','B5_tracking']
def save(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def source_hashes():return {str(p.relative_to(REPO)):sha(p) for p in (PKG/'phase6').rglob('*.py')}
