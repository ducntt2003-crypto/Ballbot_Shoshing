import os,json,hashlib
from pathlib import Path
import numpy as np
from ballbot_sloshing_research.phase6.common import REPO,PKG,P4,P5
from ballbot_sloshing_research.phase5.common import read,sha,isolated_rng,polar,LATENT_SCALE
P6=PKG/'results/phase6_20261102_feasibility_v1'
ROOT=Path(os.environ.get('BALLBOT_SLOSH_PHASE7_RESULTS',PKG/'results/phase7_20261102_sparse_v1')).resolve()
ID=ROOT.name
TRAIN=[1010100,1020100,1030100]
VAL=[610300,610301,610302]
DEV=[710600,710601,710602]
FINAL=list(range(710800,710805))
CASES=['B1_displacement','B2_velocity','B3_accelerate_stop','B5_tracking']
OLD=P6/'observation_study/O1/bc.zip'
THRESHOLDS=[.5,.7,.9]
def source_hashes():
 return {str(p.relative_to(REPO)):sha(p) for p in sorted((PKG/'phase7').rglob('*.py'))}
def save(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvwrite(name,rows):
 import csv
 if not rows:return
 assert all('experiment' not in r or r['experiment']==ID for r in rows)
 rows=[dict(r,experiment=ID) for r in rows];p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
