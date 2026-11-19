import os,json,csv,time,hashlib
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];PKG=REPO/'ballbot_sloshing_research';K=PKG/'results/track8_k_20261107_v1';J=PKG/'results/track8_j_20261106_v1';C=PKG/'results/track8_c_20261105_v1'
ROOT=Path(os.environ.get('TRACK8L_RESULTS',PKG/'results/track8_l_20261107_v1')).resolve()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(n,o):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvwrite(n,rows):
 if not rows:return
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='') as f:w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
