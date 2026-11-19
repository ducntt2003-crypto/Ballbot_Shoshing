import os,json,csv,time,hashlib
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];PKG=REPO/'ballbot_sloshing_research'
G=PKG/'results/track8_g_20261106_v1';F=PKG/'results/track8_f_20261106_v1';E=PKG/'results/track8_e_20261106_v1';D=PKG/'results/track8_d_20261106_v1';C=PKG/'results/track8_c_20261105_v1'
H=PKG/'results/track8_h_20261106_v2'
ROOT=Path(os.environ.get('TRACK8I_RESULTS',PKG/'results/track8_i_20261106_v1')).resolve()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvwrite(name,rows):
 if not rows:return
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
