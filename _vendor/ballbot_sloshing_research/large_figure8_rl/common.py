import os,json,csv,time,hashlib
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];PKG=REPO/'ballbot_sloshing_research';O=PKG/'results/track8_o_20261107_v1';N=PKG/'results/track8_n_20261107_v1';M=PKG/'results/track8_m_20261107_v1';L=PKG/'results/track8_l_20261107_v1'
ROOT=Path(os.environ.get('LARGE8_RESULTS',PKG/'results/large_figure8_rl_20261111_v1')).resolve()
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

P=PKG/'results/track8_p_20261107_v1'

Q=PKG/'results/track8_q_20261109_v1'

R=PKG/'results/track8_r_20261109_v1'

S=PKG/'results/track8_s_20261109_v1'

T=PKG/'results/track8_t_20261109_v1'

U=PKG/'results/track8_u_20261109_v1'

V=PKG/'results/track8_v_20261109_v1'

W=PKG/'results/track8_w_20261109_v1'

X=PKG/'results/track8_x_20261110_v1'

Y=PKG/'results/track8_y_20261110_v1'
Z=PKG/'paper/track8_z_20261111_v1'
