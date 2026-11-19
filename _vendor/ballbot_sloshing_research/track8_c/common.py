import os,json,csv,io,hashlib
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];PKG=REPO/'ballbot_sloshing_research';OLD=PKG/'results/track8_20261105_v1'
ROOT=Path(os.getenv('TRACK8C_RESULTS',PKG/'results/track8_c_20261105_v1')).resolve();ID=ROOT.name
REPRO=[981001,981002,981003];DEV=[1081001,1081002,1081003];VAL=[1082001,1082002,1082003];FINAL=list(range(1083001,1083006));TDEV=[1084001,1084002,1084003];TVAL=[1085001,1085002,1085003];TFINAL=list(range(1086001,1086006))
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)
 with p.open('x') as f:f.write(text)
def csvwrite(name,rows):
 if not rows:return
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r if k!='experiment'));s=io.StringIO(newline='');w=csv.DictWriter(s,['experiment']+keys);w.writeheader();w.writerows(dict(r,experiment=ID) for r in rows)
 with p.open('x',newline='') as f:f.write(s.getvalue())
