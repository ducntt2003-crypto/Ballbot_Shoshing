import os,json,hashlib
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]
PKG=REPO/'ballbot_sloshing_research'
ROOT=Path(os.getenv('TRACK8_RESULTS',PKG/'results/track8_20261105_v1')).resolve()
ID=ROOT.name
P9=PKG/'results/phase9_rl_20261103_replication_v1'
DEV=[981001,981002,981003];VAL=[982001,982002];FINAL=[983001,983002,983003,983004,983005]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvwrite(name,rows):
 import csv,io
 if not rows:return
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r if k!='experiment'))
 content=io.StringIO(newline='');w=csv.DictWriter(content,['experiment']+keys);w.writeheader();w.writerows(dict(r,experiment=ID) for r in rows);value=content.getvalue()
 if p.exists():
  if p.read_bytes()==value.encode():return
  original=p;p=p.with_stem(p.stem+'_recovered')
  if p.exists():
   assert p.read_bytes()==value.encode();return
  save(str(p.relative_to(ROOT))+'.recovery.json',dict(original=str(original),original_sha256=sha(original),corrected=str(p),reason='Preserve invalid partialCSV from failed export; corrected file written separately, no physics/trial rerun.'))
 with p.open('x',newline='') as f:f.write(value)
