from pathlib import Path
import os,json,hashlib
REPO=Path(__file__).resolve().parents[1]
PKG=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('BALLBOT_SLOSH_RESULTS',PKG/'results/research_20261101')).resolve()
def save(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 encoded=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)
 with p.open('x') as f:f.write(encoded)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
