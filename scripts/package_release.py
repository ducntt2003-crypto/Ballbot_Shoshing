"""Freeze a source-tree release and build a new archive; never publish or delete history."""
from pathlib import Path
import json,hashlib,tarfile,sys
ROOT=Path(__file__).resolve().parents[1]
def included(p):
 return p.is_file() and not any(x in ['.cache','__pycache__'] or x.endswith('.egg-info') for x in p.relative_to(ROOT).parts) and p.name not in ['artifact_manifest.json','MUJOCO_LOG.TXT']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
if __name__=='__main__':
 manifest=ROOT/'docs/artifact_manifest.json';assert not manifest.exists(),'Do not overwrite a release manifest.'
 files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if included(p)}
 with manifest.open('x') as f:json.dump(dict(schema=1,files=files,excludes=['.cache','__pycache__','*.egg-info','MUJOCO_LOG.TXT','manifest itself']),f,indent=2)
 print('FROZEN_MANIFEST',len(files),sha(manifest),flush=True)
