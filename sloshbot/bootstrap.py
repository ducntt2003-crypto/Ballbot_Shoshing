"""Resolve the bundled, unchanged historical runtime; build local observer plugins."""
from pathlib import Path
import sys, os, json
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'_vendor'
if str(VENDOR) not in sys.path:sys.path.insert(0,str(VENDOR))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

def setup():
    import mujoco
    if mujoco.__version__!='3.3.2':raise RuntimeError('This release requires MuJoCo 3.3.2; native observer ABI is versioned.')
    from ballbot_sloshing_research.track8_l import adapter as A
    from ballbot_sloshing_research.track8_i import observer as O
    cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
    A.BUILD=cache/'contact';O.BUILD=cache/'observer'
    # Route build metadata into cache too; keep the immutable public tree clean.
    def cache_writer(base):
        def record(name,obj):
            with (base/Path(name).name).open('x') as f:json.dump(obj,f,indent=2)
        return record
    A.save=cache_writer(A.BUILD);O.save=cache_writer(O.BUILD)
    if not A.BUILD.exists():A.compile_all()
    if not O.BUILD.exists():O.compile_observer()
    import torch
    torch.set_num_threads(1)
    return A,O
