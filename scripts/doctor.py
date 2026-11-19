from pathlib import Path
import sys,json,shutil,importlib.metadata as meta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sloshbot.bootstrap import ROOT,setup
setup()
versions={n:meta.version(n) for n in ['mujoco','torch','numpy','scipy','stable-baselines3','gymnasium','glfw','Pillow','opencv-python']}
print(json.dumps(dict(root=str(ROOT),versions=versions,compiler=shutil.which('cc'),checkpoint_present=(ROOT/'pretrained/ppo_500k.zip').is_file(),notes='No native physics steps; plugins are process-local and do not modify engine binaries.'),indent=2))
