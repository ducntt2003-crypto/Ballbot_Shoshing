import os,time,json
from contextlib import contextmanager
import numpy as np,torch
from .geometry import ROOT
@contextmanager
def legacy_cwd():
    previous=os.getcwd();os.chdir(ROOT/'scripts')
    try:yield
    finally:os.chdir(previous)
def load_legacy_ppo():
    from stable_baselines3 import PPO
    import policies
    with legacy_cwd():return PPO.load(ROOT/'trained_agents/example_model.zip',device='cpu')
class PIDAdapter:
    def __init__(self,dt):
        import policies
        g=json.loads((ROOT/'chrono_fsi/configs/integration/controller.json').read_text())['pid_gains'];self.pid=policies.PID(dt,*g)
    def act(self,env,obs):
        from scipy.spatial.transform import Rotation
        R=torch.as_tensor(Rotation.from_rotvec(np.asarray(obs['orientation'])).as_matrix(),dtype=torch.float32);a,_=self.pid.act(R)
        return np.clip(a.numpy(),-1,1).astype(np.float32)
class PPOAdapter:
    def __init__(self,env):
        if env.observation_contract!='legacy_v1' or not env.cameras or env.controller_dt!=.002:raise ValueError('Legacy PPO requires explicit legacy_v1 and actual cameras; corrected_v2 needs retraining')
        self.policy=load_legacy_ppo()
        if self.policy.observation_space!=env.observation_space:raise ValueError('PPO observation mismatch')
    def act(self,env,obs):return self.policy.predict(obs,deterministic=True)[0]
