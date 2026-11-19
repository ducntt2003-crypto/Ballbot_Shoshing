"""One controlled change: initialize derivative history from measured initial error."""
import numpy as np,torch
from scipy.spatial.transform import Rotation
from ..integration.controllers import PIDAdapter
class BumplessPIDAdapter(PIDAdapter):
    def __init__(self,dt):super().__init__(dt);self.started=False
    def act(self,env,obs):
        if not self.started:
            R=Rotation.from_rotvec(np.asarray(obs['orientation'])).as_matrix();roll=np.arctan2(R[2,1],R[2,2]);pitch=np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2));self.pid.prev_err=torch.tensor([-pitch,-roll],dtype=torch.float32);self.started=True
        return super().act(env,obs)
class ReferencePIDAdapter(PIDAdapter):
    def __init__(self,dt,bumpless=True,motion='hold'):
        super().__init__(dt);self.started=False;self.bumpless=bumpless;self.motion=motion
    def reset(self):
        self.pid.integral.zero_();self.pid.prev_err.zero_();self.started=False
    def act(self,env,obs):
        R=Rotation.from_rotvec(np.asarray(obs['orientation'])).as_matrix();target=0.
        if self.motion=='pitch_ramp':
            u=np.clip((env.data.time-.02)/.08,0,1);target=.003*(10*u**3-15*u**4+6*u**5)
        if self.bumpless and not self.started:
            roll=np.arctan2(R[2,1],R[2,2]);pitch=np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2));self.pid.prev_err=torch.tensor([target-pitch,-roll],dtype=torch.float32)
        self.started=True;a,_=self.pid.act(torch.as_tensor(R,dtype=torch.float32),setpoint_p=target);return np.clip(a.numpy(),-1,1).astype(np.float32)
