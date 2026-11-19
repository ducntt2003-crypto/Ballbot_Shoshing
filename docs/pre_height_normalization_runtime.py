"""Public adapters. Historical controllers/physics are not edited."""
from .bootstrap import ROOT,setup
setup()
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from ballbot_sloshing_research.large_figure8_rl.env import TrackingPPOEnv, Guard
from ballbot_sloshing_research.track8.controller import TrackingV2FF,TrackingV3FF,FeedforwardMixin
from ballbot_fast_rl.robust_control.v2.variant import ConstraintAwareCascade
from ballbot_fast_rl.robust_control.v3.variant import BandwidthCascade
from ballbot_fast_rl.robust_control.v4.variant import HeadingCascade
from ballbot_fast_rl.robust_control.config import GAINS
from ballbot_fast_rl.outer_loop.control import kinematics,heading

class FigureEnv(TrackingPPOEnv):
    def __init__(self,controller='V4',**kw):
        super().__init__(**kw)
        self.controller_name=controller
    def reset(self,**kw):
        obs,info=super().reset(**kw)
        if self.controller_name in ['V2','V3','V2_FFv1','V3_FFv1']:
            cls={'V2':ConstraintAwareCascade,'V3':BandwidthCascade,'V2_FFv1':TrackingV2FF,'V3_FFv1':TrackingV3FF}[self.controller_name]
            self.e.backbone=cls(GAINS);self.e.backbone.reset(kinematics(self.e),self.e.reference());self.e._prepare_control();self.e.frame_history.clear();self.e.frame_history.append(self.e.frame());self.obs=self.observation()
        self.rows_initial=np.r_[self.e.data.time,self.e.data.qpos,self.e.data.qvel]
        return self.obs.copy(),info
    def step(self,action):
        if self.controller_name not in ['V4','PPO']:
            # Existing step composes the V4 residual; override its action source only.
            class Zero:
                def action(self,e):return np.zeros(3)
            self.e._n_actor=Zero()
        return super().step(action)
    def close(self):
        if not self.closed:
            self.e.pre_sample_hook=None;self.e.sample_hook=None;self.e.close();self.session.__exit__(None,None,None);self.closed=True

class CommandAdapter:
    """Bounded forward/yaw motion -> world p/v/a and heading reference."""
    speed_limit=.04; yaw_rate_limit=.08; acceleration_limit=.04; yaw_acceleration_limit=.08
    def __init__(self,p=(0,0),yaw=0):
        self.p=np.array(p,dtype=float);self.yaw=float(yaw);self.speed=0.;self.turn=0.;self.v=np.zeros(2);self.a=np.zeros(2);self.request=np.zeros(2)
    def request_keys(self,keys):
        keys=set(keys)
        self.request=np.array([int('UP' in keys)-int('DOWN' in keys),int('LEFT' in keys)-int('RIGHT' in keys)],float)
        if 'SPACE' in keys:self.request[:]=0
    def advance(self,dt):
        goal=self.request*np.array([self.speed_limit,self.yaw_rate_limit]);old=self.v.copy()
        self.speed+=float(np.clip(goal[0]-self.speed,-self.acceleration_limit*dt,self.acceleration_limit*dt))
        self.turn+=float(np.clip(goal[1]-self.turn,-self.yaw_acceleration_limit*dt,self.yaw_acceleration_limit*dt))
        self.yaw+=self.turn*dt;self.v=heading(self.yaw)@np.array([self.speed,0.]);self.a=(self.v-old)/dt;self.p+=self.v*dt

class TeleopCascade(FeedforwardMixin,HeadingCascade):
    """Declared adapter: inherited FF targets + existing P6 heading common mode."""
    pass

def playground():
    from ballbot_sloshing_research.track8.terrain import N,EXTENT
    axis=np.linspace(-EXTENT,EXTENT,N);x,y=np.meshgrid(axis,axis)
    # Flat centre, mild +/-1mm rough sector, smooth +/-2mm undulations,
    # and two compact 1mm raised smooth features. No new friction/materials.
    gate=np.clip((np.hypot(x,y)-.25)/.25,0,1);gate=gate*gate*(3-2*gate)
    mild=.001*np.sin(9*x)*np.sin(11*y)
    smooth=.002*np.sin(3*x)*np.sin(3*y)
    z=np.where(y<0,mild,smooth)*gate
    for a,b in [(.7,.7),(-.7,.7)]:z+=.001*np.exp(-((x-a)**2+(y-b)**2)/.08**2)
    z[np.hypot(x,y)<.25]=0
    return z

class TeleopEnv(FigureEnv):
    def __init__(self,terrain='playground',modal_feedback=False,**kw):
        super().__init__(controller='V3_FFv1',**kw)
        self.modal_feedback=modal_feedback;self.controller_name='V4' if modal_feedback else 'V3_FFv1'
        self.terrain_kind=terrain;self.telemetry=[];self.command=CommandAdapter();self.native_count=0
        e=self.e
        if terrain=='flat':e.heights[:]=0
        elif terrain=='playground':e.heights[:]=playground()
        elif terrain!='benchmark':raise ValueError(terrain)
        hid=e.model.hfield('track8_collision').id;adr=int(e.model.hfield_adr[hid]);e.model.hfield_data[adr:adr+e.heights.size]=(e.heights.ravel()+.002)/.004
        e.cfg['workspace_m']=1.8
        # Getter is pure; command evolves only in the explicit native-step hook.
        e.target=lambda:(self.command.p.copy(),self.command.v.copy(),self.command.a.copy())
        original=e._prepare_control
        def prepare():
            if e.data.time>getattr(self,'last_command_t',float('inf'))+1e-10:
                self.command.advance(float(e.data.time)-self.last_command_t);self.last_command_t=float(e.data.time)
            if isinstance(e.backbone,TeleopCascade):e.backbone.yaw_reference=self.command.yaw
            return original()
        e._prepare_control=prepare
    def reset(self,**kw):
        # Canonical assembly reset, same seed/state rules; no in-run correction.
        self.last_command_t=float('inf');self.command=CommandAdapter();super().reset(**kw)
        s=kinematics(self.e);self.command=CommandAdapter(s['p'][:2],s['rpy'][2]);self.e.backbone=TeleopCascade(GAINS);self.e.backbone.reset(s,self.command.p);self.last_command_t=float(self.e.data.time);self.e._prepare_control();self.e.frame_history.clear();self.e.frame_history.append(self.e.frame());self.obs=self.observation();self.telemetry=[];self.native_count=0
        return self.obs.copy(),{}
    def collect(self,e):
        super().collect(e);s=kinematics(e);q,v=e.fluid_state();contact,bad,gap=e.contacts()
        # World angular velocity z equals euler yaw derivative only near upright;
        # use actual derivative of heading for the explicit yaw-rate metric.
        r,p=s['rpy'][:2];wy,wz=s['omega_B'][1:];yawrate=(np.sin(r)*wy+np.cos(r)*wz)/np.cos(p)
        self.telemetry.append(np.r_[e.data.time,self.command.request,self.command.speed,self.command.turn,self.command.v,s['v'][:2],yawrate,s['rpy'],self.command.yaw,e.slosh_energy(),q,v,e.data.ctrl,int(contact),int(bad),gap,self.command.p])
    def native(self,r):
        super().native(r);self.native_count+=1
        if np.max(abs(self.e.data.geom_xpos[self.e.ball_geom,:2]))>1.8:raise Guard('WORKSPACE_INVALID')
        if any(self.e.contacts()[1:2]):raise Guard('BODY_GROUND_INVALID')
