import numpy as np,mujoco
from scipy.spatial.transform import Rotation

DT=.002
MIX=np.column_stack([np.cos(np.arange(3)*2*np.pi/3),np.sin(np.arange(3)*2*np.pi/3)])
def kinematics(env):
    """Body origin W position and velocity at the SAME point; body angular velocity."""
    R=env.data.xmat[env.base].reshape(3,3).copy();six=np.empty(6)
    mujoco.mj_objectVelocity(env.model,env.data,mujoco.mjtObj.mjOBJ_XBODY,env.base,six,0)
    rpy=Rotation.from_matrix(R).as_euler('xyz')
    return dict(p=env.data.xpos[env.base].copy(),v=six[3:].copy(),omega_B=R.T@six[:3],rpy=rpy,R=R)
def heading(yaw):
    c,s=np.cos(yaw),np.sin(yaw);return np.array([[c,-s],[s,c]])
def cap_norm(x,limit):
    x=np.asarray(x,float);n=np.linalg.norm(x);return x*min(1.,limit/max(n,1e-15)),n>limit
class TiltInner:
    """Reference-aware variant of original tilt PI-D; derivative on measurement."""
    def __init__(self,dt=DT,kp=20.,ki=15.,kd=2.,integral_limit=.04):
        self.dt=dt;self.kp=kp;self.ki=ki;self.kd=kd;self.limit=integral_limit;self.reset()
    def reset(self):self.integral=np.zeros(2);self.last_saturated=False
    def act(self,state,reference,valid=True):
        if not valid or not np.isfinite(np.r_[state['rpy'],state['omega_B'],reference]).all():
            return np.zeros(3),dict(inner_integral=self.integral.copy(),motor_saturated=False,inner_aw=True,invalid=True)
        r,p,_=state['rpy'];wx,wy,wz=state['omega_B'];rate=np.array([wx+np.sin(r)*np.tan(p)*wy+np.cos(r)*np.tan(p)*wz,np.cos(r)*wy-np.sin(r)*wz])
        err=np.asarray(reference)-state['rpy'][:2];trial=np.clip(self.integral+err*self.dt,-self.limit,self.limit)
        raw=MIX@(self.kp*err+self.ki*trial-self.kd*rate);sat=bool(np.any(abs(raw)>1.))
        if not sat:self.integral=trial
        raw=MIX@(self.kp*err+self.ki*self.integral-self.kd*rate);self.last_saturated=bool(np.any(abs(raw)>=1.))
        return np.clip(raw,-1,1).astype(np.float32),dict(inner_integral=self.integral.copy(),motor_saturated=self.last_saturated,inner_aw=sat,invalid=False)
class Cascade:
    def __init__(self,config):
        self.cfg=config;self.inner=TiltInner();self.reset(None)
    def reset(self,state,reference=None):
        self.inner.reset();self.integral=np.zeros(2);self.reference=None if state is None else (state['p'][:2].copy() if reference is None else np.array(reference,float));self.tilt=np.zeros(2) if state is None else np.asarray(state['rpy'][:2],dtype=float).copy();self.enabled_time=0.
    def act(self,state,valid=True):
        c=self.cfg
        if self.reference is None:self.reset(state)
        valid=valid and np.isfinite(np.r_[state['p'],state['v'],state['omega_B'],state['rpy']]).all() and np.linalg.norm(state['rpy'][:2])<.20
        if not valid:
            a,log=self.inner.act(state,self.tilt,False)
            return a,dict(log,ep=np.zeros(2),ev=np.zeros(2),integral=self.integral.copy(),desired_tilt=self.tilt.copy(),tilt_clamped=False,slew_limited=False,outer_aw=True,a_cmd=np.zeros(2))
        ep=self.reference-state['p'][:2];ev=-state['v'][:2]
        # G maps [roll,pitch] to heading-frame horizontal acceleration; measured by pilot.
        invG=np.linalg.inv(np.array(c['G']));Q=heading(state['rpy'][2]);trial=np.clip(self.integral+self.inner.dt*ep,-c['integral_limit'],c['integral_limit'])
        acc=c['kp']*ep+c['kd']*ev+c['ki']*trial
        desired=invG@(Q.T@acc);desired,clamp=cap_norm(desired,c['tilt_limit'])
        delta,slew=cap_norm(desired-self.tilt,c['tilt_rate_limit']*self.inner.dt)
        # Conservative conditional integration at any inner/output constraint; no runaway.
        aw=bool(clamp or slew or self.inner.last_saturated)
        if not aw:self.integral=trial
        acc=c['kp']*ep+c['kd']*ev+c['ki']*self.integral
        desired,clamp=cap_norm(invG@(Q.T@acc),c['tilt_limit']);delta,slew=cap_norm(desired-self.tilt,c['tilt_rate_limit']*self.inner.dt);self.tilt+=delta
        a,inner=self.inner.act(state,self.tilt,valid)
        return a,dict(inner,ep=ep,ev=ev,integral=self.integral.copy(),desired_tilt=self.tilt.copy(),tilt_clamped=clamp,slew_limited=slew,outer_aw=aw,a_cmd=acc)
