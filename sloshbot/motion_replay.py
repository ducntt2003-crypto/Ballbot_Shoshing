"""Recorded tank-body-origin motion packet, SI/WORLD. Not an active coupling."""
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation,RotationSpline
class ContainerReplay:
    def __init__(self,motion):
        x=np.asarray(motion);self.t=x[:,0];self.raw=x
        if len(x)<3 or not np.all(np.diff(self.t)>0) or not np.isfinite(x).all():raise ValueError('Finite strictly increasing native motion samples required')
        self.position=CubicHermiteSpline(self.t,x[:,1:4],x[:,16:19]);self.rotation=RotationSpline(self.t,Rotation.from_matrix(x[:,4:13].reshape(-1,3,3)))
    def at(self,t):
        if t<self.t[0] or t>self.t[-1]:raise ValueError('No out-of-range motion extrapolation')
        R=self.rotation(t);return dict(t_s=float(t),position_W_m=self.position(t),velocity_W_m_s=self.position(t,1),acceleration_W_m_s2=self.position(t,2),R_WT=R.as_matrix(),omega_W_rad_s=R.apply(self.rotation(t,1)),alpha_W_rad_s2=R.apply(self.rotation(t,2)))
