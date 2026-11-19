"""World XY position, velocity, acceleration and jerk, exact analytic derivatives."""
import numpy as np
def reference(t,kind='figure8',A=.5,B=.25,T=30.):
 w=2*np.pi/T
 if kind=='hold':return (np.zeros(2),)*4
 if kind in ['step','line']:
  duration=1.5 if kind=='step' else T-2
  u=float(np.clip((t-1)/duration,0,1));s=10*u**3-15*u**4+6*u**5
  d=(30*u*u-60*u**3+30*u**4)/duration if 0<u<1 else 0.
  dd=(60*u-180*u*u+120*u**3)/duration**2 if 0<u<1 else 0.
  ddd=(60-360*u+360*u*u)/duration**3 if 0<u<1 else 0.
  vec=np.array([A,B]);return tuple(vec*x for x in [s,d,dd,ddd])
 if kind=='circle':
  return (np.array([A*np.sin(w*t),A*(1-np.cos(w*t))]),
          A*w*np.array([np.cos(w*t),np.sin(w*t)]),
          A*w*w*np.array([-np.sin(w*t),np.cos(w*t)]),
          -A*w**3*np.array([np.cos(w*t),np.sin(w*t)]))
 if kind!='figure8':raise ValueError(kind)
 scale=np.array([A,B]);freq=np.array([w,2*w]);phase=freq*t
 return scale*np.sin(phase),scale*freq*np.cos(phase),-scale*freq**2*np.sin(phase),-scale*freq**3*np.cos(phase)

def feasibility(A=.5,B=.25,T=30.):
 t=np.linspace(0,T,30001);x=np.array([reference(v,A=A,B=B,T=T) for v in t]);p,v,a,j=x.transpose(1,0,2)
 speed=np.linalg.norm(v,axis=1);acc=np.linalg.norm(a,axis=1);curv=abs(v[:,0]*a[:,1]-v[:,1]*a[:,0])/np.maximum(speed,1e-15)**3
 return dict(A_m=A,B_m=B,T_s=T,desired_speed_max_m_s=float(speed.max()),desired_accel_max_m_s2=float(acc.max()),jerk_max_m_s3=float(np.linalg.norm(j,axis=1).max()),curvature_max_1_m=float(curv.max()),min_speed_m_s=float(speed.min()),point_mass_tilt_estimate_rad=float(np.arctan(acc.max()/9.81)),scope='a/g is only a point-mass estimate, not BallBot motor feasibility proof. Curvature uses nonzero analytic speed. Initial reference velocity is nonzero, robot starts at rest; startup tracking error is included.')
