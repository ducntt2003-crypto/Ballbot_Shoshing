from dataclasses import dataclass,asdict
import numpy as np
@dataclass(frozen=True)
class ModalParameters:
 volume:float=.0005
 radius:float=.07
 rho:float=998.
 zeta:float=.02
 frequency_scale:float=1.
 def __post_init__(self):
  if not np.isfinite([self.volume,self.radius,self.rho,self.zeta,self.frequency_scale]).all() or min(self.volume,self.radius,self.rho,self.frequency_scale)<=0 or self.zeta<0:
   raise ValueError('Finite positive mass/geometry/frequency and nonnegative passive damping required')
 @property
 def depth(self):return self.volume/(np.pi*self.radius**2)
 @property
 def mass(self):return self.rho*self.volume
 @property
 def omega(self):return self.frequency_scale*np.sqrt(9.81*1.8411837813406593/self.radius*np.tanh(1.8411837813406593*self.depth/self.radius))
 @property
 def moving(self):
  xi=1.8411837813406593
  return self.mass*2*self.radius*np.tanh(xi*self.depth/self.radius)/(self.depth*xi*(xi**2-1))
 @property
 def fixed(self):return self.mass-self.moving
 @property
 def stiffness(self):return self.moving*self.omega**2
 @property
 def damping(self):return 2*self.zeta*self.omega*self.moving
 @property
 def inertia(self):return self.mass*np.diag([(3*self.radius**2+self.depth**2)/12]*2+[self.radius**2/2])
 def metadata(self):
  return dict(**asdict(self),depth=self.depth,mass=self.mass,moving_mass=self.moving,fixed_mass=self.fixed,omega=self.omega,frequency_Hz=self.omega/(2*np.pi),k=self.stiffness,c=self.damping,model='linear_cylinder_first_mode_2D_spring_v1',mass_accounting='one moving mass with two slides, not one mass per axis',damping='ASSUMED',moment_model='COM-height approximation; not full modal pressure-moment equivalence')
 def energy(self,q,v):return float(.5*self.moving*np.dot(v,v)+.5*self.stiffness*np.dot(q,q))
 def rhs(self,t,x,acc=lambda t:np.zeros(2),gravity_tangent=lambda t:np.zeros(2)):
  return np.r_[x[2:],-self.omega**2*x[:2]-2*self.zeta*self.omega*x[2:]+gravity_tangent(t)-acc(t)]
def rk4(fun,t,x,dt):
 a=fun(t,x);b=fun(t+dt/2,x+dt*a/2);c=fun(t+dt/2,x+dt*b/2);d=fun(t+dt,x+dt*c);return x+dt*(a+2*b+2*c+d)/6
