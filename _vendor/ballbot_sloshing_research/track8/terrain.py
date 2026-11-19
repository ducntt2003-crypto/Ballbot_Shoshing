"""Deterministic smooth, triangulated MuJoCo collision heightfield."""
import numpy as np
N=129;EXTENT=2.;WAVELENGTH=.6
def terrain(amplitude,seed):
 rng=np.random.default_rng(seed);axis=np.linspace(-EXTENT,EXTENT,N);x,y=np.meshgrid(axis,axis);z=np.zeros_like(x)
 for _ in range(4):
  theta=rng.uniform(0,2*np.pi);phase=rng.uniform(-np.pi,np.pi);k=2*np.pi/rng.uniform(.8*WAVELENGTH,1.2*WAVELENGTH)
  z+=np.sin(k*(np.cos(theta)*x+np.sin(theta)*y)+phase)-np.sin(phase)
 z*=amplitude/max(abs(z).max(),1e-15)
 dy,dx=np.gradient(z,axis,axis);sl=np.hypot(dx,dy)
 return z,dict(amplitude_bound_m=amplitude,actual_min_m=float(z.min()),actual_max_m=float(z.max()),wavelength_range_m=[.8*WAVELENGTH,1.2*WAVELENGTH],RMS_slope=float(np.sqrt(np.mean(sl**2))),max_slope=float(sl.max()),grid_N=N,grid_pitch_m=float(axis[1]-axis[0]),half_extent_m=EXTENT,terrain_seed=seed)

def sample(z,x,y):
 # MuJoCo hfield: lower-left to upper-right diagonal, piecewise planar facets.
 u=np.clip((x+EXTENT)/(2*EXTENT)*(N-1),0,N-1-1e-10);v=np.clip((y+EXTENT)/(2*EXTENT)*(N-1),0,N-1-1e-10)
 i,j=int(np.floor(v)),int(np.floor(u));a,b=u-j,v-i;h=2*EXTENT/(N-1)
 if a>=b:
  hx=(z[i,j+1]-z[i,j])/h;hy=(z[i+1,j+1]-z[i,j+1])/h;value=z[i,j]+hx*a*h+hy*b*h
 else:
  hy=(z[i+1,j]-z[i,j])/h;hx=(z[i+1,j+1]-z[i+1,j])/h;value=z[i,j]+hx*a*h+hy*b*h
 normal=np.array([-hx,-hy,1.]);normal/=np.linalg.norm(normal)
 return float(value),normal
