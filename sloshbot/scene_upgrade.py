"""Terrain-projected trail, landmarks and read-only ROM surface for native MuJoCo."""
import numpy as np,mujoco
from .surface import draw_surface
from ballbot_sloshing_research.track8.terrain import sample

def connector(scene,a,b,kind=mujoco.mjtGeom.mjGEOM_LINE,color=(.1,.95,.18,1),width=3):
    if scene.ngeom>=scene.maxgeom:return
    g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,kind,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,np.float32));mujoco.mjv_connector(g,kind,width,np.array(a),np.array(b));scene.ngeom+=1

def decorate(scene,m,d,world_heights,history,arrow=True):
    path=np.asarray(history)
    if len(path):
        # Thin display samples to preserve responsiveness; source history is actual XYZ.
        path=path[::max(1,len(path)//220)]
        points=np.array([[a[0],a[1],sample(world_heights,a[0],a[1])[0]+.0025] for a in path])
        for a,b in zip(points[:-1],points[1:]):connector(scene,a,b)
    # World coordinate grid is a scene decoration, not invisible collision obstacles.
    for x in np.arange(-1.5,1.51,.5):connector(scene,[x,-1.5,.004],[x,1.5,.004],color=(.3,.35,.4,.4),width=1)
    for y in np.arange(-1.5,1.51,.5):connector(scene,[-1.5,y,.004],[1.5,y,.004],color=(.3,.35,.4,.4),width=1)
    base=m.body('base').id;p=d.xpos[base];R=d.xmat[base].reshape(3,3)
    if arrow:connector(scene,p+np.array([0,0,.05]),p+.12*R[:,0]+np.array([0,0,.05]),kind=mujoco.mjtGeom.mjGEOM_ARROW,color=(1,.7,.1,1),width=.008)
    return draw_surface(scene,m,d)
