"""Read-only first-mode Bessel surface reconstruction; visualization, not CFD."""
import numpy as np,mujoco
from scipy.special import j1,jv
from ballbot_sloshing_research.fluid.reduced_order import ModalParameters
ALPHA=1.8411837813406593

def elevation(x,y,q,p=None):
    p=p or ModalParameters();x=np.asarray(x);y=np.asarray(y);r=np.hypot(x,y);k=ALPHA/p.radius
    # Continuous first volume-moment matches the actual equivalent modal COM:
    # delta_COM_xy = (moving/mass)*q. No amplitude fitting to rendered frames.
    radial=np.divide(j1(k*r),r,out=np.full_like(r,k/2,dtype=float),where=r>1e-14)
    return (p.moving/p.mass)*k*p.depth/jv(2,ALPHA)*radial*(q[0]*x+q[1]*y)

def mesh(q,p=None,nr=4,nt=32):
    p=p or ModalParameters();rad=np.repeat(np.arange(1,nr+1)*p.radius/nr,nt);theta=np.tile(np.arange(nt)*2*np.pi/nt,nr)
    x=np.r_[0,rad*np.cos(theta)];y=np.r_[0,rad*np.sin(theta)];points=np.c_[x,y,p.depth+elevation(x,y,q,p)];faces=[]
    for it in range(nt):faces.append([0,1+it,1+(it+1)%nt])
    for ir in range(nr-1):
        for it in range(nt):
            a=1+ir*nt+it;b=1+ir*nt+(it+1)%nt;c=a+nt;d=b+nt;faces.extend([[a,c,d],[a,d,b]])
    return points,np.array(faces)

def triangle(scene,vertices,color=(.04,.6,.95,.82)):
    if scene.ngeom>=scene.maxgeom:return
    a,b,c=vertices;u=b-a;v=c-a;n=np.cross(u,v);lengths=np.array([np.linalg.norm(u),np.linalg.norm(v),np.linalg.norm(n)])
    if min(lengths)<1e-16:return
    matrix=np.column_stack([u/lengths[0],v/lengths[1],n/lengths[2]])
    g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_TRIANGLE,lengths,a,matrix.ravel(),np.array(color,np.float32));g.category=mujoco.mjtCatBit.mjCAT_DECOR;g.specular=.6;g.shininess=.5;scene.ngeom+=1

def draw_surface(scene,model,data,p=None):
    p=p or ModalParameters();tank=model.body('water_tank').id;R=data.xmat[tank].reshape(3,3);o=data.xpos[tank];q=np.array([data.qpos[model.joint(n).qposadr[0]] for n in ['slosh_x','slosh_y']]);vertices,faces=mesh(q,p);world=vertices@R.T+o
    # Only visual scene copies are changed, never model/data physics arrays.
    for i in range(scene.ngeom):
        g=scene.geoms[i]
        if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and 0<=g.objid<model.ngeom:
            name=model.geom(g.objid).name or ''
            if name=='equivalent_modal_mass':g.rgba[3]=0
            elif name.startswith('tank_side_'):g.rgba[3]=.13
    for face in faces:triangle(scene,world[face])
    return dict(q_m=q.tolist(),surface_min_T_m=float(vertices[:,2].min()),surface_max_T_m=float(vertices[:,2].max()),visual_only=True,bottom_wetted=bool(vertices[:,2].min()>0),overflow=bool(vertices[:,2].max()>.16))
