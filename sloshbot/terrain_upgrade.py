"""Physics-based application terrain; original benchmark and gains are unchanged."""
import numpy as np,mujoco,xml.etree.ElementTree as ET,hashlib
from .runtime import TeleopEnv
from ballbot_sloshing_research.track8.terrain import terrain,N,EXTENT,sample
LEVELS=['FLAT','MILD','RANDOM','BUMPS','SLOPES','COURSE']
BALL_RADIUS=.09
FEATURE_HEIGHT=.02*BALL_RADIUS

def smooth(u):
    u=np.clip(u,0,1);return u*u*u*(10+u*(-15+6*u))
def patch(x,a,b,c,d):return smooth((x-a)/(b-a))*(1-smooth((x-c)/(d-c)))
def heights(level,seed=2617001):
    if level not in LEVELS:raise ValueError(level)
    axis=np.linspace(-EXTENT,EXTENT,N);x,y=np.meshgrid(axis,axis);z=np.zeros_like(x);gate=smooth((np.hypot(x,y)-.15)/.1)
    if level=='MILD':z=terrain(.002,seed)[0]*gate
    elif level=='RANDOM':
        rng=np.random.default_rng(seed);z=sum(np.sin(rng.uniform(4,8)*x+rng.uniform(-3,3))*np.sin(rng.uniform(4,8)*y+rng.uniform(-3,3)) for _ in range(4));z*=.0015/max(abs(z).max(),1e-15);z*=gate
    elif level=='BUMPS':z=FEATURE_HEIGHT*patch(x,.18,.36,.36,.54)
    elif level=='SLOPES':z=FEATURE_HEIGHT*patch(x,.18,.35,.45,.62)
    elif level=='COURSE':z=.0012*patch(x,.18,.28,.28,.38)+.0012*patch(x,.42,.50,.54,.62)
    # Profiles span the route's whole corridor; bumps are collision heightfield geometry.
    return z

def region(level):
    return dict(approach_x_max=.14,obstacle_x_min=.18,obstacle_x_max=.62 if level in ['SLOPES','COURSE'] else .54 if level=='BUMPS' else .55,exit_x_min=.67 if level in ['SLOPES','COURSE'] else .62,half_width_m=.25)

def traversal(x,level,physical,complete):
    # Require ordered approach -> obstacle region -> progress beyond it, and route completion.
    r=region(level);x=np.asarray(x)
    if not len(x):return dict(passed=False,reason='NO_NATIVE_TRANSLATION')
    ids=[];start=0
    for mask in [x[:,1]<=r['approach_x_max'],(x[:,1]>=r['obstacle_x_min'])&(x[:,1]<=r['obstacle_x_max'])&(abs(x[:,2])<=r['half_width_m']),x[:,1]>=r['exit_x_min']]:
        found=np.flatnonzero(mask & (np.arange(len(x))>=start))
        if not len(found):break
        start=int(found[0])+1;ids.append(start-1)
    dist=float(np.linalg.norm(np.diff(x[:,1:4],axis=0),axis=1).sum());disp=float(np.linalg.norm(x[-1,1:3]-x[0,1:3]))
    return dict(passed=bool(len(ids)==3 and physical and complete and disp>=.5),ordered_stage_times_s=[float(x[i,0]) for i in ids],horizontal_displacement_m=disp,cumulative_3D_distance_m=dist,elevation_range_m=float(np.ptp(x[:,3])),net_elevation_m=float(x[-1,3]-x[0,3]),region=r,physical_valid=physical,route_completed=complete)

class TerrainEnv(TeleopEnv):
    def __init__(self,level='MILD',**kw):
        super().__init__(terrain='flat',**kw);self.level=level;self.route_states=[];self.surface_snapshots=[];self.motion=[];e=self.e;e.heights[:]=heights(level,e.terrain_seed)
        lo=float(e.heights.min());span=max(float(np.ptp(e.heights)),.004);tree=ET.parse(e.xml_path);root=tree.getroot();h=root.find(".//hfield[@name='track8_collision']");size=h.get('size').split();size[2]=str(span);h.set('size',' '.join(size));root.find(".//geom[@name='terrain']").set('pos',f'0 0 {lo}')
        path=e.xml_path.parent/'terrain_upgrade.xml';tree.write(path);old=e.model;hdata=old.hfield_data.copy();e.model=mujoco.MjModel.from_xml_path(str(path));e.model.hfield_data[:]=hdata;hid=e.model.hfield('track8_collision').id;adr=int(e.model.hfield_adr[hid]);e.model.hfield_data[adr:adr+N*N]=(e.heights.ravel()-lo)/span;self.world_heights=e.model.hfield_data[adr:adr+N*N].reshape(N,N)*span+lo
        for n in ['body_mass','body_inertia','jnt_stiffness','dof_damping','pair_friction','actuator_ctrlrange']:np.testing.assert_array_equal(getattr(old,n),getattr(e.model,n))
        e.xml_path=path;e.model_hash=hashlib.sha256(path.read_bytes()).hexdigest();e.data=mujoco.MjData(e.model);self.native_count=0
        e.model.vis.global_.offwidth=max(960,e.model.vis.global_.offwidth);e.model.vis.global_.offheight=max(720,e.model.vis.global_.offheight)
    def reset(self,**kw):
        out=super().reset(**kw);self.route_states=[];self.motion=[];return out
    def collect(self,e):
        super().collect(e)
        from ballbot_fast_rl.outer_loop.control import kinematics
        s=kinematics(e);R=e.data.xmat[e.tank_id].reshape(3,3).copy();v=np.empty(6);mujoco.mj_objectVelocity(e.model,e.data,mujoco.mjtObj.mjOBJ_XBODY,e.tank_id,v,0)
        self.route_states.append(np.r_[e.data.time,s['p'],s['v']])
        self.motion.append(np.r_[e.data.time,e.data.xpos[e.tank_id],R.ravel(),v])
    def native(self,r):
        super().native(r)
        if self.e.contacts()[1]:raise RuntimeError('BODY_GROUND_INVALID')
