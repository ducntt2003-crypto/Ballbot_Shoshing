import xml.etree.ElementTree as ET
import numpy as np,mujoco
from .common import sha
from .reference import reference
from .terrain import terrain,N,EXTENT,sample
from .controller import TrackingV2FF,TrackingV3FF
from ballbot_sloshing_research.phase2.env import Phase2Env
from ballbot_sloshing_research.coupling.model import build_model
from ballbot_fast_rl.robust_control.v2.variant import ConstraintAwareCascade
from ballbot_fast_rl.robust_control.v3.variant import BandwidthCascade
from ballbot_fast_rl.robust_control.config import GAINS
from ballbot_fast_rl.outer_loop.control import kinematics

class TrackEnv(Phase2Env):
 def __init__(self,controller='V2',payload='liquid',amplitude=0.,terrain_seed=981001,kind='figure8',A=.5,B=.25,T=30.,duration=None):
  self.track_cfg=dict(kind=kind,A=A,B=B,T=T);self.control_label=controller;self.terrain_amplitude=amplitude;self.terrain_seed=terrain_seed
  super().__init__(backbone='V2',observation='O1',reward='R1',scenario='nominal',duration=duration or T,frozen=False)
  if payload=='rigid':
   self.model,self.xml_path=build_model(self.tmp.name,self.pfluid,'rigid',.002);self.data=mujoco.MjData(self.model);self.liquid_mode='rigid';self.slosh_joints=np.array([],int);self.slosh_q=np.array([],int);self.slosh_v=np.array([],int);self.fluid_ids=[self.model.body('fluid_fixed').id]
  if payload not in ['rigid','liquid']:raise ValueError(payload)
  self.heights,self.terrain_info=terrain(amplitude,terrain_seed)
  if amplitude:
   tree=ET.parse(self.xml_path);root=tree.getroot();asset=root.find('asset')
   if asset is None:asset=ET.SubElement(root,'asset')
   ET.SubElement(asset,'hfield',name='track8_collision',nrow=str(N),ncol=str(N),size=f'{EXTENT} {EXTENT} {2*amplitude} .05')
   ground=root.find(".//geom[@name='terrain']");ground.set('type','hfield');ground.set('hfield','track8_collision');ground.set('pos',f'0 0 {-amplitude}');ground.attrib.pop('size',None)
   path=self.xml_path.parent/'track8_collision.xml';tree.write(path);self.xml_path=path;self.model=mujoco.MjModel.from_xml_path(str(path));hid=self.model.hfield('track8_collision').id;adr=int(self.model.hfield_adr[hid]);self.model.hfield_data[adr:adr+N*N]=(self.heights.ravel()+amplitude)/(2*amplitude);self.data=mujoco.MjData(self.model)
  # A rebuilt rigid model removes two joints and one visual geom. IDs are not stable.
  self.base=self.model.body('base').id;self.base_joint=self.model.joint('base_free_joint').id;self.ball_joint=self.model.joint('ball_free_joint').id
  self.base_q=int(self.model.jnt_qposadr[self.base_joint]);self.base_d=int(self.model.jnt_dofadr[self.base_joint]);self.ball_q=int(self.model.jnt_qposadr[self.ball_joint]);self.ball_d=int(self.model.jnt_dofadr[self.ball_joint])
  self.ball_geom=self.model.geom('the_ball').id;self.ground_geom=self.model.geom('terrain').id;self.tank_id=self.model.body('water_tank').id
  self.wheel_ids=np.array([self.model.joint(f'wheel_joint_{i}').id for i in range(3)]);self.wheel_dofs=self.model.jnt_dofadr[self.wheel_ids].astype(int);self.wheel_qpos=self.model.jnt_qposadr[self.wheel_ids].astype(int)
  self.motor_ids=np.array([self.model.actuator(f'motor_{i}').id for i in range(3)]);self.limits=self.model.actuator_ctrlrange[self.motor_ids].copy();self.torque_scale=self.limits[:,1]
  self.model.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_ENERGY)
  self.model_hash=sha(self.xml_path);self.payload=payload
  self.reference_mass=self.model.body_mass.copy();self.reference_inertia=self.model.body_inertia.copy()
  cls={'V2':ConstraintAwareCascade,'V3':BandwidthCascade,'V2_FFv1':TrackingV2FF,'V3_FFv1':TrackingV3FF}.get(controller,ConstraintAwareCascade)
  self.backbone=cls(GAINS)
 def target(self):
  p,v,a,_=reference(float(self.data.time),**self.track_cfg)
  return self.origin[:2]+p,v,a
 def _prepare_control(self):
  s=kinematics(self);p,v,a=self.target()
  if hasattr(self.backbone,'set_reference'):self.backbone.set_reference(p,v,a)
  else:self.backbone.reference=p.copy()
  self.base_action=self.backbone.act(s,True)[0];return s
 def reset(self,*,seed=None,options=None):
  options=dict(options or {});options.setdefault('slosh_q',np.zeros(2));options.setdefault('slosh_v',np.zeros(2))
  o,info=super().reset(seed=seed,options=options)
  if self.terrain_amplitude:
   xy=self.data.qpos[self.ball_q:self.ball_q+2];height,normal=sample(self.heights,*xy);radius=self.model.geom_size[self.ball_geom,0]
   # Initial placement only, before t=0. No trial-time state corrections.
   dz=height+radius*(1/normal[2]-1)
   for q in [self.base_q,self.ball_q]:self.data.qpos[q+2]+=dz
   mujoco.mj_forward(self.model,self.data);self.origin=self.data.xpos[self.base].copy();self.backbone.reset(kinematics(self),self.reference());self._prepare_control();self.frame_history.clear();self.frame_history.append(self.frame());o=self.observation()
  info.update(initial_qpos=self.data.qpos.copy(),initial_qvel=self.data.qvel.copy());return o,info
