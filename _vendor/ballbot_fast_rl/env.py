"""Lean FAST environment: reuse geometry only; no history I/O or Chrono worker."""
import hashlib,json,tempfile
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from chrono_fsi.integration.geometry import make_model,ROOT

DEFAULT=dict(mode='original',physics_dt=.002,control_dt=.002,duration_s=3.,tilt_reset_rad=.025,
 angular_reset_rad_s=.03,position_reset_m=.005,fall_tilt_rad=.35,workspace_m=.75,
 reward_weights=dict(alive=1.,tilt=2.,angular_velocity=.1,position=.15,velocity=.1,torque=.02,action_rate=.01))
OBS_CONTRACT='fast_balance_v1'
ACTION_CONTRACT='wheel_torque_v1'
REWARD_VERSION='balance_integral_v1'

class BallBotFastBalanceEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':100}
    def __init__(self,config=None,render_mode=None):
        self.cfg=dict(DEFAULT,**(config or {}));self.mode=self.cfg['mode']
        if self.mode not in ['original','empty','rigid']:raise ValueError('FAST accepts original/empty/rigid only; no SPH or surrogate')
        self.dt=self.cfg['control_dt'];self.physics_dt=self.cfg['physics_dt'];self.substeps=round(self.dt/self.physics_dt)
        if abs(self.substeps*self.physics_dt-self.dt)>1e-12:raise ValueError('Nonintegral physics/control ratio')
        self.max_steps=round(self.cfg['duration_s']/self.dt)
        self.tmp=tempfile.TemporaryDirectory(prefix='ballbot_fast_model_')
        self.model,self.geometry=make_model(Path(self.tmp.name)/'model.xml',self.mode,{},self.physics_dt)
        self.model_hash=hashlib.sha256((Path(self.tmp.name)/'model.xml').read_bytes()).hexdigest()
        self.data=mujoco.MjData(self.model);self.base=self.model.body('base').id
        self.reference_mass=self.model.body_mass.copy();self.reference_inertia=self.model.body_inertia.copy()
        self.base_joint=self.model.joint('base_free_joint').id;self.ball_joint=self.model.joint('ball_free_joint').id
        self.base_q=int(self.model.jnt_qposadr[self.base_joint]);self.ball_q=int(self.model.jnt_qposadr[self.ball_joint])
        self.base_d=int(self.model.jnt_dofadr[self.base_joint]);self.ball_d=int(self.model.jnt_dofadr[self.ball_joint])
        self.motor_ids=np.array([self.model.actuator(f'motor_{i}').id for i in range(3)])
        self.wheel_ids=np.array([self.model.joint(f'wheel_joint_{i}').id for i in range(3)])
        self.wheel_dofs=self.model.jnt_dofadr[self.wheel_ids].astype(int)
        self.wheel_qpos=self.model.jnt_qposadr[self.wheel_ids].astype(int)
        self.limits=self.model.actuator_ctrlrange[self.motor_ids].copy()
        if not np.allclose(self.limits,[[-10,10]]*3) or not np.allclose(self.model.actuator_gear[self.motor_ids,0],1):raise ValueError('Unexpected torque transmission; audit before using this contract')
        self.torque_scale=self.limits[:,1];self.ball_geom=self.model.geom('the_ball').id;self.ground_geom=self.model.geom('terrain').id
        self.action_space=gym.spaces.Box(-1,1,(3,),np.float32)
        self.observation_space=gym.spaces.Box(-10,10,(18,),np.float32)
        self.render_mode=render_mode;self.renderer=None
    def state(self):
        mujoco.mj_forward(self.model,self.data)
        rot=self.data.xmat[self.base].reshape(3,3).copy();six=np.empty(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.base,six,1)
        xyz=self.data.xpos[self.base].copy();error=rot.T@(xyz-self.origin)
        rv=Rotation.from_matrix(rot).as_rotvec();tilt=float(np.arccos(np.clip(rot[2,2],-1,1)))
        return dict(rot=rot,rotvec=rv,roll_pitch_yaw=Rotation.from_matrix(rot).as_euler('xyz'),
            tilt=tilt,omega=six[:3].copy(),velocity=six[3:].copy(),xyz=xyz,error=error,
            wheels=self.data.qvel[self.wheel_dofs].copy())
    def observation(self,s=None):
        s=s or self.state()
        obs=np.concatenate([s['rot'].T@np.array([0.,0.,-1.]),s['omega']/5,s['wheels']/10,
            s['error'][:2]/.5,s['velocity'][:2],self.previous_action,np.zeros(2)])
        if not np.isfinite(obs).all():raise FloatingPointError('Invalid sensor proxy')
        return np.clip(obs,-10,10).astype(np.float32)
    def contacts(self):
        valid=False;body_ground=False;minimum=0.
        for i in range(self.data.ncon):
            c=self.data.contact[i];pair={int(c.geom1),int(c.geom2)};minimum=min(minimum,float(c.dist))
            if self.ground_geom in pair:
                if self.ball_geom in pair:valid=True
                else:body_ground=True
        return valid,body_ground,minimum
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {};mujoco.mj_resetData(self.model,self.data);mujoco.mj_forward(self.model,self.data)
        self.model.body_mass[:]=self.reference_mass;self.model.body_inertia[:]=self.reference_inertia
        self.mass_scale=1.
        if self.cfg.get('mass_randomization',False):
            self.mass_scale=float(self.np_random.uniform(.98,1.02))
            self.model.body_mass[self.base]*=self.mass_scale;self.model.body_inertia[self.base]*=self.mass_scale
            mujoco.mj_setConst(self.model,self.data)
        tilt=np.asarray(options.get('tilt',self.np_random.uniform(-self.cfg['tilt_reset_rad'],self.cfg['tilt_reset_rad'],2)))
        omega=np.r_[self.np_random.uniform(-self.cfg['angular_reset_rad_s'],self.cfg['angular_reset_rad_s'],2),0.]
        if options.get('zero_velocity'):omega[:]=0
        R=Rotation.from_rotvec(np.r_[tilt,0]).as_matrix();quat=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
        center=self.data.geom_xpos[self.ball_geom].copy();offset=np.r_[self.np_random.uniform(-self.cfg['position_reset_m'],self.cfg['position_reset_m'],2),0.]
        # Rotate the entire free-body assembly about sphere center: wheel/ball gaps stay physical.
        for qa,da in [(self.base_q,self.base_d),(self.ball_q,self.ball_d)]:
            position=center+R@(self.data.qpos[qa:qa+3]-center)+offset
            self.data.qpos[qa:qa+3]=position;self.data.qpos[qa+3:qa+7]=quat
            self.data.qvel[da:da+3]=np.cross(omega,position-center-offset)
            self.data.qvel[da+3:da+6]=R.T@omega
        mujoco.mj_forward(self.model,self.data);self.origin=self.data.xpos[self.base].copy()
        heading=float(self.np_random.uniform(0,2*np.pi)) if self.cfg.get('push_N',0)>0 else 0.
        self.push_W=np.array([np.cos(heading),np.sin(heading),0.])*self.cfg.get('push_N',0.)
        self.previous_action=np.zeros(3);self.steps=0;self.no_contact_s=0.;self.episode_terms={k:0. for k in ['reward_total','reward_alive','penalty_tilt','penalty_angular_velocity','penalty_position','penalty_velocity','penalty_torque','penalty_action_rate','penalty_failure']}
        s=self.state();return self.observation(s),dict(termination_reason='RESET',state=self.info_state(s))
    def info_state(self,s):
        contact,bad,dist=self.contacts()
        ball_six=np.empty(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_GEOM,self.ball_geom,ball_six,0)
        return dict(time_s=float(self.data.time),roll=float(s['roll_pitch_yaw'][0]),pitch=float(s['roll_pitch_yaw'][1]),yaw=float(s['roll_pitch_yaw'][2]),
            tilt=s['tilt'],omega=s['omega'].tolist(),position=s['xyz'].tolist(),position_error=s['error'].tolist(),velocity=s['velocity'].tolist(),
            ball_velocity=ball_six[3:].tolist(),torque=self.data.actuator_force[self.motor_ids].tolist(),ball_ground_contact=contact,body_ground_contact=bad,minimum_contact_distance=dist,external_force_W_N=self.data.xfrc_applied[self.base,:3].tolist())
    def step(self,action):
        a=np.asarray(action,dtype=float)
        if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Action must be finite shape (3,)')
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return np.zeros(18,np.float32),-2.,True,False,dict(termination_reason='INVALID_STATE',success=False)
        a=np.clip(a,-1,1);old=self.previous_action.copy();self.data.ctrl[self.motor_ids]=-self.torque_scale*a
        for _ in range(self.substeps):
            self.data.xfrc_applied[:]=0
            start=self.cfg.get('push_start_s',2.);duration=self.cfg.get('push_duration_s',.1)
            if start<=self.data.time<start+duration:self.data.xfrc_applied[self.base,:3]=self.push_W
            mujoco.mj_step(self.model,self.data)
        self.steps+=1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return np.zeros(18,np.float32),-2.,True,False,dict(termination_reason='INVALID_STATE',success=False)
        s=self.state();contact,bad,dist=self.contacts();self.no_contact_s=0 if contact else self.no_contact_s+self.dt
        w=self.cfg['reward_weights'];raw=dict(alive=1.,tilt=(s['tilt']/.10)**2,angular_velocity=float(np.sum(s['omega']**2)),position=float(np.sum(s['error'][:2]**2))/.3**2,velocity=float(np.sum(s['velocity'][:2]**2))/.5**2,torque=float(np.mean(a*a)),action_rate=float(np.mean(((a-old)/(.2*self.dt/.01))**2)))
        terms={'reward_alive':self.dt*w['alive']}
        for k in raw:
            if k!='alive':terms['penalty_'+k]=self.dt*w[k]*raw[k]
        reward=terms['reward_alive']-sum(v for k,v in terms.items() if k!='reward_alive');terms['reward_total']=reward
        reason='RUNNING'
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():reason='INVALID_STATE'
        elif s['tilt']>self.cfg['fall_tilt_rad'] or bad:reason='FALL'
        elif np.linalg.norm(s['error'][:2])>self.cfg['workspace_m']:reason='OUT_OF_BOUNDS'
        elif self.no_contact_s>.1:reason='CONTACT_FAILURE'
        terminated=reason!='RUNNING';truncated=self.steps>=self.max_steps and not terminated
        if truncated:reason='TIME_LIMIT'
        terms['penalty_failure']=2. if terminated else 0.
        if terminated:reward-=terms['penalty_failure'];terms['reward_total']=reward
        self.previous_action=a.copy()
        for k,v in terms.items():self.episode_terms[k]+=v
        info=dict(termination_reason=reason,state=self.info_state(s),reward_terms=terms,episode_terms=self.episode_terms.copy(),success=bool(truncated))
        return self.observation(s),float(reward),bool(terminated),bool(truncated),info
    def render(self):
        if self.renderer is None:
            # Original zfar was scaled for a finite hfield; flat-plane model extent is smaller.
            # These visual-only fields never change the plant used by training/evaluation.
            self.model.vis.map.zfar=20.
            self.model.vis.headlight.ambient[:]=.4
            self.model.vis.headlight.diffuse[:]=.8
            self.renderer=mujoco.Renderer(self.model,height=480,width=640)
        camera=mujoco.MjvCamera();camera.lookat[:]=self.data.xpos[self.base];camera.lookat[2]=.32;camera.distance=1.4;camera.azimuth=130;camera.elevation=-18
        self.renderer.update_scene(self.data,camera);return self.renderer.render()
    def close(self):
        if self.renderer:self.renderer.close()
        self.tmp.cleanup()
