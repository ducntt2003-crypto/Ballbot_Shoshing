import json,csv
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from ballbot_fast_rl.position_hold.core import PositionEnv
from .control import TiltInner,kinematics,heading
ROOT=Path(__file__).resolve().parents[1]/'results/outer_loop_position_control'
def orient(e,yaw):
    Q=Rotation.from_euler('z',yaw).as_matrix()
    for qa,da in [(e.base_q,e.base_d),(e.ball_q,e.ball_d)]:
        rel=e.data.qpos[qa:qa+3]-e.origin;e.data.qpos[qa:qa+3]=e.origin+Q@rel
        old=Rotation.from_quat(e.data.qpos[qa+3:qa+7][[1,2,3,0]]);q=(Rotation.from_matrix(Q)*old).as_quat();e.data.qpos[qa+3:qa+7]=q[[3,0,1,2]]
        e.data.qvel[da:da+3]=Q@e.data.qvel[da:da+3]
    mujoco.mj_forward(e.model,e.data)
def main():
    out=ROOT/'mapping_pilot';out.mkdir(exist_ok=False);rows=[];summary=[]
    for yaw in [0.,np.pi/2]:
        responses={}
        for axis in [0,1]:
            for sign in [-1,1]:
                e=PositionEnv(dict(duration_s=1.2));e.reset(seed=120000,options=dict(tilt=[0,0],zero_velocity=True));orient(e,yaw);inner=TiltInner();states=[]
                for k in range(e.max_steps):
                    state=kinematics(e);t=e.data.time;target=np.zeros(2);u=np.clip((t-.3)/.15,0,1);target[axis]=sign*.003*(10*u**3-15*u**4+6*u**5)
                    action,log=inner.act(state,target);_,_,term,trunc,info=e.step(action);s=kinematics(e);states.append(np.r_[e.data.time,s['p'][:2],s['v'][:2],s['rpy'][:2],target,action])
                    if term or trunc:break
                x=np.array(states);name=f'yaw{yaw:.3f}_axis{axis}_sign{sign}';np.savetxt(out/(name+'.csv'),x,delimiter=',',header='time,px,py,vx,vy,roll,pitch,ref_roll,ref_pitch,a0,a1,a2',comments='')
                mask=(x[:,0]>=.7)&(x[:,0]<=1.1);acc=np.array([np.polyfit(x[mask,0],x[mask,i],1)[0] for i in [3,4]])
                responses[axis,sign]=heading(yaw).T@acc
                summary.append(dict(yaw=yaw,axis=axis,sign=sign,actual_tilt_mean=x[mask,5:7].mean(axis=0).tolist(),acceleration_W_m_s2=acc.tolist(),survival_s=float(e.data.time),reason=info['termination_reason'],peak_action=float(abs(x[:,9:12]).max())))
                e.close()
        G=np.column_stack([(responses[a,1]-responses[a,-1])/.006 for a in [0,1]])
        rows.append(dict(yaw=yaw,G_heading_accel_per_rad=G.tolist(),condition_number=float(np.linalg.cond(G))))
    (out/'summary.json').write_text(json.dumps(dict(maps=rows,episodes=summary),indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
