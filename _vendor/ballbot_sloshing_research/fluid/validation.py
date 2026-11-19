import json,csv,tempfile,time
from pathlib import Path
import numpy as np,mujoco
from scipy.signal import find_peaks
from .reduced_order import ModalParameters,rk4
from ..coupling.model import fluid_elements,reaction_wrench
from ..common import ROOT,save
import xml.etree.ElementTree as ET

def standalone(p,dt,tilt=0,free_cart=False):
 root=ET.Element('mujoco');ET.SubElement(root,'option',timestep=str(dt),integrator='RK4',gravity='0 0 -9.81');world=ET.SubElement(root,'worldbody');tank=ET.SubElement(world,'body',name='tank',euler=f'0 {np.degrees(tilt)} 0')
 if free_cart:
  ET.SubElement(tank,'joint',name='tank_x',type='slide',axis='1 0 0');ET.SubElement(tank,'inertial',mass='.2',pos='0 0 0',diaginertia='.001 .001 .001')
 fluid_elements(tank,p,'liquid');m=mujoco.MjModel.from_xml_string(ET.tostring(root).decode());m.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_ENERGY);d=mujoco.MjData(m);mujoco.mj_forward(m,d);return m,d

def main(output_name='fluid_validation'):
 start=time.monotonic();folder=ROOT/output_name;folder.mkdir(exist_ok=False);p=ModalParameters();records=[];convergence=[];native=[]
 for dt in [.002,.001,.0005]:
  for name in ['static','impulse','harmonic_low','harmonic_resonant','tilt']:
   x=np.array([.002,0,0,0.]) if name=='static' else np.zeros(4);theta=.02 if name=='tilt' else 0.;freq=.7*p.omega if name=='harmonic_low' else p.omega
   acc=lambda t:np.array([.1 if name=='impulse' and .1<=t<.14 else .02*np.sin(freq*t) if name.startswith('harmonic') else 0.,0.]);gt=lambda t:np.array([-9.81*np.sin(theta),0.]);rows=[];work=0.;loss=0.;E0=p.energy(x[:2],x[2:])-p.moving*gt(0)@x[:2]
   for k in range(round(4/dt)):
    t=k*dt
    if output_name=='fluid_validation':
     old=x.copy();x=rk4(lambda t,y:p.rhs(t,y,acc,gt),t,x,dt);power0=-p.moving*acc(t)@old[2:];power1=-p.moving*acc(t+dt)@x[2:];work+=dt*.5*(power0+power1);loss+=dt*.5*p.damping*(old[2:]@old[2:]+x[2:]@x[2:])
    else:
     def augmented(t,y):return np.r_[p.rhs(t,y[:4],acc,gt),-p.moving*acc(t)@y[2:4],p.damping*(y[2:4]@y[2:4])]
     state=rk4(augmented,t,np.r_[x,work,loss],dt);x=state[:4];work,loss=state[4:]
    energy=p.energy(x[:2],x[2:])-p.moving*gt(t+dt)@x[:2];force=p.stiffness*x[:2]+p.damping*x[2:];rows.append([t+dt,*x,energy,work,loss,energy-E0-work+loss,*force])
   a=np.array(rows);peaks,_=find_peaks(a[:,1]);measured=float(1/np.mean(np.diff(a[peaks,0]))) if len(peaks)>2 else None
   csvpath=folder/f'{name}_dt{dt}.csv';np.savetxt(csvpath,a,delimiter=',',header='t,qx,qy,vx,vy,E,work,loss,energy_residual,Fx_modal,Fy_modal',comments='')
   residual=float(abs(a[:,-3]).max());scale=max(float(abs(a[:,5]-E0).max()),E0,1e-8)
   row=dict(case=name,dt_s=dt,frequency_Hz=measured,peak_q_m=float(abs(a[:,1]).max()),final_q_m=float(a[-1,1]),energy_residual_J=residual,energy_relative=residual/scale,work_J=float(a[-1,6]),damping_loss_J=float(a[-1,7]));records.append(row)
   if name=='static':convergence.append(row)
 # Native fixed tank vs exact analytic passive equations, plus native wrench check.
 for tilt in [0,.02]:
  m,d=standalone(p,.002,tilt);ids=[m.body('fluid_fixed').id,m.body('fluid_modal').id];j=m.joint('slosh_x').id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];d.qpos[qa]=.002;mujoco.mj_forward(m,d);initial=d.qpos[qa];maxgap=0.;wrenchgap=0.;rows=[]
  x=np.r_[d.qpos[qa:qa+2],d.qvel[va:va+2]].copy();R=d.xmat[m.body('tank').id].reshape(3,3);gravity=(R.T@m.opt.gravity)[:2]
  for k in range(2000):
   x=rk4(lambda t,y:p.rhs(t,y,gravity_tangent=lambda t:gravity),k*.002,x,.002);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);maxgap=max(maxgap,float(np.max(abs(d.qpos[qa:qa+2]-x[:2]))));W=reaction_wrench(m,d,ids,d.xpos[m.body('tank').id]);expected=R@np.r_[p.stiffness*d.qpos[qa:qa+2]+p.damping*d.qvel[va:va+2]+p.fixed*gravity,-p.mass*9.81*np.cos(tilt)]
   wrenchgap=max(wrenchgap,float(np.linalg.norm(W[:3]-expected)));rows.append([d.time,*d.qpos[qa:qa+2],*W])
  np.savetxt(folder/f'native_tilt{tilt}.csv',rows,delimiter=',',header='t,qx,qy,Fx,Fy,Fz,Tx,Ty,Tz',comments='');native.append(dict(tilt_rad=tilt,max_ODE_position_gap_m=maxgap,max_force_gap_N=wrenchgap,final_q_m=float(d.qpos[qa]),expected_static_q_m=float(gravity[0]/p.omega**2)))
 # Free horizontal cart: internal reaction, momentum and work-energy with damping.
 m,d=standalone(p,.002,free_cart=True);tx=m.joint('tank_x').qposadr[0];jq=m.joint('slosh_x').qposadr[0];jv=m.joint('slosh_x').dofadr[0];tv=m.joint('tank_x').dofadr[0];d.qpos[jq]=.002;mujoco.mj_forward(m,d);initialE=d.energy.sum();loss=0.;maxE=0.;maxP=0.;body=m.body('tank').id
 for k in range(2000):
  old=d.qvel[jv:jv+2].copy();mujoco.mj_step(m,d);mujoco.mj_forward(m,d);v=d.qvel[jv:jv+2];loss+=.001*p.damping*(old@old+v@v);maxE=max(maxE,float(abs(d.energy.sum()-initialE+loss)));momentum=(.2+p.mass)*d.qvel[tv]+p.moving*d.qvel[jv];maxP=max(maxP,abs(float(momentum)))
 checks=dict(static_decay=all(r['final_q_m']**2<.002**2 and r['damping_loss_J']>0 for r in convergence),frequency=all(abs(r['frequency_Hz']/(p.omega*np.sqrt(1-p.zeta**2)/(2*np.pi))-1)<.01 for r in convergence),energy=all(r['energy_relative']<.005 for r in records),native_ODE=max(r['max_ODE_position_gap_m'] for r in native)<1e-8,native_wrench=max(r['max_force_gap_N'] for r in native)<1e-7,timestep_amplitude=max(abs(r['peak_q_m']/convergence[-1]['peak_q_m']-1) for r in convergence)<.01,free_cart_momentum=maxP<1e-9,free_cart_energy=maxE/max(p.energy([.002,0],[0,0]),1e-8)<.005)
 save(output_name+'/summary.json',dict(parameters=p.metadata(),records=records,native=native,free_cart_momentum_max_kg_m_s=maxP,free_cart_energy_residual_J=maxE,checks=checks,gate1=all(checks.values()),work_accounting='same RK4 stages as state' if output_name!='fluid_validation' else 'endpoint trapezoid',scope='Internal equation/numerical validation of an approximate first-mode model, not physical water/CFD validation',wall_s=time.monotonic()-start))
 print('Fluid gates',checks,flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='fluid_validation');args=parser.parse_args();main(args.output_name)
