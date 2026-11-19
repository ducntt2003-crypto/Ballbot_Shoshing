import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from chrono_fsi.integration.geometry import make_model
from ..fluid.reduced_order import ModalParameters

def fluid_elements(tank,p,mode):
 if mode=='empty':return
 if mode not in ['rigid','liquid','oneway_diagnostic']:raise ValueError(mode)
 center=[0,0,p.depth/2];tiny=1e-8
 def inertial(body,m,I):ET.SubElement(body,'inertial',mass=str(m),pos='0 0 0',diaginertia=' '.join(map(str,np.diag(I))))
 fixed=ET.SubElement(tank,'body',name='fluid_fixed',pos=' '.join(map(str,center)))
 if mode=='rigid':inertial(fixed,p.mass,p.inertia);return
 inertial(fixed,p.fixed,p.inertia-np.eye(3)*tiny)
 moving=ET.SubElement(tank,'body',name='fluid_modal',pos=' '.join(map(str,center)))
 inertial(moving,p.moving,np.eye(3)*tiny)
 for axis,name in [('1 0 0','slosh_x'),('0 1 0','slosh_y')]:ET.SubElement(moving,'joint',name=name,type='slide',axis=axis,stiffness=str(p.stiffness),damping=str(p.damping),limited='false')
 ET.SubElement(moving,'geom',name='equivalent_modal_mass',type='sphere',size='.008',mass='0',rgba='.1 .5 1 .9',contype='0',conaffinity='0',group='2')

def build_model(folder,p=ModalParameters(),mode='liquid',dt=.002,mount_offset=(0,0,0)):
 folder=Path(folder);original=folder/'tank_factory.xml';make_model(original,'empty',{},dt);tree=ET.parse(original);tank=tree.getroot().find(".//body[@name='water_tank']")
 pos=np.fromstring(tank.get('pos'),sep=' ')+mount_offset;tank.set('pos',' '.join(map(str,pos)));fluid_elements(tank,p,mode)
 # Keep tank wall collisions as factory baseline, modal mass is a visualization of generalized coordinates only.
 path=folder/'coupled.xml';ET.indent(tree);tree.write(path);model=mujoco.MjModel.from_xml_path(str(path));model.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_ENERGY)
 return model,path

def reaction_wrench(model,data,fluid_ids,origin):
 """Fluid-on-tank wrench [F_W,tau_W] from independent rigid-body momentum balance.

 No load is applied again. Returned diagnostic includes liquid gravity once.
 """
 F=np.zeros(3);T=np.zeros(3)
 mujoco.mj_rnePostConstraint(model,data)
 for body in fluid_ids:
  six=np.zeros(6);vel=np.zeros(6)
  mujoco.mj_objectAcceleration(model,data,mujoco.mjtObj.mjOBJ_BODY,body,six,0)
  mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,vel,0)
  # rnePostConstraint initializes world cacc=-gravity; this API returns a_COM-g.
  R=data.ximat[body].reshape(3,3);I=R@np.diag(model.body_inertia[body])@R.T
  f=-model.body_mass[body]*six[3:];t=-(I@six[:3]+np.cross(vel[:3],I@vel[:3]));F+=f;T+=np.cross(data.xipos[body]-origin,f)+t
 return np.r_[F,T]
