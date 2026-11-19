from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2]
TANK_CONFIG=json.loads((ROOT/'chrono_fsi/configs/integration/tank.json').read_text())
R=TANK_CONFIG['inner_radius_m'];H=TANK_CONFIG['inner_height_m'];T=TANK_CONFIG['wall_bottom_thickness_m'];MOUNT=np.array(TANK_CONFIG['inner_floor_B_m'])
def tank_components(shell_scale=1.):
    ro=R+T
    side=TANK_CONFIG['polymer_density']*np.pi*(ro*ro-R*R)*H*shell_scale
    bottom=TANK_CONFIG['polymer_density']*np.pi*ro*ro*T*shell_scale
    post=2700*np.pi*.003**2*.023*shell_scale
    components=[(side,np.array([0,0,H/2]),np.diag([side*(3*(ro*ro+R*R)+H*H)/12]*2+[side*(ro*ro+R*R)/2])),
                (bottom,np.array([0,0,-T/2]),np.diag([bottom*(3*ro*ro+T*T)/12]*2+[bottom*ro*ro/2]))]
    for angle in np.arange(4)*np.pi/2:
        components.append((post,np.array([.05*np.cos(angle),.05*np.sin(angle),-.0135]),np.diag([post*(3*.003**2+.023**2)/12]*2+[post*.003**2/2])))
    mass=sum(c[0] for c in components);com=sum(m*p for m,p,_ in components)/mass
    inertia=sum(I+m*((p-com)@(p-com)*np.eye(3)-np.outer(p-com,p-com)) for m,p,I in components)
    return mass,com,inertia

def make_model(path,mode='sph',params=None,physics_dt=.00005):
    params=params or {};tree=ET.parse(ROOT/json.loads((ROOT/'chrono_fsi/configs/integration/robot.json').read_text())['original_xml']);xml=tree.getroot()
    comp=xml.find('compiler');comp.set('meshdir',str(ROOT/'ballbotgym/ballbotgym/assets/stl_files'));comp.set('texturedir',str(ROOT/'ballbotgym/ballbotgym/assets/textures'))
    for e in xml.findall('option'):
        if e.get('timestep'):e.set('timestep',str(physics_dt))
    base=xml.find(".//body[@name='base']")
    # Coherent mass/inertia through density and ballast location; no independent tensor-entry noise.
    for geom in base.iter('geom'):
        rho=float(geom.get('density','1000'));geom.set('density',str(rho*params.get('robot_mass_scale',1.)))
    ballast=base.find("geom[@name='ballast']");offset=params.get('ballast_offset_m',[0,0,0]);ballast.set('pos',' '.join(map(str,np.array([0,0,.002])+offset)))
    ground=xml.find(".//geom[@name='terrain']");ground.attrib.clear();ground.attrib.update(name='terrain',type='plane',size='5 5 .1',friction=f"{params.get('ground_friction',1.)} .005 .0001",rgba='.15 .18 .2 1')
    for pair in xml.findall('.//pair'):
        slip=params.get('wheel_friction_scale',1.);pair.set('friction',f'{.001*slip} {1*slip}')
    for actuator in xml.findall('.//motor'):
        lim=10*params.get('motor_torque_scale',1.);actuator.set('ctrlrange',f'{-lim} {lim}')
    for bodyname in ['base','ball']:
        b=xml.find(f".//body[@name='{bodyname}']");pos=np.fromstring(b.get('pos'),sep=' ');pos[2]-=.0299;b.set('pos',' '.join(map(str,pos)))
    mount=MOUNT+params.get('mount_offset_m',np.zeros(3));mass,com,I=tank_components(params.get('shell_mass_scale',1.))
    if mode!='original':
        tank=ET.SubElement(base,'body',name='water_tank',pos=' '.join(map(str,mount)),euler=' '.join(map(str,params.get('mount_orientation_deg',[0,0,0]))))
        def inertial(body,m,c,tensor):
            ET.SubElement(body,'inertial',mass=str(m),pos=' '.join(map(str,c)),fullinertia=' '.join(map(str,[tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]])))
        inertial(tank,mass,com,I)
        ET.SubElement(tank,'geom',name='tank_bottom',type='cylinder',size=f'{R+T} {T/2}',pos=f'0 0 {-T/2}',rgba='.25 .65 .85 .7')
        for i,angle in enumerate(np.arange(24)*2*np.pi/24):
            r=R+T/2;ET.SubElement(tank,'geom',name=f'tank_side_{i}',type='box',size=f'{T/2} {r*np.tan(np.pi/24)} {H/2}',pos=f'{r*np.cos(angle)} {r*np.sin(angle)} {H/2}',euler=f'0 0 {np.degrees(angle)}',rgba='.25 .65 .85 .25')
        for i,angle in enumerate(np.arange(4)*np.pi/2):
            ET.SubElement(tank,'geom',name=f'tank_support_{i}',type='cylinder',size='.003 .0115',pos=f'{.05*np.cos(angle)} {.05*np.sin(angle)} -.0135',rgba='.5 .5 .5 1')
        if mode=='rigid':
            vol=params.get('water_volume_m3',.0005);rho=params.get('water_density',998.);depth=vol/(np.pi*R**2);nz=int(np.floor(depth/.004));grid=np.arange(-int(np.ceil(R/.004)),int(np.ceil(R/.004))+1)*.004
            pts=np.array([[x,y,(k+.5)*.004] for k in range(nz) for x in grid for y in grid if x*x+y*y<R*R]);mp=6.3820334353089766e-5*rho/998.;mw=mp*len(pts);center=pts.mean(axis=0);r=pts-center;Iw=mp*(np.sum(r*r)*np.eye(3)-r.T@r);depth=nz*.004
            # Randomized FAST payload is a continuum rigid cylinder, so sampled V -> exact mass/COM/inertia.
            # Nominal control remains matched to actual discrete SPH mass and moments.
            if 'water_volume_m3' in params:
                depth=vol/(np.pi*R**2);mw=rho*vol;center=np.array([0,0,depth/2]);Iw=np.diag([mw*(3*R*R+depth*depth)/12]*2+[mw*R*R/2])
            payload=ET.SubElement(tank,'body',name='rigid_water_control');inertial(payload,mw,center,Iw)
            ET.SubElement(payload,'geom',type='cylinder',size=f'{R} {depth/2}',pos=f'0 0 {depth/2}',rgba='.1 .4 .9 .5',contype='0',conaffinity='0')
    ET.indent(tree);tree.write(path)
    model=mujoco.MjModel.from_xml_path(str(path));model.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_ENERGY);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    allmass=float(model.body_mass.sum());comall=(model.body_mass[:,None]*data.xipos).sum(axis=0)/allmass
    totalI=np.zeros((3,3))
    for b in range(1,model.nbody):
        rot=data.ximat[b].reshape(3,3);r=data.xipos[b]-comall;totalI+=rot@np.diag(model.body_inertia[b])@rot.T+model.body_mass[b]*(r@r*np.eye(3)-np.outer(r,r))
    result={'label':'ASSUMED_GEOMETRY_NON_VALIDATED','mode':mode,'rigid_payload_definition':'continuum_volume' if 'water_volume_m3' in params else 'matched_discrete_SPH','dry_robot_and_tank_mass_kg':allmass,'tank_shell_and_mount_mass_kg':mass,'tank_COM_T_m':com.tolist(),'tank_inertia_at_COM_kg_m2':I.tolist(),'mount_position_B_m':mount.tolist(),'locked_aggregate_COM_W_m':comall.tolist(),'locked_aggregate_inertia_W_kg_m2':totalI.tolist(),'note':'open polymer shell + four aluminium posts, no hypothetical lid/allowance. Rigid mode is a control case only; SPH mode has no fluid mass in MuJoCo.'}
    return model,result
