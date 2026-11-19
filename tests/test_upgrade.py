import unittest,json
import numpy as np,mujoco
from scipy.special import jvp
from sloshbot.terrain_upgrade import heights,LEVELS,traversal,TerrainEnv
from sloshbot.surface import elevation,mesh,draw_surface,ALPHA
from sloshbot.motion_replay import ContainerReplay
from sloshbot.bootstrap import ROOT
from ballbot_sloshing_research.fluid.reduced_order import ModalParameters
class UpgradeTests(unittest.TestCase):
    def test_actual_translation_required(self):
        x=np.zeros((100,7));x[:,0]=np.linspace(0,26,100);self.assertFalse(traversal(x,'BUMPS',True,True)['passed'])
    def test_approach_exit_order(self):
        x=np.zeros((100,7));x[:,0]=np.linspace(0,26,100);x[:,1]=np.linspace(0,.72,100);self.assertTrue(traversal(x,'BUMPS',True,True)['passed']);self.assertFalse(traversal(x[::-1],'BUMPS',True,True)['passed']);self.assertFalse(traversal(x,'BUMPS',False,True)['passed'])
    def test_terrain_normalized_and_height_bound(self):
        for level in LEVELS:
            z=heights(level);self.assertEqual(z.shape,(129,129));self.assertEqual(z[64,64],0);self.assertTrue(np.isfinite(z).all());self.assertLessEqual(np.max(abs(z)),.002+1e-15)
    def test_bessel_wall_no_radial_flux(self):self.assertLess(abs(jvp(1,ALPHA)),1e-14)
    def test_surface_mean_and_first_moment(self):
        p=ModalParameters();q=np.array([.003,-.002]);u,w=np.polynomial.legendre.leggauss(40);r=(u+1)*p.radius/2;wt=w*p.radius/2;theta=np.arange(128)*2*np.pi/128;x=r[:,None]*np.cos(theta);y=r[:,None]*np.sin(theta);eta=elevation(x,y,q,p);area=wt[:,None]*r[:,None]*(2*np.pi/128);self.assertLess(abs(np.sum(eta*area)),1e-17);com=np.array([np.sum(x*eta*area),np.sum(y*eta*area)])/p.volume;np.testing.assert_allclose(com,p.moving/p.mass*q,atol=1e-14)
    def test_surface_no_arbitrary_wave(self):
        p=ModalParameters();v,f=mesh([0,0],p);np.testing.assert_array_equal(v[:,2],np.full(len(v),p.depth));a,_=mesh([.003,0],p);b,_=mesh([-.003,0],p);np.testing.assert_allclose(a[:,2]+b[:,2],2*p.depth,atol=1e-16);self.assertLess(a[:,2].max(),.16)
    def test_renderer_does_not_mutate_physics(self):
        e=TerrainEnv(level='BUMPS',duration=.01)
        try:
            e.reset(seed=2618001);m=e.e.model;d=e.e.data;fields=['qpos','qvel','qacc','qacc_warmstart','ctrl','xfrc_applied'];before={k:getattr(d,k).copy() for k in fields};mass=m.body_mass.copy();scene=mujoco.MjvScene(m,maxgeom=1000);info=draw_surface(scene,m,d)
            for k in fields:np.testing.assert_array_equal(before[k],getattr(d,k))
            np.testing.assert_array_equal(mass,m.body_mass);self.assertTrue(info['visual_only']);self.assertGreater(scene.ngeom,100)
        finally:e.close()
    def test_motion_replay_node_and_rotation_derivative(self):
        from scipy.spatial.transform import Rotation
        t=np.linspace(0,1,21);R=Rotation.from_rotvec(np.c_[.1*t,.2*t*t,.03*t]).as_matrix();x=np.c_[t,t*t,np.zeros((21,2)),R.reshape(21,9),np.zeros((21,3)),2*t,np.zeros((21,2))];p=ContainerReplay(x)
        for i in [0,5,20]:np.testing.assert_allclose(p.at(t[i])['position_W_m'],x[i,1:4],atol=1e-12)
        h=1e-5;v=p.at(.5);omega=Rotation.from_matrix(p.at(.5+h)['R_WT']@p.at(.5-h)['R_WT'].T).as_rotvec()/(2*h);np.testing.assert_allclose(v['omega_W_rad_s'],omega,atol=1e-7)
    def test_new_native_sph_outputs(self):
        r=json.loads((ROOT/'research/high_fidelity/screen_summary.json').read_text());self.assertEqual(r['native_steps'],1600);self.assertEqual(r['physical_liquid_status'],'SPH_BLOCKED');self.assertFalse(r['records'][1]['screen_pass']);self.assertEqual(r['records'][0]['escaped_max'],0)

    def test_camera_world_and_tank_transforms(self):
        from sloshbot.cameras import configure_camera
        e=TerrainEnv(level='FLAT',duration=.01)
        try:
            e.reset(seed=2618001);cam=mujoco.MjvCamera();d=e.e.data;m=e.e.model;before=d.qpos.copy();configure_camera(cam,m,d,1);np.testing.assert_array_equal(cam.lookat,d.xpos[e.e.base]);configure_camera(cam,m,d,5);np.testing.assert_allclose(cam.lookat,d.xpos[e.e.tank_id]+[0,0,.035]);np.testing.assert_array_equal(before,d.qpos)
        finally:e.close()
