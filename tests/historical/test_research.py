import unittest,json
import numpy as np,mujoco
from ballbot_sloshing_research.envs import BallBotSloshingEnv
from ballbot_sloshing_research.fluid.reduced_order import ModalParameters
from ballbot_sloshing_research.fluid.validation import standalone
from ballbot_sloshing_research.coupling.model import reaction_wrench
from ballbot_sloshing_research.common import ROOT

class ResearchTests(unittest.TestCase):
 def env(self,**cfg):e=BallBotSloshingEnv(**cfg);self.addCleanup(e.close);return e
 def test_mass_single_moving_body_two_coordinates(self):
  a=self.env(mode='empty');b=self.env(mode='rigid');c=self.env(mode='liquid');p=c.pfluid
  self.assertAlmostEqual(c.model.body_mass.sum()-a.model.body_mass.sum(),p.mass,places=12);self.assertAlmostEqual(b.model.body_mass.sum(),c.model.body_mass.sum(),places=12);self.assertEqual(len(c.slosh_joints),2);self.assertEqual(len(c.fluid_ids),2)
 def test_modal_mass_frequency_and_fill_domain(self):
  for volume in [.0003,.0005,.0007]:
   p=ModalParameters(volume=volume);self.assertGreater(p.fixed,0);self.assertGreater(p.moving,0);self.assertAlmostEqual(p.fixed+p.moving,p.mass);self.assertGreater(p.omega,0);self.assertGreater(p.damping,0)
 def test_invalid_fluid_parameters_rejected(self):
  for cfg in [dict(volume=-1),dict(radius=0),dict(zeta=-.01),dict(frequency_scale=float('nan'))]:
   with self.assertRaises(ValueError):ModalParameters(**cfg)
 def test_rigid_and_liquid_reference_COM_match(self):
  a=self.env(mode='rigid',initial_slosh=0);b=self.env(mode='liquid',initial_slosh=0)
  for e in [a,b]:e.reset(seed=210200,options={'tilt':[0,0],'zero_velocity':True})
  ca=sum(a.model.body_mass[i]*a.data.xipos[i] for i in a.fluid_ids)/a.pfluid.mass;cb=sum(b.model.body_mass[i]*b.data.xipos[i] for i in b.fluid_ids)/b.pfluid.mass;np.testing.assert_allclose(ca,cb,atol=1e-12)
 def test_static_tilt_equilibrium_and_gravity_once(self):
  p=ModalParameters();m,d=standalone(p,.002,tilt=.02);joint=m.joint('slosh_x');R=d.xmat[m.body('tank').id].reshape(3,3);gravity_tangent=(R.T@m.opt.gravity)[0];d.qpos[joint.qposadr[0]]=gravity_tangent/p.omega**2;mujoco.mj_forward(m,d);self.assertLess(abs(d.qacc[joint.dofadr[0]]),1e-10);ids=[m.body(n).id for n in ['fluid_fixed','fluid_modal']];W=reaction_wrench(m,d,ids,d.xpos[m.body('tank').id]);np.testing.assert_allclose(W[:3],[0,0,-p.mass*9.81],atol=1e-10)
 def test_observation_has_no_instantaneous_hidden_fluid_leak(self):
  e=self.env();e.reset(seed=210200);a=e.frame().copy();e.data.qpos[e.slosh_q]=[.002,-.001];e.data.qvel[e.slosh_v]=[.03,-.02];mujoco.mj_forward(e.model,e.data);np.testing.assert_array_equal(a,e.frame())
 def test_O2_history_shapes_and_causal_reset(self):
  e=self.env(observation='O2');o,_=e.reset(seed=210200);self.assertEqual(o.shape,(184,));np.testing.assert_array_equal(o[:23],o[-23:]);e.step(np.zeros(3));self.assertEqual(e.observation().shape,(184,))
 def test_motor_action_sign_and_bounds(self):
  e=self.env(residual=False);e.reset(seed=210200);e.step(np.array([1.,-1,0.]));np.testing.assert_allclose(e.data.ctrl[e.motor_ids],[-10,10,0],atol=1e-12)
 def test_ROM_domain_honest_failure_not_clamp(self):
  e=self.env();e.reset(seed=210200,options={'slosh_q':[.02,0]});_,_,term,trunc,info=e.step(np.zeros(3));self.assertTrue(term);self.assertFalse(trunc);self.assertFalse(info['success']);self.assertEqual(info['termination_reason'],'ROM_DOMAIN');self.assertGreater(e.fluid_state()[0][0],.0084)
 def test_reset_replay_exact(self):
  e=self.env(tilt=.005);states=[]
  for _ in range(2):
   e.reset(seed=210201)
   for k in range(10):e.step(np.zeros(3))
   states.append(np.r_[e.data.qpos,e.data.qvel])
  np.testing.assert_array_equal(*states)
 def test_native_passive_joint_equations(self):
  e=self.env();e.reset(seed=210200);q,v=e.fluid_state();expected=-e.pfluid.stiffness*q-e.pfluid.damping*v;np.testing.assert_allclose(e.data.qfrc_passive[e.slosh_v],expected,atol=1e-10)
 def test_published_internal_validation_artifacts(self):
  s=json.loads((ROOT/'fluid_validation_stage_work/summary.json').read_text());self.assertTrue(s['gate1']);self.assertTrue(all(s['checks'].values()));c=json.loads((ROOT/'coupling_validation.json').read_text());self.assertTrue(c['gate2']);self.assertTrue(c['gate3'])
if __name__=='__main__':unittest.main()
