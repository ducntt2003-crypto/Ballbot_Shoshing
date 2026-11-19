import unittest,os,numpy as np,mujoco
from ballbot_sloshing_research.large_figure8_rl.common import *
from ballbot_sloshing_research.large_figure8_rl.reference import reference,nodes
from ballbot_sloshing_research.large_figure8_rl.resources import Process
from ballbot_sloshing_research.large_figure8_rl.env import classify
from ballbot_sloshing_research.track8.reference import reference as old
class Benchmark(unittest.TestCase):
 def test_derivatives(self):
  h=1e-4
  for t in np.linspace(0,30,101):
   p,v,a,j=reference(t);pm,vm,*_=reference(t-h);pp,vp,*_=reference(t+h);np.testing.assert_allclose((pp-pm)/(2*h),v,atol=1e-9);np.testing.assert_allclose((vp-vm)/(2*h),a,atol=1e-9)
 def test_reference_scale_closure(self):
  for t in np.linspace(0,30,99):np.testing.assert_allclose(reference(t),np.array(old(t))*1.5,atol=1e-15)
  np.testing.assert_allclose(reference(0)[0],reference(30)[0],atol=1e-15)
 def test_nodes(self):
  ts,pts=nodes();self.assertEqual(len(ts),16);self.assertEqual(ts[-1],30);self.assertAlmostEqual(max(np.linalg.norm(reference(t)[0]) for t in np.linspace(0,30,30001)),.75);np.testing.assert_allclose(pts[3],[.75,0],atol=1e-15)
 def test_resources(self):
  p=Process();self.assertGreater(p.memory_info().rss,0);self.assertGreaterEqual(p.cpu_times().user,0);self.assertGreaterEqual(p.cpu_times().system,0)
 def test_guards(self):
  self.assertEqual(classify(.00841,0,0,0),'ROM_INVALID');self.assertEqual(classify(0,.121,0,0),'BALANCE_INVALID');self.assertEqual(classify(0,0,.251,0),'TASK_TRACKING_INVALID');self.assertEqual(classify(0,0,0,10.01),'ACTUATOR_INVALID');self.assertEqual(classify(0,0,0,0,contact=True),'CONTACT_INVALID');self.assertIsNone(classify(.0084,.12,.25,10))
 def test_physics_frozen(self):
  a=mujoco.MjModel.from_xml_path(str(ROOT/'evaluations/V4V2_DEV_1_T30/model.xml'));b=mujoco.MjModel.from_xml_path(str(ROOT/'evaluations/V4_DEV_1_T30/model.xml'))
  for name in ['body_mass','body_inertia','body_ipos','body_iquat','jnt_stiffness','dof_damping','geom_friction','pair_friction','actuator_ctrlrange','actuator_gear']:np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
 def test_metric_fulltask_requires_nodes(self):
  from ballbot_sloshing_research.track8.evaluate import metrics
  x=np.load(ROOT/'evaluations/V4V2_DEV_1_T30/trace.npz')['rows'];m=metrics(x,30,'figure8',.75,.375,30,'TIME_LIMIT','liquid');self.assertTrue(m['survival']);self.assertFalse(m['completion']);self.assertEqual(m['ordered_nodes_passed'],15)
 def test_checkpoint_density_reload_receipt(self):
  r=read(ROOT/'initial_policy/contract.json');self.assertEqual(r['reload_gap'],0);self.assertLess(r['logprob_gap'],1e-6);self.assertEqual(r['trained_transitions'],0)
 def test_parallel_accounting(self):
  v=read(ROOT/'vector_validation.json')
  for r in v['records']:
   self.assertEqual(r['timed_native_steps'],r['timed_transitions']*5);self.assertEqual(len(set(r['pids'])),r['workers']);self.assertTrue(all(s['bitwise_state_match'] for s in r['serial_equivalence']))
 def test_protocol_unchanged(self):
  for n,h in read(ROOT/'protocol_registration_v2.json').items():self.assertEqual(sha(ROOT/n),h)
if __name__=='__main__':unittest.main()
