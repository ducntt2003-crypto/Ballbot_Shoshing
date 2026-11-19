import json,unittest
from pathlib import Path
import numpy as np,mujoco,torch
from stable_baselines3 import PPO
from ballbot_fast_rl.env import BallBotFastBalanceEnv
from ballbot_fast_rl.contracts import verify
from ballbot_fast_rl.contact_validation.diagnostics import geometry,contact_sample,solver_residuals

R=Path('ballbot_fast_rl/results/contact_validation_20261029')
OLD=Path('ballbot_fast_rl/results/20261029_stage1')
class ContactTests(unittest.TestCase):
 def test_capsule_sphere_distance_matches_rotated_native_contacts(self):
  e=BallBotFastBalanceEnv();e.reset(seed=17);g=geometry(e);rows,_=contact_sample(e)
  for w in g['wheels']:
   self.assertEqual(w['geom_type'],int(mujoco.mjtGeom.mjGEOM_CAPSULE));native=next(x for x in rows if x['pair']==f"wheel_{w['wheel']}");self.assertAlmostEqual(w['analytic_signed_gap_m'],native['gap_m'],places=12);self.assertAlmostEqual(w['radius_m'],.025);self.assertGreater(abs(w['analytic_signed_gap_m']),.01)
  e.close()
 def test_force_frames_virtual_work_and_anisotropic_axes(self):
  e=BallBotFastBalanceEnv();e.reset(seed=17);e.step([.01,-.01,0]);rows,s=contact_sample(e);self.assertLess(s['contact_generalized_force_max_abs_gap_native_units'],1e-10);self.assertLess(abs(s['contact_power_gap_W']),1e-10)
  for row in rows:
   self.assertLessEqual(row['friction_ellipse_utilization'],1+1e-8)
   if row['pair'].startswith('wheel_'):self.assertAlmostEqual(row['mu1'],.001);self.assertAlmostEqual(row['mu2'],1.);self.assertGreater(row['low_axis_alignment'],.999999)
  e.close()
 def test_diagnostic_calls_do_not_mutate_state_or_warmstart(self):
  e=BallBotFastBalanceEnv();e.reset(seed=27);e.step([.02,0,-.01]);keys=['qpos','qvel','qacc','qacc_warmstart','ctrl'];before={k:getattr(e.data,k).copy() for k in keys};contact_sample(e);solver_residuals(e)
  for k in keys:np.testing.assert_array_equal(before[k],getattr(e.data,k))
  e.close()
 def test_positive_negative_motor_generalized_torque(self):
  e=BallBotFastBalanceEnv();vel=[]
  for sign in [-1,1]:
   e.reset(seed=12,options={'tilt':[0,0],'zero_velocity':True});e.step([sign*.01,0,0]);self.assertAlmostEqual(e.data.qfrc_actuator[e.wheel_dofs[0]],-sign*.1);vel.append(e.data.qvel[e.wheel_dofs[0]])
  self.assertLess(vel[1],vel[0]);e.close()
 def test_reset_reproducible_and_default_contact_parameters(self):
  e=BallBotFastBalanceEnv();e.reset(seed=44);a=e.data.qpos.copy();c=geometry(e);e.reset(seed=44);np.testing.assert_array_equal(a,e.data.qpos);self.assertEqual(c,geometry(e));np.testing.assert_allclose(e.model.pair_solref,[[.02,1]]*3);self.assertTrue((e.model.pair_margin==0).all());self.assertTrue((e.model.pair_gap==0).all());e.close()
 def test_no_unauthorized_correction_training_or_20s_gate(self):
  d=json.loads((R/'contact_decision.json').read_text());self.assertEqual(d['corrected_variants'],0);self.assertFalse(d['fine_tuning_authorized_by_gate']);g=json.loads((R/'nominal/longer_horizon_gate.json').read_text());self.assertFalse(g['gate10_pass']);self.assertFalse((R/'nominal/horizon_20s').exists())
 def test_frozen_policy_compatible_and_new_heldout_evaluation(self):
  verify(OLD/'train_stage2/best_model.zip',dict(env=dict(mode='original')));s=json.loads((R/'nominal/horizon_5s/summary.json').read_text());self.assertEqual(s['ppo']['episodes'],50);self.assertEqual(s['ppo']['success_rate'],1.);self.assertAlmostEqual(s['ppo']['strict_gate_success_rate'],.98)
 def test_existing_torchscript_parity_on_current_sensor_proxies(self):
  torch.set_num_threads(1);model=PPO.load(OLD/'train_stage2/best_model.zip',device='cpu');actor=torch.jit.load(str(OLD/'nominal_export/actor.pt'));scale=np.array([1]*3+[5]*3+[10]*3+[.5]*2+[1]*2+[1]*3+[1]*2,np.float32);e=BallBotFastBalanceEnv()
  for seed in range(10):
   obs,_=e.reset(seed=seed);raw=obs*scale
   with torch.no_grad():actual=actor(torch.tensor(raw[None])).numpy()[0]
   np.testing.assert_allclose(actual,model.predict(obs,deterministic=True)[0],atol=1e-6,rtol=0)
  e.close()
if __name__=='__main__':unittest.main()
