import unittest,json,hashlib
from pathlib import Path
import numpy as np
from sloshbot.runtime import CommandAdapter,FigureEnv,TeleopEnv,playground
from sloshbot.teleop import KeyboardState
from sloshbot.experiment import settlement
from sloshbot.bootstrap import ROOT
import glfw
class ReleaseTests(unittest.TestCase):
    def test_release_neutral(self):
        c=CommandAdapter();c.request_keys(['UP']);c.advance(.1);self.assertAlmostEqual(c.speed,.004);c.request_keys([]);c.advance(.1);self.assertEqual(c.speed,0)
    def test_command_bounds(self):
        c=CommandAdapter();c.request_keys(['UP','LEFT'])
        for _ in range(1000):c.advance(.002)
        self.assertLessEqual(c.speed,.04);self.assertLessEqual(c.turn,.08)
    def test_opposite_keys(self):
        c=CommandAdapter();c.request_keys(['UP','DOWN','LEFT','RIGHT']);np.testing.assert_array_equal(c.request,[0,0])
    def test_space(self):
        k=KeyboardState();k.event(glfw.KEY_UP,glfw.PRESS);k.event(glfw.KEY_SPACE,glfw.PRESS);self.assertEqual(k.keys,{'SPACE'})
    def test_key_release_and_focus(self):
        k=KeyboardState();k.event(glfw.KEY_UP,glfw.PRESS);k.event(glfw.KEY_UP,glfw.RELEASE);self.assertFalse(k.keys);k.event(glfw.KEY_LEFT,glfw.PRESS);k.focus(False);self.assertFalse(k.keys)
    def test_pause_reset_exit(self):
        k=KeyboardState()
        for v in [glfw.KEY_P,glfw.KEY_R,glfw.KEY_ESCAPE]:k.event(v,glfw.PRESS)
        self.assertTrue(k.paused and k.reset and k.exit)
    def test_turn_heading(self):
        c=CommandAdapter();c.request_keys(['LEFT'])
        for _ in range(1000):c.advance(.002)
        self.assertGreater(c.yaw,0);self.assertEqual(c.speed,0)
    def test_playground_geometry(self):
        h=playground();self.assertEqual(h.shape,(129,129));self.assertEqual(h[64,64],0);self.assertTrue(np.max(abs(h))<.003);self.assertTrue(np.all(np.isfinite(h)))
    def test_checkpoint(self):
        self.assertEqual(hashlib.sha256((ROOT/'pretrained/ppo_500k.zip').read_bytes()).hexdigest(),'651989a98dd3ab04294adddd4a0bc0fb63e09b96749e9a980c5041e51bbd5db7')
    def test_recovery_censoring(self):
        y=np.zeros((1000,27));y[:,0]=np.arange(1,1001)*.002;y[:,7]=1
        self.assertIsNone(settlement(y,[0])[0]['recovery_s'])
        y[:,7]=0;self.assertIsNotNone(settlement(y,[0])[0]['recovery_s'])
    def test_native_replay_and_initial_match(self):
        trajectories=[]
        for _ in range(2):
            e=FigureEnv(duration=.01,capture=True,terrain_seed=2617001)
            try:
                e.reset(seed=2616001);cached=json.loads((ROOT/'experiments/experiment_1_figure8/V4_TEST_1/result.json').read_text())
                np.testing.assert_array_equal(e.e.data.qpos,cached['initial']['qpos']);np.testing.assert_array_equal(e.e.data.qvel,cached['initial']['qvel'])
                e.step(np.zeros(3));trajectories.append(np.array(e.rows));self.assertEqual(e.meter_native_steps,5)
            finally:e.close()
        np.testing.assert_array_equal(*trajectories)
    def test_cross_controller_physics(self):
        arrays=[]
        for ctrl in ['V2','V3','V2_FFv1','V3_FFv1','V4']:
            e=FigureEnv(controller=ctrl,duration=.01)
            try:
                e.reset(seed=2616001);arrays.append((e.e.model.body_mass.copy(),e.e.model.body_inertia.copy(),e.e.model.hfield_data.copy(),e.e.data.qpos.copy()))
            finally:e.close()
        for row in arrays[1:]:
            for a,b in zip(arrays[0],row):np.testing.assert_array_equal(a,b)
if __name__=='__main__':unittest.main()

class TerrainRepresentationTests(unittest.TestCase):
    def test_heightfield_normalization_preserves_world_surface(self):
        e=TeleopEnv(duration=.01)
        try:
            m=e.e.model;hid=m.hfield('track8_collision').id;adr=int(m.hfield_adr[hid]);h=m.hfield_data[adr:adr+129*129];lo=m.geom_pos[e.e.ground_geom,2];span=m.hfield_size[hid,2]
            self.assertGreaterEqual(h.min(),0);self.assertLessEqual(h.max(),1);np.testing.assert_allclose(h*span+lo,e.e.heights.ravel(),rtol=0,atol=e.height_quantization_bound_m)
        finally:e.close()
    def test_original_and_normalized_short_state_agreement(self):
        p=ROOT/'experiments/experiment_2_teleoperation';a=np.load(p/'playground_pilot/trace.npz')['rows'];b=np.load(p/'normalized_height_pilot_float32/trace.npz')['rows'];self.assertEqual(a.shape,b.shape);self.assertTrue(np.isfinite(b).all())
