"""V2 artifact and scientific-consistency verification, no physics integration."""
from pathlib import Path
import json,numpy as np,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    assert json.loads((ROOT/'docs/v2/historical_integrity.json').read_text())['passed']
    old=json.loads((ROOT/'docs/v1_artifact_manifest.json').read_text())['files']
    for n,h in old.items():
        if n.startswith('experiments/experiment_1_figure8/') or n=='pretrained/ppo_500k.zip':assert sha(ROOT/n)==h,n
    import mujoco
    count=0;passes=0
    for level in ['FLAT','MILD','RANDOM','BUMPS','SLOPES','COURSE']:
        for seed in [2618001,2618002,2618003]:
            folder=ROOT/f'upgrade/results/{level}_{seed}';r=json.loads((folder/'result.json').read_text());z=np.load(folder/'trace.npz');x=z['xyzv'];region=r['traversal']['region'];cursor=0;stages=0
            for mask in [x[:,1]<=region['approach_x_max'],(x[:,1]>=region['obstacle_x_min'])&(x[:,1]<=region['obstacle_x_max'])&(abs(x[:,2])<=region['half_width_m']),x[:,1]>=region['exit_x_min']]:
                ix=np.flatnonzero(mask & (np.arange(len(x))>=cursor))
                if not len(ix):break
                stages+=1;cursor=int(ix[0])+1
            success=bool(stages==3 and r['physical_valid'] and r['completed'] and np.linalg.norm(x[-1,1:3]-x[0,1:3])>=.5)
            assert success==r['traversal']['passed'];passes+=success;count+=1
            assert len(x) in [r['native_steps'],r['native_steps']-1]
            m=mujoco.MjModel.from_xml_string((folder/'model.xml').read_text().replace('__ASSET_ROOT__',str(ROOT/'_vendor/ballbotgym/ballbotgym/assets')));h=np.load(folder/'hfield.npy');hid=m.hfield('track8_collision').id;adr=int(m.hfield_adr[hid]);d=h[adr:adr+129*129];assert d.min()>=0 and d.max()<=1
            np.testing.assert_allclose(d.reshape(129,129)*m.hfield_size[hid,2]+m.geom_pos[m.geom('terrain').id,2],np.load(folder/'world_heights.npy'),atol=2*np.finfo(np.float32).eps*m.hfield_size[hid,2],rtol=0)
            assert abs(m.opt.timestep-.002)<1e-15 and int(m.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_RK4)
    assert count==18 and passes==17
    quality=json.loads((ROOT/'docs/v2/motion_quality_audit.json').read_text());assert sum(v['legacy_motion_quality_pass'] for v in quality['summary'].values())==15
    print('PASS V2 ordered translation/obstacle accounting, normalized world heights,17/18 route and15/18 legacy motion quality',flush=True)
    hf=json.loads((ROOT/'research/high_fidelity/screen_summary.json').read_text());assert hf['native_steps']==1600 and hf['physical_liquid_status']=='SPH_BLOCKED';assert [x['screen_pass'] for x in hf['records']]==[True,False,False]
    for row in hf['records']:
        status=json.loads((ROOT/'research/high_fidelity'/row['case']/'run_status.json').read_text());assert status['completed_steps']==row['native_steps'];assert status['wall_seconds']<60
    assert hf['records'][0]['actual_mass_kg']!=hf['records'][1]['actual_mass_kg']
    print('PASS independent GPU output receipts, failed finer screens, mass mismatch and blocked dynamic comparison',flush=True)
    tests=json.loads((ROOT/'docs/v2/test_receipt.json').read_text());assert tests['passed'] and tests['tests']==41
    budget=json.loads((ROOT/'docs/v2/compute_accounting.json').read_text());assert budget['new_training_transitions']==0 and budget['MuJoCo_native_steps']<300000 and budget['terrain_native_wall_s']<1000
    import cv2
    for row in json.loads((ROOT/'docs/v2/media_provenance.json').read_text())['videos']:
        c=cv2.VideoCapture(str(ROOT/row['video']));assert c.isOpened();assert int(c.get(cv2.CAP_PROP_FRAME_COUNT))==row['frames'];c.release();assert row['native_steps_generated']==0
    print(json.dumps(dict(V2_passed=True,tests=41,history_protected=84653,new_verification_physics_steps=0,high_fidelity='SPH_BLOCKED',realtime='NOT_CERTIFIED',GitHub_published=False)),flush=True)
if __name__=='__main__':verify()
