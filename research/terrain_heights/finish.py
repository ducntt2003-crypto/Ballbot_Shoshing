from pathlib import Path
import json,hashlib,sys,csv,numpy as np
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text());A=json.loads((R/'analysis.json').read_text());G=json.loads((R/'geometry.json').read_text());SOURCE=R.parents[2]/P['source_release']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
text='''# Địa hình nhấp nhô cao hơn — 19/11/2026

Đúng: roughness 2 mm trước đây rất nhẹ, nhất là với contact mềm đã được audit có độ xuyên khoảng 5–6 mm. Đợt này tạo **collision heightfield mới** với gò danh nghĩa **5, 10 và 20 mm**, cùng một sân MIXED chứa cả ba độ cao. Không phóng đại chiều cao bằng camera hoặc sửa mass, inertia, damping, torque limits hay controller gains. Dữ liệu cũ được giữ nguyên.

## Xem ngay

- [GIF sân trộn 5–10–20 mm](media/mixed_5_10_20mm.gif)
- [GIF đối chiếu bốn sân và hai controller](media/height_comparison.gif)
- [MP4](media/height_comparison.mp4)
- [Bản đồ độ cao thật](figures/01_height_maps.png)
- [Profile địa hình dọc quỹ đạo](figures/02_reference_height_profile.png)
- [Tracking/energy theo độ cao](figures/03_performance_by_height.png)
- [Độ cao robot thực sự gặp dưới ball](figures/04_actual_height_encountered.png)

## Hình học đã thay đổi

Sân 4 × 4 m, grid 129 × 129. Đây là độ cao gò so với nền phẳng, không phải peak-to-peak noise hoặc bậc thẳng đứng. Các gò compact C2 có đoạn lên và xuống liên tục. Ba sân 5/10/20 dùng cùng footprint, bán kính gò 0,60 m; chỉ scale độ cao. Bán kính này giữ slope tối đa khoảng 3,40°, dưới screening 0,06 rad, trong khi tilt guard cũ vẫn 0,12 rad. Vùng xuất phát bán kính 0,12 m phẳng. MIXED có các gò 5/10/20 mm ở những vị trí riêng; nó không phải điểm tiếp theo của cùng chuỗi scale.

| Sân | Đỉnh collision thực | Slope cực đại |
|---|---:|---:|
'''
for c in P['conditions']:text+=f"| {c} | {G[c]['max_height_m']*1000:.2f} mm | {G[c]['max_slope_deg']:.2f}° |\n"
text+='''
## Kết quả mô phỏng mới

Mỗi sân có pilot V4 3 s trước khi mở ba cặp seed V4/PPO, mỗi lượt 30 s. Chỉ mở mức kế tiếp khi cả hai controller có ít nhất 2/3 lượt hoàn thành trong hard physical bounds ở mức trước. Toàn bộ bốn mức đã qua gate để được thử; không có retuning sau khi xem kết quả.

| Sân | V4 đủ 30 s | PPO đủ 30 s | RMSE V4 (mm) | RMSE PPO (mm) | Windows V4/PPO |
|---|---:|---:|---:|---:|---|
'''
for c in P['conditions']:
 v=A['summary'][c]['V4'];q=A['summary'][c]['PPO'];text+=f"| {c} | {v['physical30s']}/3 | {q['physical30s']}/3 | {v['mean_full30_RMSE_m']*1000:.2f} | {q['mean_full30_RMSE_m']*1000:.2f} | {v['windows']} / {q['windows']} |\n"
text+='''
Cả 24 lượt chính đều đủ thời gian và không báo vi phạm tilt/ROM/contact/torque guard; **không lượt nào đạt toàn bộ task figure-eight gốc**. Ở 20 mm, RMSE trung bình khoảng 114 mm đã vượt giới hạn 100 mm, dù robot vẫn giữ physical bounds và đi hết thời gian. Vì vậy không gọi đó là hoàn thành tracking thành công.

Độ cao mặt đất thực sự gặp dưới sphere đạt gần 19,94 mm; độ nâng ball centre sau khi tách contact settlement đạt gần 19,7 mm. Có đoạn đi lên và hạ xuống thật, không lấy quay ball tại chỗ làm bằng chứng. Các số liệu từng lượt, excursion, path length, modal state và torque ở [comparison.csv](comparison.csv).

Năng lượng sloshing không tăng đơn điệu theo chiều cao vì chuyển động robot và closed-loop response thay đổi; không suy diễn từ ảnh rằng nước dao động mạnh hơn hoặc yếu hơn. PPO frozen gần V4; đợt này không có training/adaptation mới cho terrain cao.

## Mở cửa sổ 3D để quan sát

Đây là **replay từ state native đã ghi**, có camera tương tác, không phải một lượt tính physics mới:

```bash
/home/duc/OpenBallBot-RL/.venv/bin/python -B /home/duc/OpenBallBot-RL/ballbot_sloshing_research/results/terrain_heights_20261119_v1/replay_viewer.py --condition MIXED --controller V4
```

Dùng Space để pause, R để chạy lại, 1–5 để đổi camera, Esc để thoát. Có thể chọn H05/H10/H20/MIXED và V4/PPO. Viewer đã qua smoke test mở cửa sổ; không có human-driving study.

Để **tính mô phỏng mới** ở một thư mục output mới, thay vì replay:

```bash
/home/duc/OpenBallBot-RL/.venv/bin/python -B /home/duc/OpenBallBot-RL/ballbot_sloshing_research/results/terrain_heights_20261119_v1/experiment.py H20 V4 2611801 30 /tmp/sloshbot_new_H20_run
```

Giữ nguyên output cũ; không chạy lại batch đã hoàn thành chỉ để xem GIF.

## Giới hạn và kiểm chứng

- Gò trơn cao 20 mm không chứng minh robot vượt được bậc thẳng đứng 20 mm, địa hình cao hơn, hoặc toàn bộ sân ngoài các tuyến đã thử.
- Contact law kế thừa chưa được hiệu chuẩn hardware. Hiệu ứng compliant penetration cỡ 5–6 mm vẫn còn; không giảm stiffness/damping hay khóa state để che vấn đề.
- Chất lỏng vẫn là ROM, surface trong GIF chỉ là visualization từ modal states. SPH_BLOCKED và các hạn chế torque/moment của ROM chưa được giải quyết bằng đợt thử terrain này.
- Ba seed là evaluation contexts, không phải independent PPO training runs. Không có timestep convergence mới ở những độ cao này.
- Không so các con số mới với roughness 2 mm cũ như một ablation chỉ thay amplitude: terrain 2 mm cũ khác footprint. Chuỗi H05→H10→H20 mới mới là so sánh cùng footprint.

Có 4 pilot và 24 lượt chính, tổng **366.000 bước MuJoCo mới**; worker wall khoảng 493 s, dưới cap 850 s và 400.000 bước. Render/replay chỉ đọc dữ liệu và không tạo bước physics. Source/checkpoint V2 được đối chiếu 929 SHA-256 trước và sau; không chỉnh training hoặc hardware.
'''
with (R/'REPORT_VI.md').open('x') as f:f.write(text)
with (R/'README.md').open('x') as f:f.write('''# Higher terrain comparison — 19 November 2026

New true collision mounds: nominal5/10/20mm and a mixed-height playground. Frozen V4/PPO, three paired contexts,30s enlarged figure-eight, unchanged robot/liquid/contact/actuators.24physically complete full-duration runs and4pilots; no full-task pass. No training.

Open REPORT_VI.md, media/mixed_5_10_20mm.gif, media/height_comparison.gif and figures/01_height_maps.png. The grid shown in native videos follows actual terrain geometry; no vertical exaggeration. ROM surface rendering is not CFD. Source runtime dependency: public_releases/SloshBot-RL_20261116_v2.

replay_viewer.py provides camera-interactive playback of recorded native states, not new simulation. experiment.py computes a new native run only into a new output directory. Raw traces, terminal states, models, normalized heightfields and all per-case metrics remain under runs/. protocols and registration are frozen in this result folder.
''')
# Read-only consistency checks; no physics integration.
import mujoco,cv2
from PIL import Image
assert sha(R/'protocol.json')==json.loads((R/'registration.json').read_text())['protocol.json'];manifest=json.loads((SOURCE/'docs/artifact_manifest.json').read_text());assert all(sha(SOURCE/n)==h for n,h in manifest['files'].items())
count=0;steps=0;maxheight=0
for cond in P['conditions']:
 for seed in P['seeds']:
  base=None;m0=None
  for method in P['controllers']:
   f=R/f'runs/{cond}_{method}_{seed}';a=json.loads((f/'result.json').read_text());z=np.load(f/'trace.npz');m=mujoco.MjModel.from_xml_string((f/'model.xml').read_text().replace('__ASSET_ROOT__',str(SOURCE/'_vendor/ballbotgym/ballbotgym/assets')));initial=np.r_[a['initial']['qpos'],a['initial']['qvel']]
   if base is None:base=initial;m0=m;hf0=np.load(f/'hfield.npy')
   else:
    np.testing.assert_array_equal(base,initial);np.testing.assert_array_equal(hf0,np.load(f/'hfield.npy'))
    for n in ['body_mass','body_inertia','jnt_stiffness','dof_damping','pair_friction','geom_friction','actuator_ctrlrange']:np.testing.assert_array_equal(getattr(m0,n),getattr(m,n))
   assert a['completed'] and a['physical_valid'] and a['native_steps']==15000 and not a['metrics']['completion'];assert np.isfinite(z['native_rows']).all();assert a['contact_stats']['unsupported']==0;assert m.opt.timestep==.002
   assert np.linalg.norm(z['rows'][:,14:16],axis=1).max()<=.0084;assert abs(z['rows'][:,19:22]).max()<=10+1e-12
   hid=m.hfield('track8_collision').id;adr=int(m.hfield_adr[hid]);hf=np.load(f/'hfield.npy')[adr:adr+129*129];assert hf.min()>=0 and hf.max()<=1;np.testing.assert_array_equal(hf.astype(float).reshape(129,129)*m.hfield_size[hid,2]+m.geom_pos[m.geom('terrain').id,2],np.load(f/'world_heights.npy'))
   steps+=a['native_steps'];count+=1;maxheight=max(maxheight,float(np.load(f/'world_heights.npy').max()))
for f in (R/'runs').glob('*pilot/result.json'):a=json.loads(f.read_text());assert a['completed'] and a['physical_valid'];steps+=a['native_steps']
assert count==24 and steps==366000 and maxheight>.019
for name in ['height_comparison.gif','mixed_5_10_20mm.gif']:assert Image.open(R/'media'/name).n_frames==151
cap=cv2.VideoCapture(str(R/'media/height_comparison.mp4'));assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==151;cap.release()
with (R/'verification.json').open('x') as f:json.dump(dict(passed=True,full_duration_runs=24,pilot_runs=4,native_steps=steps,training_steps=0,source_files_unchanged=929,world_geometry_is_normalized=True,max_collision_height_m=maxheight,verification_physics_steps=0,original_fulltask_passes=0),f,indent=2)
print('REPORT_AND_VERIFY_PASS',count,steps)
