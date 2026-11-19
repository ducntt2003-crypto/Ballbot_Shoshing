# Địa hình nhấp nhô cao hơn — 19/11/2026

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
| H05 | 4.99 mm | 0.85° |
| H10 | 9.97 mm | 1.70° |
| H20 | 19.95 mm | 3.40° |
| MIXED | 19.95 mm | 3.40° |

## Kết quả mô phỏng mới

Mỗi sân có pilot V4 3 s trước khi mở ba cặp seed V4/PPO, mỗi lượt 30 s. Chỉ mở mức kế tiếp khi cả hai controller có ít nhất 2/3 lượt hoàn thành trong hard physical bounds ở mức trước. Toàn bộ bốn mức đã qua gate để được thử; không có retuning sau khi xem kết quả.

| Sân | V4 đủ 30 s | PPO đủ 30 s | RMSE V4 (mm) | RMSE PPO (mm) | Windows V4/PPO |
|---|---:|---:|---:|---:|---|
| H05 | 3/3 | 3/3 | 71.80 | 71.77 | [15, 15, 15] / [15, 15, 15] |
| H10 | 3/3 | 3/3 | 81.35 | 81.32 | [14, 14, 14] / [14, 14, 14] |
| H20 | 3/3 | 3/3 | 113.87 | 113.85 | [10, 11, 11] / [10, 11, 11] |
| MIXED | 3/3 | 3/3 | 100.07 | 100.03 | [12, 12, 12] / [12, 12, 12] |

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
