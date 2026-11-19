# SloshBot-RL V2 — Báo cáo bàn giao, 16/11/2026

Đã hoàn thành bản nâng cấp riêng, gồm terrain có collision, ứng dụng bàn phím, trail trực tiếp trong MuJoCo, mặt nước từ trạng thái ROM thật, dữ liệu SPH độc lập và bản thảo tiếng Anh. **Chưa đạt physical liquid validation; chưa publish GitHub; không có training mới.** V1 và lịch sử nghiên cứu được giữ nguyên.

## Kết quả terrain

Mỗi điều kiện có ba reset seeds và cùng script lệnh 26 giây. Route-pass yêu cầu tiếp cận, đi vào vùng chướng ngại theo thứ tự, vượt exit, dịch chuyển XY ít nhất 0,5 m, giữ physical guards và hoàn thành tuyến. Đây là tiêu chí đã đăng ký, khác với việc chỉ sống đủ thời gian.

| Terrain | Route-pass | Đồng thời đạt quality velocity/yaw của V1 |
|---|---:|---:|
| FLAT | 3/3 | 3/3 |
| MILD | 3/3 | 3/3 |
| RANDOM | 3/3 | 3/3 |
| BUMPS | 3/3 | 2/3 |
| SLOPES | 3/3 | 3/3 |
| COURSE | 2/3 | 1/3 |

Tổng cộng **17/18 route-pass**, nhưng chỉ **15/18** đạt thêm ngưỡng quality vận tốc/yaw cũ. COURSE seed 2618001 chạm ROM guard khoảng 19,570 s. Không thay gain hoặc bỏ kết quả lỗi để cải thiện tỷ lệ.

Các giới hạn quan trọng:

- Bump/plateau chỉ cao khoảng 1,8 mm, COURSE 1,2 mm; chưa thử chướng ngại lớn. Slope cực đại của các terrain dưới 0,03 rad.
- Ball thực sự nâng lên khoảng 1,80–1,83 mm tại bump/plateau rồi hạ lại. Tuy nhiên, contact compliant hiện tại xuyên khoảng 5,4–5,6 mm, lớn hơn chiều cao feature. Kết quả chưa chứng minh khả năng vượt vật cứng hoặc terrain ngoài thực.
- Torque chạm ±10 Nm; tỷ lệ samples có ít nhất một motor đạt 9,9 Nm lên tới 39,4%. Không được đồng nhất route-pass với dư địa điều khiển tốt.
- GUI có surface/trail chạy khoảng 26 FPS trong smoke ngắn, nhưng simulation/wall chỉ khoảng 0,59–0,77. Chưa đạt chứng nhận real-time hoặc human usability.

## Đáp án 25 câu hỏi

| # | Câu hỏi | Kết quả được chứng minh và giới hạn |
|---|---|---|
| 1 | Robot tịnh tiến thật? | Có. Native XYZ và vận tốc WORLD được ghi mỗi 2 ms; không dùng tốc độ quay của ball làm bằng chứng. |
| 2 | Qua vùng nào? | FLAT/MILD/RANDOM/BUMPS/SLOPES: 3/3 route; COURSE: 2/3. Chỉ áp dụng cho các tuyến đã thử. |
| 3 | Độ cao nào? | Mild bound 2 mm; random 1,5 mm; bump sampled peak 1,79084 mm; plateau 1,8 mm; course 1,2 mm. Feature mới giới hạn ở 2% bán kính ball 90 mm. |
| 4 | Slope nào? | MILD khoảng 1,237°; RANDOM 1,095°; BUMPS 1,050°; SLOPES 1,112°; COURSE 1,449°. Không thử slope lớn hơn. |
| 5 | Tilt lớn nhất? | Khoảng 0,0361266 rad, dưới guard 0,12 rad. RMS roll/pitch từng lượt nằm trong CSV. |
| 6 | Vi phạm ROM/contact/torque? | Một ROM guard ở COURSE. Không có unsupported-contact hoặc torque-overlimit được báo ở các route còn lại. Có saturation cao và compliant penetration lớn; chúng không bị giấu. |
| 7 | Đi xa bao nhiêu? | Displacement XY khoảng 0,656–0,763 m. Path length, XYZ và độ cao lưu riêng; không gọi contact settlement ban đầu là độ cao vượt obstacle. |
| 8 | Path trực tiếp trong MuJoCo? | Có, trail xanh từ position thật, chiếu lên facet collision thật với offset hiển thị 2,5 mm. |
| 9 | Bàn phím hoạt động? | GLFW press/release, arrows/WASD, stop/reset/pause/exit, focus-neutral và năm camera. Có native GUI smoke; chưa có thử nghiệm lái bởi người dùng. Forward theo commanded-heading reference, với feedback từ heading đo được. |
| 10 | Có mặt nước nhìn thấy? | Có; có camera close-up và MP4/GIF. Transparency chỉ sửa scene copy. |
| 11 | Dùng modal state thật? | Có. Bessel reconstruction từ q thật, giữ mean height và first volume moment của ROM. Không thêm oscillator, noise hoặc amplitude multiplier để làm đẹp. |
| 12 | Nước đáp ứng tăng tốc, stop và turn? | Hình ảnh dùng q đã sinh bởi các chuyển động native đó. Đây là kiểm chứng liên kết state–renderer, chưa phải kiểm chứng hydrodynamics ngoài thực. |
| 13 | Có reaction coupling? | Có trong native multibody mass matrix và passive slide joints. Wrench diagnostic không được apply thêm lần nữa. |
| 14 | Có double-count mass? | Không trong path được audit/test: 0,499 kg gồm fixed 0,160115 kg và một moving body 0,338885 kg có hai tọa độ. Moment-height approximation và damping chưa được hiệu chuẩn thực nghiệm. |
| 15 | Independent fluid solver? | Có: Chrono 10.0.0, GPU RTX 4090 Laptop, ba run tĩnh, 1.600 bước mới. Status: HIGH_FIDELITY_REFERENCE_IMPLEMENTED; vẫn SPH_BLOCKED. |
| 16 | Có resolution study? | Có screening 8/6 mm và timestep 50/25 µs tại 6 mm. Không gọi là spatial convergence vì actual fluid mass khác giữa spacing và các fine screens thất bại. |
| 17 | Đã so ROM với high-fidelity? | Mới audit các observable tĩnh tương thích. Chưa hoàn thành dynamic comparison về surface, frequency, damping, phase hoặc reaction load. |
| 18 | Accuracy đo được? | SPH pressure RMSE peak khoảng 6,39 Pa và 36,98 Pa; RMS relative velocity peak khoảng 1,18 và 2,04 mm/s. Đây là residual của candidate solver, không phải accuracy đã đo của ROM. |
| 19 | Có thực nghiệm không? | Có thí nghiệm số MuJoCo và GPU SPH mới. Không có hardware hoặc liquid laboratory experiments. |
| 20 | Physical liquid validation? | Chưa. SPH cũng là mô hình số, không phải experimental ground truth. Giữ SPH_BLOCKED. |
| 21 | GIF? | Năm GIF mới từ dữ liệu thật. Không tạo ROM-versus-SPH GIF khi dynamic comparison chưa qua gate. |
| 22 | README? | Đã cập nhật tiếng Anh, đúng hai thí nghiệm chính và ba runtime modes. |
| 23 | Paper? | Bản V2 riêng, có terrain, surface, high-fidelity screening và các hạn chế. Chưa compile PDF vì thiếu TeX; không bịa DOI hoặc authors. |
| 24 | Giữ lịch sử? | SHA-256 của 84.653 entries không đổi, gồm V1. Không sửa driver, CUDA, Chrono core, MuJoCo hoặc checkpoints gốc. |
| 25 | Cần gì để tăng độ tin cậy? | Kiểm chứng contact compliance theo kích thước obstacle, đánh giá motor headroom, giải quyết SPH fill/mass/static-force consistency trước các motion cases; sau đó mới đối chiếu ROM bằng numerical convergence và dữ liệu đo. |

## Ngân sách, kiểm chứng và quyết định

Đã thực hiện **238,619 bước MuJoCo mới**: 236,785 cho routes/pilots, 1,760 cho GUI và 74 cho regression. SPH có **1.600 bước riêng**. Không tính lại các bước lịch sử đã tái sử dụng. Wall-clock terrain đã đo khoảng 330.877 s, dưới cap 1.000 s; mỗi run SPH dưới 60 s.

Regression cuối **41/41 PASS**. Có 468 offline native contact queries không advance time. Các kiểm tra bao gồm mass/COM/gravity, contact, normalization, traversal accounting, surface volume/moment, visualization separation, motion replay và camera transforms.

Bản nâng cấp đủ để chia sẻ một ứng dụng nghiên cứu có số liệu và giới hạn rõ. Chưa đạt mục tiêu cuối: figure-eight 16/16 còn thất bại; contact compliance, motor headroom, moment fidelity của ROM và independent fluid validation còn hạn chế. Không mở thêm campaign, training hoặc full fluid–robot coupling.

Tài liệu chính:

- [README và commands](README.md)
- [Traversal CSV](upgrade/results/traversal.csv) và [motion quality CSV](upgrade/results/motion_quality.csv)
- [Contact audit](docs/v2/contact_traversal_audit.json)
- [ROM/surface/coupling audit](docs/v2/LIQUID_MODEL_VERIFICATION.md)
- [Independent fluid report](LIQUID_ROM_VS_HIGH_FIDELITY_REPORT.md)
- [Paper source](paper/main.tex)
- [Native videos và GIF](upgrade/media/)
