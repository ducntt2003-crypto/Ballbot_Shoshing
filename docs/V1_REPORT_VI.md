# SloshBot-RL — Báo cáo bàn giao 16/11/2026

**Đã hoàn thành gói nghiên cứu cục bộ gồm đúng hai thí nghiệm, ứng dụng bàn phím và bản thảo tiếng Anh. Chưa publish GitHub. Mục tiêu nghiên cứu cuối cùng chưa đạt đầy đủ.**

## Experiment 1 — Figure-eight 1,50 × 0,75 m, 30 giây

Đánh giá6 cấu hình trên3 cặp reset/terrain giống nhau:18 kết quả. Tái sử dụng6 traces V4/PPO đã có; chạy12 lượt mới choV2/V3/FF. Không huấn luyện lạiPPO.

| Controller | Đủ30s | Full-task | Kết quả chính |
|---|---:|---:|---|
| V2 |0/3|0/3|Tracking guard1,344–1,436s|
| V3 |0/3|0/3|Tracking guard1,338–1,418s|
| V2_FFv1 |0/3|0/3|ROM guard khoảng20–24s|
| V3_FFv1 |3/3|0/3|15/16windows, yaw drift khoảng85–89°|
| V4 privileged |3/3|0/3|15/16windows|
| Standard PPO+V4 privileged |3/3|0/3|15/16windows|

Giữ nguyên các ngưỡng gốc; không xếp hạng năng lượng của failed-prefix như thể đã sống đủ30s. PID balance-only, teacher và Gaussian Mixture PPO được ghi NOT EVALUATED do chưa có hợp đồng tương đương đã kiểm chứng cho bản phát hành này. Không gọi6 phương pháp có cùng thông tin: V4/PPO là privileged.

## Experiment 2 — Bàn phím và command replay

Ứng dụng `teleop_keyboard.py` hỗ trợ arrow/WASD, SPACE stop, R reset, P pause, ESC exit và4 camera. Có key-release/focus-neutral, giới hạn tốc độ0,04m/s, yaw-rate0,08rad/s và slew. Key không đi thẳng thành torque. Native logs chứa lệnh, state, modal state, torque và contact.

- Pilot3s trênflat/benchmark/playground hoàn thành.
- Script16s:2/3 đạt; seed2618001 ROM guard tại native khoảng11,986s. Trace valid cuối11,984s không được dùng để phủ nhận crossing.
- Route26s:3/3 đạt stability gates. Đi thực qua flat/transition/mild-rough/smooth-uneven, bán kính lớn nhất khoảng0,719–0,726m. Max tilt khoảng0,0293–0,0353rad; modal displacement tối đa khoảng6,49–7,04mm. Chưa kiểm chứng raised features hoặc toàn bộplayground.
- Một phép bổ sung residual modalV4, không đổi gain/physics: case1 đạt16s, case2 ROM guard khoảng13,958s; dừng theo gate, không chạy case3/route củavariant. Không chứng minh variant tốt hơn.
- GUI smoke đã chạy trên cửa sổ thật: khoảng25,5FPS, simulation/wall≈1 trong2s; không có human-driving study và không tuyên bố sustained realtime.

Định nghĩa settling rõ ràng: velocity error≤0,015m/s, yaw-rate error≤0,04rad/s, tilt≤0,06rad trong0,5s trước lần đổi lệnh kế tiếp. Các đoạn không settle giữ censored/null.

## Artifacts và kiểm chứng

- README/thuật toán/tái lập bằng tiếng Anh,2 bộ bảngCSV, biểu đồPNG/PDF/SVG,5nativevideos và4GIF từ dữ liệu thật.
- Paper source mới trong `paper/main.tex`, dựa trên manuscript trước và tổ chức lại quanh2 thí nghiệm. ThiếuTeXcompiler nên chưa có compiledpaperPDF; không bịa authors/DOI.
-31/31 essential tests hiện tạiPASS;704tests là receipt của campaign trước, không cộng thành733tests mới.
-84.083 entries lịch sử không đổi SHA-256.
- Tổng native steps mới được hạch toán: **168,103**, gồm evaluation,2GUIsmoke và các test cần thiết; training mới=0. Giữ các lỗi khởi tạo/test/cleanup trong logs và accounting.
- Không sửa model/physics/driver/Chrono/CUDA, không thay checkpoints gốc. Hai metadataJSON trong bản sao export chỉ chuẩn hóa đường dẫn checkpoint lịch sử; source originals vẫn giữ nguyên.

## Quyết định kỹ thuật

Gói phù hợp để chia sẻ code và phản biện nội bộ về benchmark cùng các giới hạn đã đo. Chưa đủ cơ sở tuyên bố đã giải quyết figure-eight+slosh, PPO superiority, teleop ổn định mọi lệnh, sim-to-real hay realfluidvalidation. `SPH_BLOCKED` còn nguyên.

Không mở thêm campaign/training. Không tạo masterprompt tiếp theo. Không publishGitHub vì yêu cầu hiện tại chưa cấp quyền publish. Xem `docs/RELEASE_READINESS.md`, `docs/CLAIM_EVIDENCE.md`, manifest và receipt đóng gói để đối chiếu.

## Bổ sung kiểm tra heightfield cuối

Playground hiện chuẩn hóa heightfield về[0,1] bằng compile XML tạm, giữ nguyên world-height trong sai số lượng tửfloat32, không sửa robot/liquid. Pilot3s vàGUIsmoke1s mới đạt. Các route16/26s đã hoàn thành vẫn thuộc biểu diễn trước sửa; chưa tuyên bố đã kiểm chứng dài cho representation mới. Sai số so sánh3s theo từng đại lượng nằm trong `docs/heightfield_fix_validation.json`. Test cuối31/31PASS; hạch toán cả các lượt test lỗi trước sửa.
