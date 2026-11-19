# Bàn giao README nghiên cứu — 2026-11-19

Bản xuất bản riêng này chuyển README sang tiếng Anh, ưu tiên mô hình, thuật toán, phương pháp PPO và số liệu thực nghiệm. Bản V2 tại thư mục gốc được giữ nguyên; các tài liệu thay đổi có bản sao lịch sử. Không huấn luyện thêm và không chạy lại các chiến dịch hoàn tất.

Đã tính lại bảng 18 thử nghiệm figure-eight và 24 thử nghiệm độ cao từ CSV, bổ sung dữ liệu nguồn cho learning curves. 500.000 transitions thực thi, 497.600 tối ưu, một seed policy; tám worker không phải tám lần lặp độc lập. Bảng 5/10/20 mm và hỗn hợp là trung bình ba seed đánh giá. PPO chỉ chênh rất nhỏ, chưa có bằng chứng ưu thế. Nhiệm vụ đủ 16 cửa sổ chưa thành công.

41/41 tests đạt. Doctor, V4 20 ms, PPO 20 ms và cửa sổ GUI 1 giây đã chạy thành công: tổng 517 native steps mới, không phải simulation nghiên cứu mới. GUI đạt 0,92 giây mô phỏng trong khoảng 1,01 giây thực; chưa khẳng định realtime. Lưu receipts và logs trong publication_validation/.

SPH_BLOCKED: ROM chưa được xác nhận độc lập bằng CFD/SPH, không có hardware validation. Gò trơn 20 mm không phải bậc thẳng đứng. Contact compliance và sai số thời gian còn giới hạn diễn giải. Không đổi mô hình, gains, checkpoint hay dữ liệu lịch sử.

Bản công bố giữ manuscript TeX hiện có, không tạo PDF giả. Manifest SHA-256 kiểm tra toàn bộ tệp xuất bản và đối chiếu 929 tệp V2 (các tài liệu thay đổi được kiểm tra ở bản sao lịch sử). Trạng thái GitHub và commit được ghi ở biên bản bàn giao bên ngoài cây Git để tránh tự tham chiếu hash commit.
