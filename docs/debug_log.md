> Cập nhật 27/11/2026: kết quả hierarchical hiện lấy từ nhánh Tien, commit fde4630, gồm 10.000 episode × 50 bước = 500.000 bước. Cấu hình mới có SAC/DQN batch size 128. Run này dùng code nền dea5c2d, trước các sửa reward penalty, replay và metric của Khang. Các nhận xét và số liệu hierarchical bên dưới thuộc run Khang 2.000 episode trước đó, có thể truy lại ở commit d4a0599. MAPPO và single-BWP vẫn là kết quả tại d4a0599. Cần đánh giá lại với cùng ngân sách và phiên bản mô hình trước khi dùng để so sánh thuật toán.

# Debug log — các mục còn mở sau training ngày 24–25/11/2026

Nguồn là ba log huấn luyện trong [results/metrics](../results/metrics/). Mỗi phương pháp có 2.000 episode liên tiếp và 200.000 bước. Các so sánh đầu/cuối bên dưới lấy trung bình trên episode 1–100 và 1.901–2.000. Kết quả thuộc một lượt huấn luyện cho mỗi phương pháp. Để so sánh hiệu năng trong bài báo cần đánh giá policy đóng băng trên seed giữ riêng và lặp lại huấn luyện với nhiều seed.

## C3 — Biến động tỷ số xác suất khi cập nhật MAPPO

`clip_fraction` trung bình giảm từ 0,7394 xuống 0,5922. `ratio_std` đạt 359.304,73 ở episode 9 (scenario seed 50), trong khi trung bình 100 episode cuối là 0,5897. Cần kiểm tra phân bố chênh lệch log-prob mới/cũ theo minibatch, đặc biệt các hành động tạo đuôi lớn ở đầu training. Sau đó mới đánh giá tác động của learning rate và số PPO epoch.

## C1 — Reward communication ít nhạy với throughput

Với MAPPO, `mean_embb_rate_mbps` giảm từ 49,73 xuống 42,63 Mbps/UE nhưng `r_comm` chỉ giảm từ 0,6321 xuống 0,6292. Ở hierarchical multi-BWP, các số tương ứng là 33,67 xuống 32,64 Mbps/UE và 0,6321 xuống 0,6304. Reward subframe dùng tỷ lệ UE được cấp PRB nên vẫn có thể gần cực đại khi tốc độ giảm. Nhóm cần chốt throughput trên ngưỡng có phải mục tiêu cần tối ưu trước khi đổi reward.

## C2 — Guardband cost ít phân biệt được các phương pháp

`guardband_cost` trung bình mỗi episode bằng 10 trong cả 2.000 episode của MAPPO và hierarchical multi-BWP. Ở single-BWP, giá trị dao động từ 9,4 đến 10. Cần kiểm tra từng BS với ba lịch cố định (tắt cả hai BWP, bật một, bật cả hai) và so sánh cost với số BWP hoạt động. Single-BWP theo dịch vụ vẫn cho phép các dịch vụ khác nhau cùng kích hoạt hai BWP tại một BS.

## C5 — Một số episode có sai số vận tốc lớn

Sai số vận tốc trung bình đã giảm ở cả ba phương pháp. Riêng hierarchical multi-BWP, sai số giảm từ 2,83 xuống 2,46 m/s, còn uncertainty vận tốc giảm từ 7,17 xuống 5,71 m²/s². Vẫn có episode có sai số trung bình cao, dù cả ba log đều ghi 0 episode mất target.

| Phương pháp | Episode | Scenario seed | Sai số vận tốc trung bình episode (m/s) |
|---|---:|---:|---:|
| MAPPO | 1.004 | 1.045 | 54,54 |
| Hierarchical multi-BWP | 500 | 541 | 49,28 |
| Hierarchical single-BWP | 912 | 953 | 87,83 |

Đối chiếu trên cùng quỹ đạo giữa cấp sensing theo policy và có phép đo liên tiếp. Kiểm tra thời điểm sai số tăng, số lần đo, sai số theo target và covariance của EKF. Log cấp episode chưa xác định được nguyên nhân.

## B2 — Phạm vi reward của từng BS

Trainer dùng team reward. Env vẫn đưa `local_rewards` vào thông tin chẩn đoán của bước. Nhóm cần xác nhận cách mô tả mục tiêu phối hợp giữa các BS trong bài. Chỉ đổi tín hiệu huấn luyện nếu bài toán yêu cầu gán reward riêng cho từng BS.
