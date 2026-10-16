Integrated Sensing and Communication (ISAC) cho phép hệ thống sử dụng chung tài nguyên truyền thông và sensing, trong đó resource allocation là một trong những bài toán quan trọng do sự tồn tại của trade-off giữa hiệu năng com munication và sensing.

 - config/

    + system.yaml: Chứa tham số hệ thống: số BS, UE, target, PRB, bandwidth, slot duration, công suất tối đa và các tham số simulation.
 
    + ekf.yaml:Chứa tham số EKF: mô hình động học, ma trận Q, covariance khởi tạo và các tham số measurement noise.

    + mappo.yaml:Chứa hyperparameter của MAPPO/PPO: learning rate, gamma, GAE lambda, clip epsilon, batch size, số epoch update, v.v.

 - scenario/

    + scenario.py: Tạo thế giới mô phỏng: vị trí BS/UE/target, trạng thái ban đầu của target và các thành phần tĩnh của scenario.

 - ppo/

    + train.py: Train mô hình (chú ý có những thành phần state chung mà các BS chia sẻ, cũng như có những thành phần trong state là riêng cho từng BS)
    + agent.py: Toàn bộ về agent: định nghĩa mô hình, cập nhật chính sách, đưa ra hành động từ state
    + env.py: Bộ điều phối một timestep: nhận action từ MAPPO, gọi channel/sensing/communication, chạy EKF, cập nhật state/AoI, tính reward và tạo observation tiếp theo.

 - Channel/

    + channel.py: Tính/duy trì channel giữa BS–UE và các đại lượng cần cho sensing. Cung cấp SINR và spectral efficiency C_{b,k,r}=log2(1+SINR) cho communication.

 - sensing/

    + detection.py: Tính xác suất detection P^D_{b,m} và xác định measurement nào của BS được sử dụng.
    + measurement.py: Tạo measurement range/angle y_{b,m}=[r_{b,m}, theta_{b,m}]^T và measurement covariance R_{b,m}. Góc dùng atan2.
    + efk.py: Thực hiện prediction và update. Nếu nhiều BS có measurement, fusion/batch update vào một common target state/covariance. Nếu không có measurement, chỉ prediction.

 - Communication/

    + communication.py: Tính SINR/rate/throughput của UE từ assignment, power và channel; kiểm tra communication QoS nếu cần.

 - evaluation/

    + metrics.py: Tính các metric báo cáo: average AoI, position/velocity uncertainty, power utilization, throughput, sensing/communication success và constraint violations.
    + benchmark.py: Chạy và so sánh các baseline/benchmark độc lập với environment và training code.

 - results/

    + Thư mục output: Lưu checkpoint, log, metric, figure và kết quả cuối cùng. Không chứa logic simulation.

 - main.py

    + Điểm chạy chính ở thư mục gốc: load config → tạo scenario → tạo environment → tạo MAPPO → chạy episode/timestep → lưu kết quả.

## Quy ước mô phỏng và reward

- Mỗi slot cập nhật chuyển động của mọi target. EKF bắt đầu từ estimate có sai số Gaussian với covariance P0 trong config/ekf.yaml; simulator giữ ground truth riêng. Khi triển khai so sánh, các baseline cần dùng bản sao của env.initial_estimates để có cùng prior.
- C2S/IBI chỉ khai báo ở config/system.yaml, mục channel_and_interference. Hệ số C2S 1e-11 là giả định coupling công suất còn dư sau cách ly/triệt nhiễu của mô phỏng hiện tại. Cấu hình MAPPO cũ còn bản sao mâu thuẫn sẽ báo lỗi.
- Reward = âm của chi phí AoI, uncertainty, công suất, trừ tổng penalty vi phạm ngưỡng đã chuẩn hóa. Penalty không âm; UE đạt hoặc vượt tốc độ tối thiểu không bị phạt và không nhận thêm thưởng throughput. Vi phạm được tính riêng từng UE/target trước khi lấy trung bình.
- Reward mới có thể âm và không so trực tiếp giá trị return với lượt dùng reward trước. Giữ nguyên alpha, beta, gamma và các trọng số c1..c4.

Chạy một lượt mới trong thư mục riêng để giữ kết quả trước:

    py -3.12 main.py --output-dir results/run_20261016

Các checkpoint trước thay đổi cấu hình/reward cần được giữ để đối chiếu; bắt đầu lượt huấn luyện mới khi đánh giá mô hình đã sửa.

## Xuất hình báo cáo

Sau training, mỗi biểu đồ được lưu thành một PNG riêng trong results/figures: training_curve, comm_rate, comm_qos_satisfaction, sensing_success_prob, sensing_aoi, sensing_pos_error, sensing_vel_error, ppo_policy_loss, ppo_critic_loss và ppo_entropy. Metric không có dữ liệu sẽ không sinh hình trống. Đường return giữ cửa sổ trung bình trượt 20 episode.

Xuất lại hình từ metrics đã có, không chạy training:

    py -3.12 main.py --figures-only --output-dir results

Các ảnh ghép do phiên bản cũ tạo không được ghi thêm ở lượt mới; nếu dùng lại thư mục cũ, chuyển chúng sang thư mục lưu trữ trước khi chọn hình báo cáo.
