Integrated Sensing and Communication (ISAC) cho phép hệ thống sử dụng
chung tài nguyên truyền thông và sensing, trong đó resource allocation là một
trong những bài toán quan trọng do sự tồn tại của trade-off giữa hiệu năng com
munication và sensing.

+ config/
 
  -system.yaml: Chứa tham số hệ thống: số BS, UE, target, PRB, bandwidth, slot duration, công suất tối đa và các tham số simulation.
 
  -ekf.yaml:Chứa tham số EKF: mô hình động học, ma trận Q, covariance khởi tạo và các tham số measurement noise.
 
  -mappo.yaml:Chứa hyperparameter của MAPPO/PPO: learning rate, gamma, GAE lambda, clip epsilon, batch size, số epoch update, v.v.

+ scenario/

  -scenario.py: Tạo thế giới mô phỏng: vị trí BS/UE/target, trạng thái ban đầu của target và các thành phần tĩnh của scenario.

+ ppo/
  -train.py: Train mô hình (chú ý có những thành phần state chung mà các BS chia sẻ, cũng như có những thành phần trong state là riêng cho từng BS)
  -agent.py: Toàn bộ về agent: định nghĩa mô hình, cập nhật chính sách, đưa ra hành động từ state
  -env.py: Bộ điều phối một timestep: nhận action từ MAPPO, gọi channel/sensing/communication, chạy EKF, cập nhật state/AoI, tính reward và tạo observation tiếp theo.




