Integrated Sensing and Communication (ISAC) cho phép hệ thống sử dụng
chung tài nguyên truyền thông và sensing, trong đó resource allocation là một
trong những bài toán quan trọng do sự tồn tại của trade-off giữa hiệu năng com
munication và sensing.

config/
 
 -system.yaml: Chứa tham số hệ thống: số BS, UE, target, PRB, bandwidth, slot duration, công suất tối đa và các tham số simulation.
 
 -ekf.yaml:Chứa tham số EKF: mô hình động học, ma trận Q, covariance khởi tạo và các tham số measurement noise.
 
 -mappo.yaml:Chứa hyperparameter của MAPPO/PPO: learning rate, gamma, GAE lambda, clip epsilon, batch size, số epoch update, v.v.

scenario/

 -scenario.py: Tạo thế giới mô phỏng: vị trí BS/UE/target, trạng thái ban đầu của target và các thành phần tĩnh của scenario.
