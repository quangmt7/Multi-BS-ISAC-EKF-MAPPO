## Lệnh chạy training

Chạy trong thư mục `D:\ISAC_BWP\Multi-BS-ISAC-EKF-MAPPO`, với môi trường Python đã cài các thư viện của dự án.

```powershell
python main.py --algo ppo
python main.py --algo hierarchical
python main.py --algo hierarchical --single-bwp
```

Ba lệnh lần lượt chạy MAPPO, hierarchical multi-BWP và hierarchical single-BWP.

Số episode và số bước lấy từ YAML. Hiện MAPPO dùng 2.000 × 100 bước; hai chế độ hierarchical dùng 10.000 × 50 bước. Khi so sánh cùng ngân sách, cần thống nhất lại số bước huấn luyện.
