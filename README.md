# SOTA — ST-DCA và các baseline

Code của phương pháp ST-DCA: lập quỹ đạo cho một UAV tránh vật cản động
bằng DCA dạng phạt, kèm biên an toàn hiệu chuẩn conformal (H1), vật cản
ellipsoid (H2) và ba baseline Tier-1 để so sánh.

## Cấu trúc

| Thư mục / file | Nội dung |
| --- | --- |
| `code/core/dca_planner.py` | Bộ giải penalty-DCA (`plan`, `PlannerParams`, `PlanResult`) |
| `code/core/ellipsoid.py` | Hình học ellipsoid: gauge, gradient, mặt cắt cho bộ giải |
| `code/core/obstacle_predict.py` | Dự báo vật cản cầu (sự thật nền, Kalman) |
| `code/core/shape_forecast.py` | Dự báo hình dạng ellipsoid hai lớp |
| `code/core/safety_margin.py` | Biên an toàn H1: δ_trôi, δ_mẫu, cửa sổ trễ, chế độ legacy |
| `code/core/conformal.py` | Hiệu chuẩn split conformal cho ρ(ε) |
| `code/core/astar_seed.py` | Điểm khởi tạo A* cho bộ giải |
| `code/core/method_adapter.py` | Bọc `plan()` cho các cấu hình ablation A1/A2/A3 |
| `code/core/conformal_tables/` | Bảng conformal `stdca_h1.{npz,json}` mà `method_adapter` nạp cho A1/A3 (trạng thái: chuẩn bị GĐ2, chưa đóng băng chính thức) |
| `code/baselines/` | Ngo-2023, CBF-2021, MPC-QP — xem `code/baselines/README_BASELINES.md` |
| `code/sim/` | Vòng lặp mô phỏng `sim_realtime`, kịch bản `scenarios`, `dynamic_scenarios`, `benchmark_scenarios` và thư viện kịch bản `scenarios_pkg/` |
| `code/experiments/` | `experiment_runner` và 6 script thực nghiệm TN1–TN6 — xem `code/experiments/HUONG_DAN_THUC_NGHIEM.md` |
| `code/tests/` | Test ngoại tuyến |
| `AGENTS.md` | Quy tắc đặt tên, comment, định dạng cho người và agent sửa code |

## Chạy

```bash
pip install -r requirements.txt
cd code
python3 -m tests.test_safety_margin             # một test
python3 -m baselines.test_baselines             # bộ test baseline
python3 experiments/tn1_khong_vat_can.py --quick  # kiểm khói thực nghiệm
```

Danh sách đầy đủ các test nằm trong `CMD` của `Dockerfile`.
