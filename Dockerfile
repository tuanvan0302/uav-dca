# Dockerfile tuỳ chọn cho gói SOTA. CHƯA được build thật; coi lần build
# đầu tiên là lần kiểm chứng.
FROM python:3.12-slim

WORKDIR /pkg
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/ ./code/
WORKDIR /pkg/code

# Mặc định: chạy toàn bộ test ngoại tuyến từ gốc code/. Mỗi test là một
# module của gói tests/ có điểm vào __main__.
CMD ["bash", "-lc", "for t in test_planner test_dynamic test_gauge test_shape_forecast test_safety_margin test_conformal test_method_adapter test_sim_realtime test_ellipsoid_obs test_astar_seed; do python3 -m tests.$t || exit 1; done && python3 -m baselines.test_baselines"]
