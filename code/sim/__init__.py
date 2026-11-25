# -*- coding: utf-8 -*-
"""sim — hạ tầng mô phỏng và thư viện kịch bản (đội dự án dựng,
nối vào cây refactor của người dùng ngày 2026-11-24).

Gồm: sim_realtime (vòng mô phỏng thời gian thực, DV-H10/DV-H12 đã
áp), dynamic_scenarios / scenarios / benchmark_scenarios (thư
viện kịch bản của Bài 1) và scenarios_pkg (sổ đăng ký 49 kịch bản
+ họ F ellipsoid). Các module này import lõi qua core.* và
baseline qua baselines.* — thư mục code/ phải nằm trên sys.path.
"""
