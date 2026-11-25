# -*- coding: utf-8 -*-
"""
Kịch bản cho hai loạt kiểm thử theo giao thức kiểm chuẩn (2026-11-13):

- Loạt A (tinh thần FlightBench): thang độ khó 8 cấp cho trường vật cản
  tĩnh ngẫu nhiên — cấp càng cao, số vật cầu càng nhiều. Mỗi cấp 3 lần
  lặp với hạt giống khác nhau, mọi tham số sinh được công bố để tái lập.
- Loạt B (tinh thần FLYINGTRUST): biến thiên nền tảng bay — 3 cấp năng
  lực (giới hạn vận tốc/gia tốc khác nhau) x các kịch bản đại diện tĩnh
  và động.
"""
from __future__ import annotations

import numpy as np

START = np.array([0.0, 0.0, 1.5])
GOAL = np.array([10.0, 0.0, 1.5])

# Số vật cầu theo cấp độ khó L1..L8
LEVEL_COUNTS = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 16, 8: 20}
REPS = 3


def random_field(level: int, rep: int) -> dict:
    """Trường vật cầu ngẫu nhiên có hạt giống cố định cho (cấp, lần lặp).

    Vùng đặt vật: x in [1.2, 8.8], y in [-2.8, 2.8], z in [0.9, 3.0];
    bán kính đều trong [0.6, 1.2] m; tâm bị loại nếu cách điểm đầu hoặc
    đích dưới r + 1.2 m.
    """
    n = LEVEL_COUNTS[level]
    rng = np.random.default_rng(1000 + 100 * level + rep)
    obs = []
    tries = 0
    while len(obs) < n and tries < 500:
        tries += 1
        c = np.array([rng.uniform(1.2, 8.8), rng.uniform(-2.8, 2.8),
                      rng.uniform(0.9, 3.0)])
        r = float(rng.uniform(0.6, 1.2))
        if np.linalg.norm(c - START) < r + 1.2:
            continue
        if np.linalg.norm(c - GOAL) < r + 1.2:
            continue
        obs.append((c, r))
    return {"start": START.copy(), "goal": GOAL.copy(), "obstacles": obs}


# Ba cấp năng lực bay: (tên, v_max m/s, u_max m/s^2)
PLATFORMS = [
    ("P1_han_che", 2.0, 3.0),     # nền tảng hạn chế
    ("P2_tieu_chuan", 3.0, 4.0),  # cấu hình gốc của bài báo
    ("P3_linh_hoat", 5.0, 8.0),   # nền tảng linh hoạt
]
