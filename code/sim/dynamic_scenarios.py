# -*- coding: utf-8 -*-
"""
Kịch bản vật cản ĐỘNG và BIẾN DẠNG (M1, Giai đoạn B).

Mỗi kịch bản cho quỹ đạo bay giống các kịch bản tĩnh: điểm đầu
(0, 0, 1,5), đích (10, 0, 1,5). Vật cản được mô tả bằng sự thật nền
theo thời gian: mỗi vật là một hàm (t) -> (tâm (3,), bán kính), để
mô-đun dự báo (M2) lấy mẫu hoặc ước lượng, và để khâu đánh giá tính
khe hở thật tại đúng thời điểm tương ứng.

  D1_cat_ngang : một vật cầu r = 1,0 cắt ngang đường bay tại x = 5
                 với vận tốc ngang không đổi 0,9 m/s.
  D2_doi_dau   : một vật cầu r = 0,9 bay ngược chiều, lệch trục 0,9 m,
                 vận tốc 1,1 m/s.
  D3_bien_dang : một vật cầu đứng yên tại (5, 0, 1,5), bán kính dao
                 động r(t) = 1,0 + 0,35.sin(2π t / 6) — phình ra tối đa
                 1,35 m, co lại tối thiểu 0,65 m, chu kỳ 6 giây.
  D4_ket_hop   : D1 và D3 cùng lúc (hai vật cản).
"""
from __future__ import annotations

import numpy as np

START = np.array([0.0, 0.0, 1.5])
GOAL = np.array([10.0, 0.0, 1.5])


def _d1(t: float):
    return np.array([5.0, -3.5 + 0.9 * t, 1.5]), 1.0


def _d2(t: float):
    return np.array([11.0 - 1.1 * t, 0.9, 1.5]), 0.9


def _d3(t: float):
    return np.array([5.0, 0.0, 1.5]), 1.0 + 0.35 * np.sin(2.0 * np.pi * t / 6.0)


_SCENARIOS = {
    "D1_cat_ngang": [_d1],
    "D2_doi_dau": [_d2],
    "D3_bien_dang": [_d3],
    "D4_ket_hop": [_d1, _d3],
}


def dynamic_scenario(name: str) -> dict:
    """Trả kịch bản: start, goal, và danh sách hàm sự thật nền của vật cản."""
    return dict(name=name, start=START.copy(), goal=GOAL.copy(),
                truth_fns=list(_SCENARIOS[name]))


def all_dynamic_names() -> list:
    return list(_SCENARIOS.keys())


def sample_truth(truth_fns, t0: float, K: int, dt: float):
    """Lấy mẫu sự thật nền trên lưới chân trời [t0, t0 + K.dt]:
    trả danh sách (tâm (K+1, 3), bán kính (K+1,)) cho từng vật cản."""
    out = []
    for fn in truth_fns:
        centers = np.empty((K + 1, 3))
        radii = np.empty(K + 1)
        for k in range(K + 1):
            c, r = fn(t0 + k * dt)
            centers[k] = c
            radii[k] = r
        out.append((centers, radii))
    return out
