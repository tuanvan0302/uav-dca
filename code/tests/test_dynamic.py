# -*- coding: utf-8 -*-
"""
Kiểm chứng lớp 1 cho Giai đoạn B: bộ giải với vật cản phụ thuộc
thời gian (động và biến dạng) ở chế độ sự thật nền, và cổng kiểm
chứng riêng của M2 (sai số dự báo tại cuối chân trời < 0,5 m —
ngưỡng ĐỀ XUẤT trong tài liệu thiết kế).

Chạy:  python3 -m tests.test_dynamic   (từ gốc code/)"""

from __future__ import annotations

import numpy as np

from core.dca_planner import PlannerParams, plan
from core.obstacle_predict import forecast_error, kalman_obstacles, truth_obstacles
from sim.dynamic_scenarios import dynamic_scenario
from sim.scenarios import scenario

PRM = PlannerParams()
FAIL = []


def _check(name: str, ok: bool, extra: str = ""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAIL.append(name)


def _test_truth_mode(name: str):
    sc = dynamic_scenario(name)
    obs = truth_obstacles(sc["truth_fns"], 0.0, PRM.K, PRM.dt)
    res = plan(sc["start"], np.zeros(3), sc["goal"], obs, PRM)
    _check(
        f"{name} (sự thật nền): bộ giải báo thành công",
        res.success,
        f"(biến dư {res.total_slack:.2e}, thời gian giải {res.solve_time:.2f} s)",
    )
    _check(
        f"{name} (sự thật nền): khe hở thật >= biên an toàn",
        res.min_clearance >= PRM.d_safe - 1e-3,
        f"(khe hở nhỏ nhất {res.min_clearance:.3f} m)",
    )


def test_static_dict_matches_tuple():
    # Vật tĩnh biểu diễn dạng dict (hằng theo k, đầu ra M2) phải cho
    # kết quả giống hệt dạng tuple (tâm, bán kính) — kiểm chứng chuẩn hoá.
    sc = scenario("S1_mot_vat")
    o, r = sc["obstacles"][0]
    res_tuple = plan(sc["start"], np.zeros(3), sc["goal"], sc["obstacles"], PRM)
    as_dict = [
        dict(
            centers=np.repeat(o[None, :], PRM.K + 1, axis=0),
            radii=np.full(PRM.K + 1, r),
            rho=np.zeros(PRM.K + 1),
        )
    ]
    res_dict = plan(sc["start"], np.zeros(3), sc["goal"], as_dict, PRM)
    _check(
        "Tương đương tĩnh: dạng dict cho cùng kết quả dạng tuple",
        res_dict.success == res_tuple.success
        and abs(res_dict.min_clearance - res_tuple.min_clearance) < 1e-6,
        f"(khe hở {res_dict.min_clearance:.3f} so với {res_tuple.min_clearance:.3f} m)",
    )


def test_m2_gate():
    rng = np.random.default_rng(20261110)
    for name in ("D1_cat_ngang", "D2_doi_dau", "D3_bien_dang"):
        sc = dynamic_scenario(name)
        pred = kalman_obstacles(sc["truth_fns"], t_now=2.0, K=PRM.K, dt=PRM.dt, rng=rng)
        err = forecast_error(sc["truth_fns"], pred, 2.0, PRM.K, PRM.dt)
        _check(
            f"Cổng M2 — {name}: sai số dự báo cuối chân trời < 0,5 m (đề xuất)",
            err < 0.5,
            f"(sai số {err:.3f} m, ρ_K = {pred[0]['rho'][-1]:.3f} m)",
        )


if __name__ == "__main__":
    test_static_dict_matches_tuple()
    for n in ("D1_cat_ngang", "D2_doi_dau", "D3_bien_dang", "D4_ket_hop"):
        _test_truth_mode(n)
    test_m2_gate()
    print("\nLayer-1 checks, phase B:", "PASS" if not FAIL else f"FAIL {FAIL}")
