# -*- coding: utf-8 -*-
"""
Test hạ tầng thực nghiệm của sim.sim_realtime theo checklist của giao
thức thực nghiệm (gói sim/ nằm ngoài code/):

  1. Quy tắc hãm bão hòa §7.1: từ v = 3 m/s, lệnh hãm không vượt
     u_max ở bất kỳ bước nào và ‖v‖ ≤ 0,05 m/s sau tối đa 3 bước
     (u_max = 4). Kèm ca chéo: clip theo chuẩn giữ ‖u‖ ≤ u_max, trong
     khi clip theo từng thành phần cho ‖u‖ tới √3·u_max.
  2. Jerk §1.6: lấy mẫu lại trên lưới Δt đều — một chunk một phần
     (0,002 s) không được làm mẫu số co lại.
  3. Tích hợp ngắn: một episode realtime rất ngắn phải có đủ trường
     T_latency = t_predict + t_plan theo chu kỳ, τ̂ của chu kỳ sau
     bằng T_latency của chu kỳ trước, và metrics có status §6.4 +
     safe_success.

Chạy: python3 -m tests.test_sim_realtime   (từ gốc code/)"""

from __future__ import annotations

import numpy as np

from sim.sim_realtime import (
    BRAKE_V_EPS,
    brake_command,
    jerk_rms_on_grid,
    resample_controls,
    run_episode,
)

DT, U_MAX = 0.25, 4.0


def test_brake_from_3ms():
    v = np.array([3.0, 0.0, 0.0])
    steps = 0
    while np.linalg.norm(v) > BRAKE_V_EPS and steps < 10:
        u = brake_command(v, DT, U_MAX)
        assert np.linalg.norm(u) <= U_MAX + 1e-12, f"vượt u_max: {u}"
        v = v + DT * u
        steps += 1
    assert steps <= 3, f"hãm từ 3 m/s mất {steps} bước (> 3)"
    assert np.linalg.norm(v) <= BRAKE_V_EPS
    # đã dừng hẳn -> lệnh 0
    assert np.allclose(brake_command(v, DT, U_MAX), 0.0)
    print(f"  hãm 3 m/s -> dừng sau {steps} bước, ‖u‖ ≤ {U_MAX} mọi bước")


def test_brake_diagonal_norm_clip():
    # Ca chéo: clip theo từng thành phần sẽ cho ‖u‖ tới √3·4 ≈ 6,93.
    v = np.array([1.9, 1.19, -0.09])
    u = brake_command(v, DT, U_MAX)
    assert np.linalg.norm(u) <= U_MAX + 1e-12
    # hướng hãm phải ngược hướng v
    assert np.dot(u, v) < 0
    # ca bão hòa đầy: v lớn -> ‖u‖ đúng bằng u_max
    v2 = np.array([3.0, 3.0, 3.0])
    u2 = brake_command(v2, DT, U_MAX)
    assert abs(np.linalg.norm(u2) - U_MAX) < 1e-12
    print("  clip theo chuẩn: ‖u‖ ≤ u_max kể cả hướng chéo, bão hòa đúng u_max")


def test_jerk_grid_resampling():
    u0 = np.array([0.0, 0.0, 0.0])
    u1 = np.array([4.0, 0.0, 0.0])
    # chunk một phần 0,002 s ở giữa: không được làm mẫu số co lại
    segs = [(0.0, 0.25, u0), (0.25, 0.002, u1), (0.252, 0.248, u1), (0.5, 0.25, u1)]
    grid = resample_controls(segs, DT, 0.75)
    assert len(grid) == 3
    assert np.allclose(grid[0], u0) and np.allclose(grid[1], u1)
    j = jerk_rms_on_grid(segs, DT, 0.75)
    expect = float(np.sqrt(np.mean([np.linalg.norm(u1 - u0) ** 2 / DT**2, 0.0])))
    assert abs(j - expect) < 1e-9, f"jerk {j} != {expect}"
    print(f"  jerk trên lưới Δt: {j:.3f} (đúng mẫu số Δt, không phình)")


def test_episode_latency_fields():
    out = run_episode(
        "D1_cat_ngang", mode="kalman", seed=9000000, time_model="realtime", t_max=3.0
    )
    m, cyc = out["metrics"], out["cycles"]
    assert m["status"] in ("ok", "solver_fail", "timeout")
    assert "safe_success" in m and "no_flight" in m
    assert len(cyc) >= 1
    for c in cyc:
        assert abs(c["T_latency"] - (c["t_predict"] + c["t_plan"])) < 1e-9
        assert c["delay_applied"] == c["T_latency"]
    for prev, cur in zip(cyc[:-1], cyc[1:]):
        assert abs(cur["tau_hat"] - prev["T_latency"]) < 1e-9
    print(
        f"  episode tích hợp: {len(cyc)} chu kỳ, "
        f"T_latency = t_predict + t_plan và τ̂ chuỗi đúng; "
        f"status={m['status']}"
    )


if __name__ == "__main__":
    test_brake_from_3ms()
    test_brake_diagonal_norm_clip()
    test_jerk_grid_resampling()
    test_episode_latency_fields()
    print("All sim_realtime infrastructure tests passed.")
