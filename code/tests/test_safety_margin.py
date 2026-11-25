# -*- coding: utf-8 -*-
"""
test_safety_margin.py — Kiểm thử core/safety_margin.py bằng chính các ví dụ số của
Chương 6 (báo cáo hợp nhất ST-DCA):
  - v̂ = 1,5 m/s, a_max = 1 m/s², T_r = 1 s  → δ_trôi = 2,0 m
  - cùng thông số, T_r = 0,25 s             → δ_trôi ≈ 0,41 m
  - u_max = 4, a_max = 0, Δt = 0,25 s       → δ_mẫu = 0,03125 ≈ 0,031 m
  - quy ước T_r dùng thời gian giải của chu kỳ LIỀN TRƯỚC
  - chế độ legacy thu về δ = 0,03 m và ρ = κ·sqrt(σ²+σ²), κ = 2
  - DV-H7: cửa sổ hiệu lực min(T_r, T_cap), cờ capped ⟺ T_r > T_cap,
    mặc định không trần (Quyết định 3 §3.6)

Chạy: python3 -m tests.test_safety_margin   (từ gốc code/)
      hoặc pytest tests/test_safety_margin.py
"""

import numpy as np

from core import safety_margin as mh


def test_drift_report_example_latency_1s():
    # Ví dụ số Chương 6: 1,5·1 + ½·1·1² = 2,0 m (bỏ qua e_bám)
    assert mh.delta_drift(1.5, 1.0, 1.0, tracking_error=0.0) == 2.0


def test_drift_report_example_latency_025s():
    # 1,5·0,25 + ½·1·0,25² = 0,40625 ≈ 0,41 m
    val = mh.delta_drift(1.5, 1.0, 0.25, tracking_error=0.0)
    assert abs(val - 0.40625) < 1e-12
    assert abs(val - 0.41) < 0.005


def test_latency_window_convention():
    # Chu kỳ đầu: chưa có kỳ trước → T_r = T_replan
    assert mh.latency_window(0.25, None) == 0.25
    # Có kỳ trước: cộng thời gian giải ĐO ĐƯỢC của kỳ trước
    assert mh.latency_window(0.25, 0.75) == 1.0
    assert mh.latency_window(0.25, 0.0) == 0.25


def test_replan_clock_uses_previous_cycle():
    clock = mh.ReplanClock(T_replan=0.25)
    assert clock.window() == 0.25  # chu kỳ 1
    clock.update(0.75)  # chu kỳ 1 giải mất 0,75 s
    assert clock.window() == 1.0  # chu kỳ 2 dùng số đo kỳ 1
    # và với T_r = 1,0 s này, δ_trôi khớp ví dụ 2,0 m của báo cáo
    assert mh.delta_drift(1.5, 1.0, clock.window(), tracking_error=0.0) == 2.0
    clock.update(0.10)
    assert abs(clock.window() - 0.35) < 1e-12


def test_sample_report_number():
    # Thông số trường hợp riêng của Bài 1: u_max = 4, vật cản vận tốc
    # không đổi (a_max = 0), Δt = 0,25 → (4+0)·0,25²/8 = 0,03125 m
    val = mh.delta_sample(4.0, 0.0, 0.25)
    assert abs(val - 0.03125) < 1e-12
    assert abs(val - 0.031) < 5e-4  # ≈ 0,031 m của báo cáo
    # Trường hợp tổng quát và tỉ lệ Δt²
    assert abs(mh.delta_sample(4.0, 1.0, 0.25) - 5 * 0.0625 / 8) < 1e-12
    assert (
        abs(mh.delta_sample(4.0, 0.0, 0.5) - 4 * mh.delta_sample(4.0, 0.0, 0.25))
        < 1e-12
    )


def test_legacy_mode_matches_paper1():
    cfg = mh.MarginConfig(mode="legacy")
    assert mh.legacy_extra_margin(cfg) == 0.03
    sp = np.array([0.10, 0.20])
    sr = np.array([0.05, 0.10])
    rho = mh.legacy_rho(sp, sr)  # κ = 2 mặc định
    assert np.allclose(rho, 2.0 * np.sqrt(sp**2 + sr**2))
    assert abs(rho[0] - 2.0 * np.sqrt(0.0125)) < 1e-12


def test_margin_components_and_planner_composition():
    K = 28
    cfg = mh.MarginConfig(mode="h1", T_replan=0.25)
    rho_eps = np.linspace(0.05, 0.40, K + 1)  # ρ(ε) giả lập tăng theo k
    comp = mh.margin_components(cfg, v_hat=0.9, a_max=0.0, T_r=0.5, rho_eps=rho_eps)
    dd = mh.delta_drift(0.9, 0.0, 0.5, cfg.tracking_error)
    ds = mh.delta_sample(4.0, 0.0, 0.25)
    assert comp["delta_drift"] == dd and comp["delta_sample"] == ds
    assert np.allclose(comp["margin_per_k"], dd + ds + rho_eps)
    assert np.allclose(comp["b_per_k"], 0.5 + dd + ds + rho_eps)
    # Tích hợp planner: need = R + d_safe + 0 + rho_planner == R + b
    rho_pl = mh.compose_rho_for_planner(rho_eps, dd, ds)
    R = 1.0
    assert np.allclose(R + cfg.d_safe + rho_pl, R + comp["b_per_k"])


def test_traj_quantile_planner_composition():
    # Đường tích hợp CHÍNH (Q4, giao thức §4.2): quantile mức quỹ
    # đạo là vô hướng, áp cho mọi k; ngưỡng planner khớp b của H1.
    K = 28
    cfg = mh.MarginConfig(mode="h1", T_replan=0.25)
    dd = mh.delta_drift(0.9, 0.0, 0.5, cfg.tracking_error)
    ds = mh.delta_sample(4.0, 0.0, 0.25)
    rho_traj = 2.5
    rho_pl = mh.compose_rho_traj_for_planner(rho_traj, dd, ds, K=K)
    assert rho_pl.shape == (K + 1,)
    assert np.allclose(rho_pl, rho_traj + dd + ds)
    comp = mh.margin_components(
        cfg, v_hat=0.9, a_max=0.0, T_r=0.5, rho_eps=rho_traj, K=K
    )
    R = 1.0
    assert np.allclose(R + cfg.d_safe + rho_pl, R + comp["b_per_k"])


def test_tracking_error_status_flag():
    # e_bám CHƯA ĐO (chờ SITL): cờ phải phản ánh đúng trạng thái này;
    # khi đã đo và đổi cờ, sửa test này cùng lúc.
    assert mh.TRACKING_ERROR_MEASURED is False
    assert mh.MarginConfig().tracking_error == mh.TRACKING_ERROR_DEFAULT


def test_estimate_speed():
    c = np.zeros((29, 3))
    c[:, 1] = 0.9 * 0.25 * np.arange(29)  # chuyển động 0,9 m/s
    assert abs(mh.estimate_speed(c, 0.25) - 0.9) < 1e-12
    assert mh.estimate_speed(c[:1], 0.25) == 0.0


# ------------------------------------------------------------------
# DV-H7 — trần cửa sổ trễ T_cap (Quyết định 3 §3.3/§3.6). Ba tính
# chất §3.6 được khóa bằng test:
#   T_r^hiệu-lực = min(Δt + T_latency(kỳ trước), T_cap);
#   chu kỳ đầu T_r^hiệu-lực = Δt;  cờ bật ⟺ T_r > T_cap.
# ------------------------------------------------------------------
def test_dvh7_effective_window_formula():
    # T_r = 0,25 + 10,0 = 10,25 → hiệu lực = min(10,25; 2,0) = 2,0
    assert mh.effective_latency_window(0.25, 10.0, 2.0) == 2.0
    # Dưới trần: hiệu lực = T_r nguyên vẹn
    assert mh.effective_latency_window(0.25, 0.75, 2.0) == 1.0
    # T_cap = None → không chặn
    assert mh.effective_latency_window(0.25, 10.0, None) == 10.25
    # Qua ReplanClock: trần áp SAU khi cửa sổ chu-kỳ-trước xác định —
    # window() vẫn trả số đo thật không chặn.
    clock = mh.ReplanClock(T_replan=0.25, T_cap=mh.T_CAP_DRAFT)
    clock.update(10.0)
    assert clock.window() == 10.25
    assert clock.effective_window() == 2.0


def test_dvh7_first_cycle_effective_window_is_dt():
    clock = mh.ReplanClock(T_replan=0.25, T_cap=2.0)
    assert clock.window() == 0.25
    assert clock.effective_window() == 0.25  # = Δt, chưa có kỳ trước
    assert clock.capped() is False


def test_dvh7_capped_flag_iff():
    clock = mh.ReplanClock(T_replan=0.25, T_cap=2.0)
    clock.update(1.0)  # T_r = 1,25 < 2,0
    assert clock.capped() is False
    assert clock.effective_window() == 1.25
    clock.update(1.75)  # T_r = 2,0 == T_cap
    assert clock.capped() is False  # bằng trần KHÔNG bật cờ
    assert clock.effective_window() == 2.0
    clock.update(1.76)  # T_r = 2,01 > T_cap
    assert clock.capped() is True
    assert clock.effective_window() == 2.0
    # Không bật trần: cờ không bao giờ bật, dù T_r rất lớn
    clock_nc = mh.ReplanClock(T_replan=0.25)
    clock_nc.update(500.0)
    assert clock_nc.capped() is False
    assert clock_nc.effective_window() == clock_nc.window() == 500.25


def test_dvh7_window_info_interface():
    # Giao diện ổn định cho adapter/runner ghi cờ drift_capped:
    # khóa {T_r, T_r_eff, capped, T_cap} — T_r là số KHÔNG chặn.
    clock = mh.ReplanClock(T_replan=0.25, T_cap=2.0)
    clock.update(141.6)  # đỉnh tranh chấp (DV-H7)
    info = clock.window_info()
    assert set(info) == {"T_r", "T_r_eff", "capped", "T_cap"}
    assert info["T_r"] == 141.85 and info["T_r_eff"] == 2.0
    assert info["capped"] is True and info["T_cap"] == 2.0
    # δ_trôi tại trần khớp căn cứ (iii) của Quyết định 3 §3.3:
    # δ_trôi(2,0; v̂=1,5; a=1) = 5,0 m
    assert mh.delta_drift(1.5, 1.0, info["T_r_eff"], tracking_error=0.0) == 5.0


def test_dvh7_default_clock_unchanged():
    # Mặc định (không truyền T_cap) không chặn: cửa sổ hiệu lực luôn
    # bằng T_r.
    clock = mh.ReplanClock(T_replan=0.25)
    assert clock.T_cap is None
    assert clock.window() == 0.25
    clock.update(0.75)
    assert clock.window() == 1.0
    assert clock.effective_window() == 1.0
    clock.update(0.10)
    assert abs(clock.window() - 0.35) < 1e-12
    assert abs(clock.effective_window() - 0.35) < 1e-12


if __name__ == "__main__":
    fns = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")
