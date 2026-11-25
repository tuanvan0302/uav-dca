# -*- coding: utf-8 -*-
"""
Test kiểm chứng cài đặt ba baseline Tier-1 — theo đúng mục kiểm
chứng bắt buộc ở cuối từng SPEC:

  SPEC-BL-NGO-2023 §7: (1) underestimator của g; (2) thu gọn về
    Bài 1 ở vòng đầu (‖ΔP‖∞ ≤ 1e-4 m, |Δtotal_slack| ≤ 1e-4 m);
    (3) hội tụ kiểu Ngo trên vật động cắt ngang (max s, δ_m → 0);
    (4) khả thi cuối tại mọi nút khi success.
  SPEC-BL-CBF-2021 §6: (1) (14) ⇒ (11d) trên lưới mẫu; (2) bất
    biến tiến / phục hồi theo đúng tốc độ γ; (3) dừng bởi tiêu chí
    (δ_viol, δ_cost) trước trần 50 vòng; (4) dấu hạng tử hằng số
    của (17) — đẳng thức hồi quy tại x_k = x_{k+1} = z_{k+1}.
  SPEC-BL-MPCQP §3: (1) không vật cản ⇒ trùng dca_planner.plan
    tới 1e-4 m; (2) tham chiếu đã an toàn ⇒ nghiệm không vi phạm
    khe hở thật tại nút quá 1e-6 m; (3) đúng 1 lần gọi solver mỗi
    plan(); (4) tham chiếu xuyên tâm vật cản: cắt suy biến được
    ghi cờ, kết quả tái lập giữa hai lần chạy.
  SPEC-FAIR-EXP4 §14: ba phương pháp khác nhau nhận mảng dự báo
    đóng băng GIỐNG HỆT nhau (checksum sha256 của (O, R, RHO)
    bằng nhau theo từng chu kỳ) trong cùng một episode.

Chạy:  python3 -m baselines.test_baselines     (từ gốc code/)
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from baselines import cbf2021, common, mpcqp, ngo2023
from core import dca_planner
from core.dca_planner import PlannerParams
from core.obstacle_predict import truth_obstacles
from sim import sim_realtime
from sim.dynamic_scenarios import dynamic_scenario

PRM = PlannerParams()


# ---------------------------------------------------------------- Ngo §7
def test_ngo_underestimator():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        o = rng.normal(size=3) * 3
        xhat = o + rng.normal(size=3)
        x = o + rng.normal(size=3)
        g_hat = np.linalg.norm(xhat - o)
        if g_hat < 1e-9:
            continue
        grad = (xhat - o) / g_hat
        under = g_hat + grad @ (x - xhat)
        worst = max(worst, under - np.linalg.norm(x - o))
    assert worst <= 1e-9, f"underestimator bị vi phạm: {worst}"
    print(
        f"  [Ngo §7.1] underestimator: g(x) ≥ tuyến tính hoá tại mọi mẫu "
        f"(dư lớn nhất {worst:.2e} ≤ 1e-9)"
    )


def _high_precision_solve_patch():
    """Thay solver của MỌI bài toán con bằng Clarabel 1e-10 trong
    một lần gọi plan() — dùng để tách nhiễu dung sai OSQP khỏi việc
    kiểm chứng cơ chế ở test Ngo §7.2 (xem DV-B3)."""
    orig_solve = cp.Problem.solve

    def hi_prec(self, *a, **kw):
        for k in ("solver", "eps_abs", "eps_rel", "max_iter", "warm_start"):
            kw.pop(k, None)
        return orig_solve(
            self,
            *a,
            solver=cp.CLARABEL,
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
            **kw,
        )

    cp.Problem.solve = hi_prec
    return orig_solve


def test_ngo_first_iteration_matches_paper1():
    """SPEC Ngo §7.2, hai tầng (lý do tách tầng: DV-B3):

    Tầng cơ chế — hai bài toán con vòng đầu được giải ở độ chính
    xác cao (Clarabel 1e-10, cùng một cách dựng bài toán của hai
    module thật): phải thỏa dung sai của SPEC, ‖ΔP‖∞ ≤ 1e-4 m và
    |Δtotal_slack| ≤ 1e-4 m. Đây là tầng kiểm chứng khẳng định
    "bài toán con của Ngo thu về bài toán con của Bài 1".

    Tầng solver sản xuất — OSQP eps 1e-4 (đóng băng bởi
    SPEC_cong_bang §7): nghiệm của hai module được phép lệch nhau
    ở thang dung sai solver; đo thực tế 1,85e-4 m / 4,4e-4 m trên
    máy này, khẳng định ở đây ở mức 5e-4 và số đo được ghi vào
    DV-B3 — KHÔNG nới dung sai của SPEC một cách im lặng.
    """
    p0 = np.array([0.0, 0.0, 2.0])
    v0 = np.zeros(3)
    goal = np.array([6.0, 0.0, 2.0])
    obs = [(np.array([3.0, 1.2, 2.0]), 0.5)]  # need = 1,03; đường thẳng cách 1,2 m
    prm1 = PlannerParams(max_iter=1)  # Bài 1: đúng vòng đầu
    cfg = ngo2023.NgoConfig(tau0=10.0, max_iterations=1)  # cùng lịch phạt

    orig = _high_precision_solve_patch()
    try:
        r_bai1_hp = dca_planner.plan(p0, v0, goal, obs, prm1)
        r_ngo_hp = ngo2023.plan(p0, v0, goal, obs, PRM, cfg=cfg)
    finally:
        cp.Problem.solve = orig
    assert r_bai1_hp.iterations == 1 and r_ngo_hp.iterations == 1
    dP_hp = float(np.max(np.abs(r_ngo_hp.P - r_bai1_hp.P)))
    dS_hp = abs(r_ngo_hp.total_slack - r_bai1_hp.total_slack)
    assert dP_hp <= 1e-4, f"[cơ chế] ‖ΔP‖∞ = {dP_hp} > 1e-4"
    assert dS_hp <= 1e-4, f"[cơ chế] |Δtotal_slack| = {dS_hp} > 1e-4"

    r_bai1 = dca_planner.plan(p0, v0, goal, obs, prm1)
    r_ngo = ngo2023.plan(p0, v0, goal, obs, PRM, cfg=cfg)
    dP = float(np.max(np.abs(r_ngo.P - r_bai1.P)))
    dS = abs(r_ngo.total_slack - r_bai1.total_slack)
    assert dP <= 5e-4, f"[OSQP] ‖ΔP‖∞ = {dP} > 5e-4 (thang dung sai solver)"
    assert dS <= 5e-4, f"[OSQP] |Δtotal_slack| = {dS} > 5e-4"
    print(
        f"  [Ngo §7.2] vòng đầu ≡ Bài 1 — cơ chế (Clarabel 1e-10): "
        f"‖ΔP‖∞ = {dP_hp:.2e} m, |Δslack| = {dS_hp:.2e} m (≤ 1e-4); "
        f"sản xuất (OSQP 1e-4): ‖ΔP‖∞ = {dP:.2e} m, |Δslack| = "
        f"{dS:.2e} m — thang dung sai solver, ghi DV-B3"
    )


def test_ngo_convergence_dynamic():
    sc = dynamic_scenario("D1_cat_ngang")
    obs = truth_obstacles(sc["truth_fns"], 2.2, PRM.K, PRM.dt)
    p0 = np.array([2.5, 0.0, 1.5])
    v0 = np.array([1.2, 0.0, 0.0])
    goal = sc["goal"]
    r = ngo2023.plan(p0, v0, goal, obs, PRM)
    assert r.meta["stop_reason"] != "cap", "chạm trần 200 vòng"
    assert r.success, f"không thành công: slack={r.total_slack}"
    assert r.total_slack <= 1e-3, f"Σ s cuối = {r.total_slack} > 1e-3"
    # §7.4: khả thi cuối tại mọi nút
    O, R, RHO = common.forecast_arrays(obs, PRM.K)
    need = common.need_array(R, RHO, PRM)
    dist = common.node_distances(r.P, O)
    viol = float(np.max(need[:, 1:] - dist[:, 1:]))
    assert viol <= 1e-3, f"vi phạm nút lớn nhất {viol} > 1e-3"
    print(
        f"  [Ngo §7.3/7.4] vật động cắt ngang: hội tụ sau "
        f"{r.iterations} vòng ({r.meta['stop_reason']}), Σ s = "
        f"{r.total_slack:.2e}, δ_m cuối = {r.meta['delta_m_final']:.2e}, "
        f"vi phạm nút ≤ {viol:.2e} m, time_to_first_feasible = "
        f"{r.meta['time_to_first_feasible']:.2f} s"
    )


def test_ngo_sensitivity_s_n1_smoke():
    p0 = np.array([0.0, 0.0, 1.5])
    goal = np.array([10.0, 0.0, 1.5])
    obs = [(np.array([5.0, 0.9, 1.5]), 1.0)]
    cfg = ngo2023.NgoConfig(use_original_objective=True, max_iterations=3)
    r = ngo2023.plan(p0, np.zeros(3), goal, obs, PRM, cfg=cfg)
    assert np.all(np.isfinite(r.P)) and r.iterations >= 1
    print(
        f"  [Ngo S-N1] mục tiêu gốc (5) qua SOCP/{r.meta['solver']}: "
        f"chạy được, {r.iterations} vòng, slack = {r.total_slack:.2e}"
    )


# ---------------------------------------------------------------- CBF §6
def test_cbf_lemma2_implication():
    rng = np.random.default_rng(1)
    gam = 0.9
    checked = 0
    tries = 0
    while checked < 500 and tries < 20000:
        tries += 1
        c = rng.normal(size=3) * 2
        need = float(rng.uniform(0.5, 2.0))
        z = c + rng.normal(size=3)
        x1 = c + rng.normal(size=3)
        # B = vế phải của (14) trừ γ‖x_k − c‖²: chọn x_k thỏa (14) với s = 0
        B = np.sum((z - c) ** 2) + 2.0 * (z - c) @ (x1 - z) + need**2 * (gam - 1.0)
        if B < 0.05:
            continue
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        xk = c + np.sqrt(B / gam) * 0.9 * w  # γ‖x_k − c‖² = 0,81·B ≤ B
        lhs = cbf2021.cbf_constraint_lhs(xk, x1, z, c, need, gam)
        assert lhs <= 1e-9, f"(14) không thỏa với s=0: lhs={lhs}"
        h_next = np.sum((x1 - c) ** 2) - need**2
        h_cur = np.sum((xk - c) ** 2) - need**2
        assert h_next >= gam * h_cur - 1e-9, (
            f"(14) thỏa nhưng (11d) sai: {h_next} < γ·{h_cur}"
        )
        checked += 1
    assert checked == 500
    print("  [CBF §6.1] (14) ⇒ (11d): đúng trên 500 mẫu ngẫu nhiên (Bổ đề 2)")


def test_cbf_constant_term_sign():
    rng = np.random.default_rng(2)
    for _ in range(200):
        c = rng.normal(size=3) * 2
        z = c + rng.normal(size=3)
        need = float(rng.uniform(0.3, 2.5))
        gam = 0.9
        lhs = cbf2021.cbf_constraint_lhs(z, z, z, c, need, gam)
        expect = (gam - 1.0) * (np.sum((z - c) ** 2) - need**2)
        assert abs(lhs - expect) <= 1e-12, f"{lhs} != {expect}"
    print(
        "  [CBF §6.4] dấu hạng tử hằng số (17): lhs(z,z,z) = (γ−1)·h(z) "
        "tới 1e-12 trên 200 mẫu"
    )


def _cbf_h_series(P, O, need):
    return np.array(
        [np.sum((P[k] - O[0, k]) ** 2) - need[0, k] ** 2 for k in range(P.shape[0])]
    )


def test_cbf_forward_invariance_and_recovery():
    gam = 0.9
    # (a) h(p0) > 0: quỹ đạo giữ h ≥ 0 (tới dung sai biến dư) và đệ quy đúng.
    obs = [(np.array([5.0, 0.9, 1.5]), 1.0)]
    d = cbf2021._plan_detailed(
        np.array([0.0, 0.0, 1.5]), np.zeros(3), np.array([10.0, 0.0, 1.5]), obs, PRM
    )
    assert d["success"], f"ca (a) không thành công: {d['meta']}"
    h = _cbf_h_series(d["P"], d["O"], d["need"])
    assert h[0] > 0
    assert np.min(h) >= -0.02, f"h âm quá dung sai: {np.min(h)}"
    s = d["S"][0]
    rec = np.min(h[1:] - (gam * h[:-1] - s))
    assert rec >= -1e-5, f"đệ quy CBF vi phạm: {rec}"
    # (b) h(p0) < 0 nhẹ: h phục hồi về 0 theo đúng tốc độ γ (đệ quy với s).
    obs2 = [(np.array([5.0, 0.0, 1.5]), 1.0)]
    d2 = cbf2021._plan_detailed(
        np.array([4.2, 0.0, 1.5]), np.zeros(3), np.array([10.0, 0.0, 1.5]), obs2, PRM
    )
    h2 = _cbf_h_series(d2["P"], d2["O"], d2["need"])
    assert h2[0] < 0
    s2 = d2["S"][0]
    rec2 = np.min(h2[1:] - (gam * h2[:-1] - s2))
    assert rec2 >= -1e-5, f"đệ quy phục hồi vi phạm: {rec2}"
    assert h2[-1] > h2[0], "h không phục hồi về phía tập an toàn"
    print(
        f"  [CBF §6.2] bất biến tiến: min h = {np.min(h):.3f} ≥ 0 (ca a); "
        f"phục hồi: h {h2[0]:.2f} → {h2[-1]:.2f} (ca b), đệ quy "
        f"h_{{k+1}} ≥ γ·h_k − s_k đúng tới 1e-5"
    )


def test_cbf_stops_by_criteria():
    obs = [(np.array([5.0, 0.9, 1.5]), 1.0)]
    r = cbf2021.plan(
        np.array([0.0, 0.0, 1.5]), np.zeros(3), np.array([10.0, 0.0, 1.5]), obs, PRM
    )
    assert r.meta["stop_reason"] == "converged", (
        f"dừng bởi {r.meta['stop_reason']} thay vì tiêu chí (δ_viol, δ_cost)"
    )
    assert r.iterations <= 50
    print(
        f"  [CBF §6.3] dừng bởi tiêu chí (δ_viol, δ_cost) sau "
        f"{r.iterations} vòng (trần 50; bài gốc ~8 vòng ở Fig. 3b), "
        f"solver nón = {r.meta['solver_conic']}"
    )


# ---------------------------------------------------------------- MPC-QP §3
def test_mpcqp_no_obstacle_matches_paper1():
    p0 = np.array([0.0, 0.0, 2.0])
    goal = np.array([5.0, 0.0, 2.0])
    r_mpc = mpcqp.plan(p0, np.zeros(3), goal, [], PRM)
    r_b1 = dca_planner.plan(p0, np.zeros(3), goal, [], PRM)
    dP = float(np.max(np.abs(r_mpc.P - r_b1.P)))
    assert r_mpc.success and dP <= 1e-4, f"‖ΔP‖∞ = {dP} > 1e-4"
    print(
        f"  [MPC §3.1] không vật cản: ‖P_MPCQP − P_Bài1‖∞ = {dP:.2e} m (dung sai 1e-4)"
    )


def test_mpcqp_safe_reference():
    obs = [(np.array([5.0, 2.5, 1.5]), 1.0)]  # tham chiếu thẳng cách tâm 2,5 m
    r = mpcqp.plan(
        np.array([0.0, 0.0, 1.5]), np.zeros(3), np.array([10.0, 0.0, 1.5]), obs, PRM
    )
    assert r.success
    O, R, RHO = common.forecast_arrays(obs, PRM.K)
    node_clear = np.min(common.node_distances(r.P, O) - R)
    assert node_clear >= -1e-6, f"khe hở nút {node_clear} < −1e-6"
    print(
        f"  [MPC §3.2] tham chiếu an toàn: khe hở nút nhỏ nhất {node_clear:.3f} m ≥ 0"
    )


def test_mpcqp_exactly_one_solve():
    orig_solve = cp.Problem.solve
    calls = {"n": 0}

    def counting(self, *a, **kw):
        calls["n"] += 1
        return orig_solve(self, *a, **kw)

    cp.Problem.solve = counting
    try:
        mpcqp.plan(
            np.array([0.0, 0.0, 1.5]),
            np.zeros(3),
            np.array([10.0, 0.0, 1.5]),
            [(np.array([5.0, 2.5, 1.5]), 1.0)],
            PRM,
        )
        n_ok = calls["n"]
        calls["n"] = 0
        r2 = mpcqp.plan(
            np.array([0.0, 0.0, 1.5]),
            np.zeros(3),
            np.array([10.0, 0.0, 1.5]),
            [(np.array([5.0, 0.0, 1.5]), 1.2)],
            PRM,
        )
        n_inf = calls["n"]
    finally:
        cp.Problem.solve = orig_solve
    assert n_ok == 1 and n_inf == 1, f"số lần gọi solver: {n_ok}/{n_inf} != 1"
    print(
        f"  [MPC §3.3] đúng 1 lần gọi solver mỗi plan() "
        f"(ca tối ưu: {n_ok}, ca xuyên vật: {n_inf}); "
        f"ca xuyên vật success={r2.success}, qp_infeasible="
        f"{r2.meta['qp_infeasible']}"
    )


def test_mpcqp_reference_through_obstacle():
    obs = [(np.array([5.0, 0.0, 1.5]), 1.2)]  # q_14 của đường thẳng = tâm vật
    args = (
        np.array([0.0, 0.0, 1.5]),
        np.zeros(3),
        np.array([10.0, 0.0, 1.5]),
        obs,
        PRM,
    )
    r1 = mpcqp.plan(*args)
    r2 = mpcqp.plan(*args)
    assert r1.meta["degenerate_cut"] and r1.meta["n_degenerate_cuts"] >= 1
    assert np.array_equal(r1.P, r2.P) and r1.success == r2.success
    print(
        f"  [MPC §3.4] tham chiếu xuyên tâm: {r1.meta['n_degenerate_cuts']} "
        f"cắt suy biến bị bỏ + ghi cờ; hai lần chạy cho kết quả "
        f"trùng từng bit (success={r1.success})"
    )


# ---------------------------------------------------------------- Công bằng §14
def test_frozen_forecast_checksum_shared():
    seen = {}

    def make_recorder(mod, tag):
        def rec(p0, v0, goal, obstacles, prm, seed_P=None):
            O, R, RHO = common.forecast_arrays(obstacles, prm.K)
            seen.setdefault(tag, []).append(common.forecast_checksum(O, R, RHO))
            return mod.plan(p0, v0, goal, obstacles, prm, seed_P=seed_P)

        return rec

    orig = sim_realtime.plan
    try:
        for tag, mod in (("mpcqp", mpcqp), ("ngo", ngo2023), ("cbf", cbf2021)):
            sim_realtime.plan = make_recorder(mod, tag)
            sim_realtime.run_episode(
                "D1_cat_ngang",
                mode="kalman",
                seed=9000000,
                time_model="frozen",
                t_max=2.0,
            )
    finally:
        sim_realtime.plan = orig
    a, b, c = seen["mpcqp"], seen["ngo"], seen["cbf"]
    n = min(len(a), len(b), len(c))
    assert n >= 8, f"quá ít chu kỳ để đối chiếu: {n}"
    assert a[:n] == b[:n] == c[:n], (
        "checksum dự báo đóng băng khác nhau giữa các phương pháp"
    )
    print(
        f"  [FAIR §14] dự báo đóng băng: checksum (O, R, RHO) của MPC-QP, "
        f"Ngo-2023 và CBF-2021 trùng nhau ở cả {n} chu kỳ đầu (sha256 "
        f"{a[0][:12]}…)"
    )


def main() -> int:
    tests = [
        test_ngo_underestimator,
        test_ngo_first_iteration_matches_paper1,
        test_ngo_convergence_dynamic,
        test_ngo_sensitivity_s_n1_smoke,
        test_cbf_lemma2_implication,
        test_cbf_constant_term_sign,
        test_cbf_forward_invariance_and_recovery,
        test_cbf_stops_by_criteria,
        test_mpcqp_no_obstacle_matches_paper1,
        test_mpcqp_safe_reference,
        test_mpcqp_exactly_one_solve,
        test_mpcqp_reference_through_obstacle,
        test_frozen_forecast_checksum_shared,
    ]
    t0 = time.perf_counter()
    for fn in tests:
        fn()
    print(
        f"\nALL {len(tests)} BASELINE TESTS PASSED ({time.perf_counter() - t0:.1f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
