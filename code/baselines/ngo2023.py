# -*- coding: utf-8 -*-
"""Baseline Tier-1: penalty-DCA của Ngo et al. (CDC 2023).

Đặc tả: SPEC-BL-NGO-2023 v1.1.
Bài gốc: A. P. Ngo et al., "Multi-UAV trajectory planning problem
using the difference of convex function programming", IEEE CDC 2023,
DOI 10.1109/CDC49753.2023.10383259 (arXiv:2303.07581v3).

Cơ chế giữ đúng bài gốc (Mục V của bài): tái phát biểu dạng phạt
[P_τ] của ràng buộc DC, bài toán con lồi hoá bằng tuyến tính hoá g
tại MỌI điểm đã thăm (tập O TÍCH LŨY KHÔNG TRẦN, không loại bỏ —
khác Bài 1 vốn giữ tối đa 6 điểm/ràng buộc và có mặt cắt điểm giữa;
cả hai thứ đó đều KHÔNG có ở đây, M-10/M-11), lịch phạt
τ ← min(µ·τ, τ̄) tăng MỖI vòng lặp.

Ánh xạ sang harness Bài 1 (bảng §3 của SPEC — mọi lệch đều đã đăng
ký D-2a…D-2e trong SPEC_cong_bang §11): 1 UAV + vật cản ảo theo dự
báo chung đóng băng; động học tích phân kép chính xác (ZOH) thay
Euler tiến; J chung của Bài 1 thay mục tiêu gốc (5) (mục tiêu gốc
chỉ ở cấu hình nhạy cảm S-N1); ngưỡng tâm–tâm d thay bằng
need_{m,k} = R + d_safe + extra_margin + RHO; bỏ điều kiện cuối
cứng (6d)–(6f) (giữ cứng vị trí cuối ở nhạy cảm S-N2); sàn cứng
(SPEC_cong_bang §10).

Tham số (§5.1 của SPEC): τ₀ = 1 (Q-N1 — lưu ý Bài 1 dùng τ₀ = 10,
chênh lệch này phải công bố, D-5), µ = 4, τ̄ = 2×10⁴ (Q-N2, bằng
Bài 1), ε = 1e-6 tương đối (Q-N3), trần 200 vòng (Q-N4), điều kiện
dừng §5.2 gồm nhánh khả thi (0) tự thêm vì đối xứng với Bài 1
(Σ s ≤ 1e-3 ∧ δ_m ≤ δ_loose = 1e-2 tương đối, Q-N5) — nhánh (0) và
vế Σ s của nhánh (ii) KHÔNG có trong Bước 4 gốc (A-3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from baselines import common
from baselines.common import BaselineResult

CONFIG_ID = "SPEC-BL-NGO-2023 v1.1 main"


@dataclass
class NgoConfig:
    """Siêu tham số riêng của Penalty-DCA Ngo et al. (SPEC-BL-NGO-2023 v1.1)."""

    tau0: float = 1.0  # Q-N1 (Bài 1: 10 — lệch D-5 phải công bố)
    mu: float = 4.0  # Q-N2 — bằng Bài 1
    tau_max: float = 2.0e4  # Q-N2 — bằng Bài 1
    eps: float = 1e-6  # Q-N3 — δ_m tương đối (nhánh dừng (i), gốc)
    delta_loose: float = 1e-2  # Q-N5 — δ_m tương đối cho nhánh khả thi (0)
    slack_tol: float = 1e-3  # m — ngưỡng Σ s khả thi (cùng slack_tol của Bài 1)
    max_iterations: int = 200  # Q-N4 — chạm trần phải gắn cờ hit_cap
    # --- Cấu hình nhạy cảm (chạy phụ, báo cáo riêng — §5.3 của SPEC)
    use_original_objective: bool = False  # S-N1: mục tiêu gốc (5), SOCP
    rho_f: float = 1.0  # S-N1: ρf (tự quyết; bài gốc không cho số)
    rho_a: float = 0.1  # S-N1: a trong ρk = a·k (quyết định)
    hard_terminal_position: bool = False  # S-N2: p_K = goal (chỉ khi đích tới được)


def _original_objective_expr(P, U, goal, cfg: NgoConfig):
    """Mục tiêu gốc (5) của Ngo: Σ_k [ρf·‖F_k‖₂ + ρk·‖p_k − p^g‖₂],
    ρk = a·k, k = 1..K; F ≡ u vì m = 1 kg (M-3). Chuẩn KHÔNG bình
    phương ⇒ bài toán con là SOCP."""
    K = U.shape[0]
    goal = np.asarray(goal, dtype=float)
    expr = 0.0
    for k in range(K):
        expr = expr + cfg.rho_f * cp.norm(U[k], 2)
        expr = expr + (cfg.rho_a * (k + 1)) * cp.norm(P[k + 1] - goal, 2)
    return expr


def _build_and_solve(P_lin_points, O, need, p0, v0, goal, prm, tau, cfg):
    """Dựng và giải MỘT bài toán con [P_τ^m] (pt. (16)–(18) của bài
    gốc, đã ánh xạ). P_lin_points: danh sách các quỹ đạo vị trí đã
    thăm (tập O tích lũy). Trả (ok, P, V, U, sum_slack_solver,
    conic_name) — conic_name là tên solver nón thực dùng ở S-N1
    (None ở đường OSQP) để ghi metadata theo SPEC_cong_bang §7/§12."""
    K, dt = prm.K, prm.dt
    M = O.shape[0]
    P = cp.Variable((K + 1, 3))
    V = cp.Variable((K + 1, 3))
    U = cp.Variable((K, 3))
    S = cp.Variable((M, K)) if M else None  # s_{m,k}, k = 1..K (cột k−1)
    cons: list = []
    common.add_dynamics(cons, P, V, U, p0, v0, dt)
    common.add_limits_hard_floor(cons, P, V, U, prm)
    if cfg.hard_terminal_position:
        cons.append(P[K] == np.asarray(goal, dtype=float))
    if M:
        cons.append(S >= 0)
        for m in range(M):
            for k in range(1, K + 1):
                o = O[m, k]
                nd = need[m, k]
                for xhat in P_lin_points:
                    dvec = xhat[k] - o
                    nrm = float(np.linalg.norm(dvec))
                    if nrm < 1e-9:
                        # A-4: ∇g không xác định khi hai tâm trùng —
                        # bỏ mặt cắt này ở vòng này (như dca_planner).
                        continue
                    unit = dvec / nrm
                    cons.append(unit @ (P[k] - o) >= nd - S[m, k - 1])
    if cfg.use_original_objective:
        obj_expr = _original_objective_expr(P, U, goal, cfg)
    else:
        obj_expr = common.objective_expr(P, U, goal, prm)
    if M:
        obj_expr = obj_expr + tau * cp.sum(S)
    prob = cp.Problem(cp.Minimize(obj_expr), cons)
    if cfg.use_original_objective:
        ok, conic_name = common.solve_conic(prob)
    else:
        ok = common.solve_osqp(prob)
        conic_name = None
    if not ok or P.value is None:
        return False, None, None, None, None, conic_name
    # Σ s*_m của bài toán con; kẹp tại 0: s ≥ 0 theo xây dựng, giá
    # trị âm ~1e-6 chỉ là nhiễu dung sai solver (OSQP eps 1e-4).
    sum_s = float(np.sum(np.maximum(S.value, 0.0))) if M else 0.0
    return (
        True,
        np.asarray(P.value),
        np.asarray(V.value),
        np.asarray(U.value),
        sum_s,
        conic_name,
    )


def plan(p0, v0, goal, obstacles, prm, seed_P=None, cfg: NgoConfig | None = None):
    """plan() của baseline Ngo-2023 — giao diện khớp dca_planner.plan.

    Trả BaselineResult: total_slack = Σ s_{m,k} của bài toán con ở
    vòng cuối (đơn vị m, đúng §4 của SPEC); cờ success nội bộ theo
    cùng quy tắc với dca_planner.plan:
    total_slack ≤ max(slack_tol, 1e-2) — chỉ là metadata, KHÔNG
    phải chỉ số so sánh (SPEC_cong_bang §9.1).
    """
    t_start = time.perf_counter()
    cfg = cfg or NgoConfig()
    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    goal = np.asarray(goal, dtype=float)
    K = prm.K
    O, R, RHO = common.forecast_arrays(obstacles, K)
    need = common.need_array(R, RHO, prm)
    checksum = common.forecast_checksum(O, R, RHO)

    # Bước 1 gốc: O := {x̂⁰}; khởi tạo theo SPEC_cong_bang §5 (A-6/Q-N/A-6).
    xhat = common.initial_trajectory(p0, goal, K, seed_P)
    lin_points = [xhat]
    tau = cfg.tau0

    ttf_solver = common.FirstFeasibleTracker(t_start)  # Σ s ≤ 1e-3 (SPEC §5.2)
    # Vi phạm thật ≤ 1e-3 m (phía evaluator)
    ttf_true = common.FirstFeasibleTracker(t_start)
    obj_history: list = []
    prev_pen_obj = None
    delta_m = np.inf
    stop_reason = "cap"
    hit_cap = False
    solver_fail = False
    iterations = 0
    P_best = V_best = U_best = None
    total_slack = np.inf
    conic_solver_name = None

    for it in range(1, cfg.max_iterations + 1):
        ok, Pn, Vn, Un, sum_s, sname = _build_and_solve(
            lin_points, O, need, p0, v0, goal, prm, tau, cfg
        )
        if sname is not None:
            conic_solver_name = sname
        if not ok:
            solver_fail = True
            stop_reason = "solver_fail"
            break
        iterations = it
        P_best, V_best, U_best = Pn, Vn, Un
        total_slack = float(sum_s)
        # δ_m (Bước 4 gốc): khe hở của MỤC TIÊU PHẠT giữa hai vòng,
        # f0 là J chung (hoặc mục tiêu gốc ở S-N1) + τ_m·Σ s*_m;
        # chuẩn hoá tương đối theo Q-N3.
        if cfg.use_original_objective:
            f0 = float(
                sum(
                    cfg.rho_f * np.linalg.norm(Un[k])
                    + cfg.rho_a * (k + 1) * np.linalg.norm(Pn[k + 1] - goal)
                    for k in range(K)
                )
            )
        else:
            f0 = common.eval_objective(Pn, Un, goal, prm)
        pen_obj = f0 + tau * total_slack
        obj_history.append(pen_obj)
        if prev_pen_obj is not None:
            delta_m = abs(pen_obj - prev_pen_obj) / max(1.0, abs(prev_pen_obj))
        prev_pen_obj = pen_obj
        ttf_solver.update(it, total_slack, cfg.slack_tol)
        ttf_true.update(it, common.true_violation_sum(Pn, O, need), common.FEAS_TOL_M)
        # --- Điều kiện dừng §5.2 (bản chốt v1.1)
        if total_slack <= cfg.slack_tol and delta_m <= cfg.delta_loose:
            stop_reason = "feasible_loose"  # (0) — tự thêm vì công bằng (A-3)
            break
        if delta_m <= cfg.eps:
            stop_reason = "delta_eps"  # (i) — tiêu chí gốc
            break
        if tau >= cfg.tau_max and total_slack <= cfg.slack_tol:
            stop_reason = "tau_cap_feasible"  # (ii) — gốc + vế Σ s tự thêm (A-3)
            break
        # Bước 3 gốc: tích lũy điểm mới (không trần, không loại bỏ);
        # τ tăng MỖI vòng.
        lin_points.append(Pn.copy())
        tau = min(cfg.mu * tau, cfg.tau_max)
    else:
        hit_cap = True
        stop_reason = "cap"

    if P_best is None:  # solver thất bại ngay vòng đầu
        P_best, V_best, U_best = common.stationary_result(p0, K)
        total_slack = np.inf
    solve_time = time.perf_counter() - t_start
    success = bool(total_slack <= max(cfg.slack_tol, 1e-2))
    meta = dict(
        method="Ngo2023",
        config_id=CONFIG_ID,
        internal_success_flag=success,
        internal_success_unit="m",
        internal_success_threshold=max(cfg.slack_tol, 1e-2),
        tau0=cfg.tau0,
        mu=cfg.mu,
        tau_max=cfg.tau_max,
        tau_final=float(tau),
        eps_relative=cfg.eps,
        delta_loose=cfg.delta_loose,
        slack_tol_m=cfg.slack_tol,
        max_iterations=cfg.max_iterations,
        stop_reason=stop_reason,
        hit_cap=bool(hit_cap),
        solver_fail=bool(solver_fail),
        delta_m_final=float(delta_m),
        n_lin_points_final=len(lin_points),
        solver=("CONIC" if cfg.use_original_objective else "OSQP"),
        solver_conic=(conic_solver_name if cfg.use_original_objective else None),
        solver_tolerances=(
            (common.ECOS_OPTS if conic_solver_name == "ECOS" else common.CLARABEL_OPTS)
            if cfg.use_original_objective
            else common.OSQP_OPTS
        ),
        true_violation_sum_m=common.true_violation_sum(P_best, O, need),
        forecast_checksum=checksum,
        sensitivity=dict(
            use_original_objective=cfg.use_original_objective,
            rho_f=cfg.rho_f,
            rho_a=cfg.rho_a,
            hard_terminal_position=cfg.hard_terminal_position,
        ),
        **ttf_solver.as_dict(),
        time_to_first_feasible_true_m=ttf_true.time_s,
        iterations_to_first_feasible_true_m=ttf_true.iteration,
    )
    return BaselineResult(
        P=P_best,
        V=V_best,
        U=U_best,
        success=success,
        min_clearance=common.min_clearance_nodes(P_best, O, R),
        total_slack=float(total_slack),
        iterations=int(iterations),
        solve_time=solve_time,
        obj_history=obj_history,
        meta=meta,
    )
