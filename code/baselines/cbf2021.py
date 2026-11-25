# -*- coding: utf-8 -*-
"""Baseline Tier-1: Convexified-CBF của Varun, Vinod & Kolathaya (ICC 2021).

Đặc tả: SPEC-BL-CBF-2021 v1.1.
Bài gốc: "Motion Planning With Dynamic Obstacles Using Convexified
Control Barrier Functions", Proc. ICC 2021, tr. 81–86.

Cơ chế giữ đúng bài gốc: ràng buộc CBF rời rạc h(x_{k+1}) ≥ γ·h(x_k)
với h_{m,k} = ‖p_k − ô_{m,k}‖² − need_{m,k}² (pt. (10)/(11d)) được
viết thành ràng buộc DC (Bổ đề 1, pt. (12)) rồi lồi hoá bằng tuyến
tính hoá hạng tử lồi thứ hai tại điểm z hiện tại (Bổ đề 2, pt.
(14)) — nghiệm của (14) thì thỏa (11d). Vì (14) là xấp xỉ trong,
bài toán con nới lỏng bằng biến dư + phạt kiểu Lipp & Boyd (pt.
(17), Thuật toán 1): z ← nghiệm mới mỗi vòng (CHỈ giữ điểm hiện
tại, không tích lũy như Ngo/Bài 1), τ ← min(μ·τ, τ_max) mỗi vòng,
dừng khi Σ s ≤ δ_viol VÀ |Δcost| ≤ δ_cost (điều kiện while in
ngược dấu ở bài gốc là lỗi in — A-1).

Ánh xạ sang harness (bảng §2 của SPEC — lệch D-3/D-4): chân trời
28×0,25 s thay 15×0,1 s; r_i hằng → need_{m,k} theo k; THÔNG TIN
THẬT của bài gốc → dự báo Kalman chung đóng băng (M-8 — cấm dùng
sự thật nền, SPEC_cong_bang §4.3); mục tiêu (11a) → J chung (gốc ở
nhạy cảm S-C2); tâm trong (11d) theo NGUYÊN VĂN c_{i,k} cho cả hai
vế (M-7/A-4 — biến thể S-C1 dùng ô_{m,k+1} cho vế x_{k+1}).

Tham số: μ = 4, τ_max = 10⁵, δ_cost = 0,1, δ_viol = 0,001 lấy
nguyên văn Mục IV của bài gốc; γ = 0,9 và τ₀ = 1 là TỰ CHỌN (bài
gốc không cho số — D-4), lưới nhạy cảm chỉ trên tập hiệu chuẩn.
cost trong tiêu chí δ_cost = J KHÔNG gồm hạng tử phạt (quyết định
tái cài đặt A-8 — bài gốc không định nghĩa; cách đọc ngược lại là
cấu hình nhạy cảm S-C3). Đơn vị biến dư: m² (A-6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from baselines import common
from baselines.common import BaselineResult

CONFIG_ID = "SPEC-BL-CBF-2021 v1.1 main"


@dataclass
class CbfConfig:
    """Siêu tham số riêng của Convexified-CBF (SPEC-BL-CBF-2021 v1.1)."""

    # TỰ CHỌN (D-4); lưới {0.5, 0.8, 0.9, 0.95, 0.99} trên tập hiệu chuẩn
    gamma: float = 0.9
    mu: float = 4.0  # nguyên văn Mục IV
    # TỰ CHỌN (A-2/Q-C1); nhạy cảm {0.1, 1, 10} trên tập hiệu chuẩn
    tau0: float = 1.0
    tau_max: float = 1.0e5  # nguyên văn Mục IV
    # Nguyên văn Mục IV — tuyệt đối trên J không phạt (A-5)
    delta_cost: float = 0.1
    delta_viol: float = 1e-3  # nguyên văn Mục IV — trên Σ s, đơn vị m² (A-6)
    # Trần an toàn tự thêm (A-7); bài gốc ≤ ~8 vòng trong thực nghiệm
    max_iterations: int = 50
    # --- Cấu hình nhạy cảm (§4.3 của SPEC — báo cáo riêng)
    cost_includes_penalty: bool = False  # S-C3: cost := J + τ·Σs (A-8)
    # S-C1: vế x_{k+1} dùng ô_{m,k+1}, need_{m,k+1}
    time_varying_center_next: bool = False
    use_original_objective: bool = False  # S-C2: (11a), Q vị trí = I, R = 0,1·I


def cbf_constraint_lhs(x_k, x_k1, z_k1, center, need_k, gamma) -> float:
    """Vế trái của (17) dạng số — dùng cho test hồi quy dấu (mục 6.4
    của SPEC) và kiểm chứng (14) ⇒ (11d):

        γ‖x_k − c‖² − ‖z_{k+1} − c‖² − 2(z_{k+1} − c)ᵀ(x_{k+1} − z_{k+1})
            − need²(γ − 1)   ≤  s_{k}
    """
    x_k = np.asarray(x_k, dtype=float)
    x_k1 = np.asarray(x_k1, dtype=float)
    z_k1 = np.asarray(z_k1, dtype=float)
    c = np.asarray(center, dtype=float)
    return float(
        gamma * np.sum((x_k - c) ** 2)
        - np.sum((z_k1 - c) ** 2)
        - 2.0 * float((z_k1 - c) @ (x_k1 - z_k1))
        - need_k**2 * (gamma - 1.0)
    )


def _solve_init_qp(p0, v0, goal, prm):
    """QP khởi tạo (16'): min J s.t. động học + giới hạn + sàn, BỎ
    hoàn toàn ràng buộc an toàn (cơ chế gốc M-9). Trả (P, V, U) hoặc
    None nếu solver thất bại."""
    K, dt = prm.K, prm.dt
    P = cp.Variable((K + 1, 3))
    V = cp.Variable((K + 1, 3))
    U = cp.Variable((K, 3))
    cons: list = []
    common.add_dynamics(cons, P, V, U, p0, v0, dt)
    common.add_limits_hard_floor(cons, P, V, U, prm)
    prob = cp.Problem(cp.Minimize(common.objective_expr(P, U, goal, prm)), cons)
    if not common.solve_osqp(prob) or P.value is None:
        return None
    return np.asarray(P.value), np.asarray(V.value), np.asarray(U.value)


def _original_objective_expr(P, U, goal):
    """(11a) cho S-C2: Q = I trên vị trí (0 trên vận tốc), R = 0,1·I
    (quyết định — bài gốc không cho Q, R; A-3)."""
    goal = np.asarray(goal, dtype=float)
    return cp.sum_squares(P - goal[None, :]) + 0.1 * cp.sum_squares(U)


def _solve_qcqp(z_P, O, need, p0, v0, goal, prm, tau, cfg: CbfConfig):
    """Giải MỘT QCQP nới lỏng (17'). Trả (ok, P, V, U, S, solver_name)."""
    K, dt = prm.K, prm.dt
    M = O.shape[0]
    gam = cfg.gamma
    P = cp.Variable((K + 1, 3))
    V = cp.Variable((K + 1, 3))
    U = cp.Variable((K, 3))
    S = cp.Variable((M, K)) if M else None  # s_{m,k}, k = 0..K−1
    cons: list = []
    common.add_dynamics(cons, P, V, U, p0, v0, dt)
    common.add_limits_hard_floor(cons, P, V, U, prm)
    if M:
        cons.append(S >= 0)
        for m in range(M):
            for k in range(K):
                o_k = O[m, k]
                nd_k = need[m, k]
                z1 = z_P[k + 1]
                if cfg.time_varying_center_next:
                    # S-C1: h(x_{k+1}) dùng tâm/ngưỡng tại k+1:
                    # γ‖p_k − ô_k‖² − lin_{z}(‖p_{k+1} − ô_{k+1}‖²)
                    #   − γ·need_k² + need_{k+1}² ≤ s
                    o_n = O[m, k + 1]
                    nd_n = need[m, k + 1]
                    lin_const = float(np.sum((z1 - o_n) ** 2))
                    lin_vec = 2.0 * (z1 - o_n)
                    lhs = (
                        gam * cp.sum_squares(P[k] - o_k)
                        - lin_const
                        - lin_vec @ (P[k + 1] - z1)
                        - gam * nd_k**2
                        + nd_n**2
                    )
                else:
                    # Nguyên văn (M-7): cả hai vế dùng ô_{m,k}, need_{m,k}.
                    lin_const = float(np.sum((z1 - o_k) ** 2))
                    lin_vec = 2.0 * (z1 - o_k)
                    lhs = (
                        gam * cp.sum_squares(P[k] - o_k)
                        - lin_const
                        - lin_vec @ (P[k + 1] - z1)
                        - nd_k**2 * (gam - 1.0)
                    )
                cons.append(lhs <= S[m, k])
    if cfg.use_original_objective:
        obj_expr = _original_objective_expr(P, U, goal)
    else:
        obj_expr = common.objective_expr(P, U, goal, prm)
    if M:
        obj_expr = obj_expr + tau * cp.sum(S)
    prob = cp.Problem(cp.Minimize(obj_expr), cons)
    ok, solver_name = common.solve_conic(prob)
    if not ok or P.value is None:
        return False, None, None, None, None, solver_name
    return (
        True,
        np.asarray(P.value),
        np.asarray(V.value),
        np.asarray(U.value),
        np.asarray(S.value) if M else np.zeros((0, K)),
        solver_name,
    )


def _plan_detailed(
    p0, v0, goal, obstacles, prm, seed_P=None, cfg: CbfConfig | None = None
) -> dict:
    """Chạy thuật toán CBF và trả dict chi tiết (kể cả mảng S cuối)
    — plan() là lớp vỏ trả BaselineResult; test kiểm chứng §6 của
    SPEC dùng bản chi tiết này."""
    t_start = time.perf_counter()
    cfg = cfg or CbfConfig()
    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    goal = np.asarray(goal, dtype=float)
    K = prm.K
    O, R, RHO = common.forecast_arrays(obstacles, K)
    need = common.need_array(R, RHO, prm)
    checksum = common.forecast_checksum(O, R, RHO)

    # Bước 1 gốc: z từ QP (16') không ràng buộc an toàn; z_0 = p_0 do
    # điều kiện đầu của chính QP đó. seed_P của harness KHÔNG dùng:
    # cơ chế khởi tạo của CBF-2021 là một phần định nghĩa phương
    # pháp (SPEC_cong_bang §5.1).
    init = _solve_init_qp(p0, v0, goal, prm)
    if init is None:
        z_P = common.straight_seed(p0, goal, K)
        z_V = np.zeros((K + 1, 3))
        z_U = np.zeros((K, 3))
        init_solver_fail = True
    else:
        z_P, z_V, z_U = init
        init_solver_fail = False

    tau = cfg.tau0
    sum_slacks = np.inf
    delta_cost = np.inf
    cost_p = np.inf
    iterations = 0
    stop_reason = "cap"
    hit_cap = False
    solver_fail = init_solver_fail
    solver_name = None
    S_last = None
    obj_history: list = []
    ttf_true = common.FirstFeasibleTracker(t_start)

    for it in range(1, cfg.max_iterations + 1):
        ok, Pn, Vn, Un, Sn, solver_name = _solve_qcqp(
            z_P, O, need, p0, v0, goal, prm, tau, cfg
        )
        if not ok:
            # A-7: solver fail ⇒ trả z hiện tại, success=False.
            solver_fail = True
            stop_reason = "solver_fail"
            break
        iterations = it
        z_P, z_V, z_U = Pn, Vn, Un
        S_last = Sn
        # Σ s (m²); kẹp tại 0 như ngo2023 (nhiễu dung sai solver).
        sum_slacks = float(np.sum(np.maximum(Sn, 0.0))) if Sn.size else 0.0
        if cfg.use_original_objective:
            base_cost = float(np.sum((Pn - goal[None, :]) ** 2) + 0.1 * np.sum(Un**2))
        else:
            base_cost = common.eval_objective(Pn, Un, goal, prm)
        cost = base_cost + (tau * sum_slacks if cfg.cost_includes_penalty else 0.0)
        obj_history.append(base_cost + tau * sum_slacks)
        delta_cost = cost - cost_p
        cost_p = cost
        ttf_true.update(it, common.true_violation_sum(Pn, O, need), common.FEAS_TOL_M)
        tau = min(cfg.mu * tau, cfg.tau_max)
        # A-1: dừng khi Σ s ≤ δ_viol VÀ |Δcost| ≤ δ_cost.
        if sum_slacks <= cfg.delta_viol and abs(delta_cost) <= cfg.delta_cost:
            stop_reason = "converged"
            break
    else:
        hit_cap = True
        stop_reason = "cap"

    solve_time = time.perf_counter() - t_start
    total_slack = float(sum_slacks) if np.isfinite(sum_slacks) else np.inf
    if iterations == 0 and solver_fail and S_last is None:
        total_slack = np.inf
    # Cờ nội bộ (SPEC §3): total_slack ≤ δ_viol; nhưng A-7: solver
    # fail ⇒ success=False kể cả khi z hiện tại đã ít vi phạm.
    success = bool(total_slack <= cfg.delta_viol and not solver_fail)
    meta = dict(
        method="CBF2021",
        config_id=CONFIG_ID,
        internal_success_flag=success,
        internal_success_unit="m^2",
        internal_success_threshold=cfg.delta_viol,
        gamma_cbf=cfg.gamma,
        mu=cfg.mu,
        tau0=cfg.tau0,
        tau_max=cfg.tau_max,
        tau_final=float(tau),
        delta_cost=cfg.delta_cost,
        delta_viol_m2=cfg.delta_viol,
        delta_cost_final=float(delta_cost),
        cost_definition=(
            "J_plus_penalty" if cfg.cost_includes_penalty else "J_without_penalty"
        ),
        max_iterations=cfg.max_iterations,
        stop_reason=stop_reason,
        hit_cap=bool(hit_cap),
        solver_fail=bool(solver_fail),
        n_init_qp=1,
        init_qp_solver_fail=bool(init_solver_fail),
        solver_conic=solver_name,
        solver_tolerances=(
            common.ECOS_OPTS if solver_name == "ECOS" else common.CLARABEL_OPTS
        ),
        sum_slacks_m2=float(total_slack) if np.isfinite(total_slack) else None,
        max_slack_m2=(
            float(np.max(S_last)) if S_last is not None and S_last.size else None
        ),
        true_violation_sum_m=common.true_violation_sum(z_P, O, need),
        forecast_checksum=checksum,
        sensitivity=dict(
            cost_includes_penalty=cfg.cost_includes_penalty,
            time_varying_center_next=cfg.time_varying_center_next,
            use_original_objective=cfg.use_original_objective,
        ),
        **ttf_true.as_dict(),
    )
    return dict(
        P=z_P,
        V=z_V,
        U=z_U,
        S=S_last,
        success=success,
        total_slack=float(total_slack),
        iterations=int(iterations),
        solve_time=solve_time,
        obj_history=obj_history,
        min_clearance=common.min_clearance_nodes(z_P, O, R),
        meta=meta,
        O=O,
        R=R,
        need=need,
    )


def plan(p0, v0, goal, obstacles, prm, seed_P=None, cfg: CbfConfig | None = None):
    """plan() của baseline CBF-2021 — giao diện khớp dca_planner.plan.

    total_slack = Σ s_{m,k} vòng cuối, đơn vị m² (A-6); cờ success
    nội bộ = (total_slack ≤ δ_viol = 1e-3 m²) — chỉ là metadata,
    không cùng đơn vị với cờ của các planner khác và KHÔNG dùng
    làm chỉ số so sánh (SPEC_cong_bang §9.1).
    """
    d = _plan_detailed(p0, v0, goal, obstacles, prm, seed_P=seed_P, cfg=cfg)
    return BaselineResult(
        P=d["P"],
        V=d["V"],
        U=d["U"],
        success=d["success"],
        min_clearance=d["min_clearance"],
        total_slack=d["total_slack"],
        iterations=d["iterations"],
        solve_time=d["solve_time"],
        obj_history=d["obj_history"],
        meta=d["meta"],
    )
