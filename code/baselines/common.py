# -*- coding: utf-8 -*-
"""Thành phần dùng chung cho ba baseline Tier-1 (SPEC-FAIR-EXP4 v1.1).

Mọi thứ ở đây đều là "luật chơi chung" của SPEC_cong_bang, không phải
cơ chế riêng của phương pháp nào:

  - Chuẩn hoá mảng dự báo ĐÓNG BĂNG (O, R, RHO) bằng đúng hàm
    normalize_obstacles của dca_planner — mọi phương pháp (kể cả
    Bài 1) suy ra cùng các mảng từ cùng đầu vào `obstacles`, và
    checksum của bộ ba mảng được ghi vào metadata từng lần gọi plan()
    để kiểm chứng tính đóng băng (checklist §14 của SPEC công bằng).
  - Ngưỡng an toàn need_{m,k} = R + d_safe + extra_margin + RHO (§4.2).
  - Hàm mục tiêu chung J của Bài 1 (SPEC_cong_bang §3):
    J = Σ_k [α‖u_k‖² + β‖u_{k+1} − u_k‖²] + γ‖p_K − goal‖², với hạng
    tử β là sum_squares(U[1:] − U[:-1]) đúng biểu thức code Bài 1.
  - Động học tích phân kép chính xác (ZOH), hộp giới hạn theo từng
    thành phần, sàn CỨNG cho baseline (SPEC_cong_bang §10 — khác
    Bài 1/ST-DCA vốn phạt sàn bằng biến dư; lệch D-6 đã đăng ký).
  - Dung sai solver chốt bằng số (SPEC_cong_bang §7): OSQP
    eps_abs = eps_rel = 1e-4, max_iter = 20000, warm_start = True
    (import thẳng dca_planner.OSQP_OPTS — KHÔNG phải mặc định
    1e-3 của OSQP); ECOS abstol = reltol = feastol = 1e-8. Nếu ECOS
    không có trong môi trường: Clarabel với tol_* = 1e-8 và việc
    thay thế được ghi vào metadata + Sổ lệch (DV-B2).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from core.dca_planner import OSQP_OPTS, PlanResult, normalize_obstacles

# --- Dung sai solver (SPEC_cong_bang §7 — chốt bằng số, cấm "mặc định").
# OSQP_OPTS lấy từ dca_planner để baseline và Bài 1 không thể lệch nhau;
# các baseline dùng nó qua common.OSQP_OPTS.
ECOS_OPTS = dict(abstol=1e-8, reltol=1e-8, feastol=1e-8)
CLARABEL_OPTS = dict(tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8)

# Ngưỡng khả thi phía bộ đánh giá (SPEC_cong_bang §9.1): vi phạm
# khoảng cách THẬT > 1e-3 m tại một nút ⇒ chu kỳ thất bại. Dùng cho
# chỉ số time_to_first_feasible theo định nghĩa evaluator.
FEAS_TOL_M = 1e-3


@dataclass
class BaselineResult(PlanResult):
    """PlanResult của Bài 1 + túi metadata của baseline.

    Các trường chuẩn (P, V, U, success, min_clearance, total_slack,
    iterations, solve_time, obj_history) giữ đúng ngữ nghĩa của
    dca_planner.PlanResult để vòng lặp mô phỏng dùng không phân
    biệt. `meta` mang các chỉ số phụ bắt buộc của
    SPEC_cong_bang §12 (time_to_first_feasible, cờ chạm trần, lý do
    dừng, solver + dung sai thực dùng, checksum dự báo đóng băng,
    cờ success nội bộ kèm đơn vị…).
    """

    meta: dict = field(default_factory=dict)


def forecast_arrays(obstacles, K: int):
    """Suy (O, R, RHO) từ đầu vào obstacles bằng đúng hàm của Bài 1.

    Baseline chỉ nhận vật cản dạng cầu (SPEC_cong_bang §4.2): đầu
    vào ellipsoid (khoá 'axes') bị từ chối tường minh thay vì âm
    thầm quy về cầu.
    """
    O, R, RHO, ELL = normalize_obstacles(obstacles, K)
    if any(e is not None for e in ELL):
        raise ValueError(
            "baselines accept spherical obstacles only (SPEC_cong_bang §4.2)"
        )
    return O, R, RHO


def forecast_checksum(O: np.ndarray, R: np.ndarray, RHO: np.ndarray) -> str:
    """sha256 của bộ ba mảng dự báo đóng băng (O, R, RHO)."""
    h = hashlib.sha256()
    for arr in (O, R, RHO):
        a = np.ascontiguousarray(arr, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def need_array(R: np.ndarray, RHO: np.ndarray, prm) -> np.ndarray:
    """need_{m,k} = R + d_safe + extra_margin + RHO (SPEC_cong_bang §4.2)."""
    return R + prm.d_safe + prm.extra_margin + RHO


def straight_seed(p0: np.ndarray, goal: np.ndarray, K: int) -> np.ndarray:
    """Hạt giống lạnh: nội suy thẳng p0 → goal theo chỉ số bước —
    đúng cách dca_planner dựng hạt giống mặc định (np.linspace)."""
    ts = np.linspace(0.0, 1.0, K + 1)[:, None]
    return p0[None, :] * (1 - ts) + goal[None, :] * ts


def initial_trajectory(p0, goal, K, seed_P):
    """Quỹ đạo khởi tạo theo SPEC_cong_bang §5: ấm = seed_P do
    harness cấp (nghiệm chu kỳ trước đã dịch thời gian); lạnh (hoặc
    seed sai dạng) = nội suy thẳng."""
    if seed_P is not None and np.shape(seed_P) == (K + 1, 3):
        return np.array(seed_P, dtype=float)
    return straight_seed(np.asarray(p0, dtype=float), np.asarray(goal, dtype=float), K)


def add_dynamics(cons, P, V, U, p0, v0, dt: float):
    """(R1) tích phân kép chính xác + (R2) điều kiện đầu."""
    K = U.shape[0]
    cons.append(P[0] == np.asarray(p0, dtype=float))
    cons.append(V[0] == np.asarray(v0, dtype=float))
    for k in range(K):
        cons.append(P[k + 1] == P[k] + dt * V[k] + 0.5 * dt * dt * U[k])
        cons.append(V[k + 1] == V[k] + dt * U[k])


def add_limits_hard_floor(cons, P, V, U, prm):
    """Hộp vận tốc/gia tốc theo từng thành phần (SPEC_cong_bang §2)
    + sàn CỨNG p^(z) >= z_floor cho baseline (§10, lệch D-6)."""
    cons.append(cp.abs(V) <= prm.v_max)
    cons.append(cp.abs(U) <= prm.u_max)
    cons.append(P[:, 2] >= prm.z_floor)


def objective_expr(P, U, goal, prm):
    """Biểu thức CVXPY của J chung (đúng biểu thức code Bài 1)."""
    K = P.shape[0] - 1
    return (
        prm.alpha * cp.sum_squares(U)
        + prm.beta * cp.sum_squares(U[1:] - U[:-1])
        + prm.gamma * cp.sum_squares(P[K] - np.asarray(goal, dtype=float))
    )


def eval_objective(P, U, goal, prm) -> float:
    """J chung tính bằng số trên một quỹ đạo đã có."""
    K = U.shape[0]
    cost = prm.alpha * float(np.sum(U**2))
    if K > 1:
        cost += prm.beta * float(np.sum(np.diff(U, axis=0) ** 2))
    cost += prm.gamma * float(np.sum((P[-1] - np.asarray(goal, dtype=float)) ** 2))
    return cost


def node_distances(P, O) -> np.ndarray:
    """‖p_k − ô_{m,k}‖ cho mọi (m, k): mảng (M, K+1)."""
    return np.linalg.norm(P[None, :, :] - O, axis=2)


def true_violation_sum(P, O, need) -> float:
    """Σ_{m, k>=1} max(0, need − ‖p_k − ô‖): vi phạm THẬT tại các nút
    (đơn vị m) — đại lượng phía bộ đánh giá (SPEC_cong_bang §9.1)."""
    if O.shape[0] == 0:
        return 0.0
    dist = node_distances(P, O)
    return float(np.sum(np.maximum(0.0, need[:, 1:] - dist[:, 1:])))


def min_clearance_nodes(P, O, R) -> float:
    """Khe hở thật nhỏ nhất tại các nút, như PlanResult của Bài 1:
    min_{m,k} (‖p_k − ô_{m,k}‖ − R_{m,k}); +inf khi không có vật cản."""
    if O.shape[0] == 0:
        return float("inf")
    return float(np.min(node_distances(P, O) - R))


def stationary_result(p0, K: int) -> tuple:
    """Quỹ đạo đứng yên tại p0 — kết quả dự phòng khi solver thất bại
    hoàn toàn (cùng quy ước dca_planner khi P_best is None)."""
    P = np.repeat(np.asarray(p0, dtype=float)[None, :], K + 1, axis=0)
    V = np.zeros((K + 1, 3))
    U = np.zeros((K, 3))
    return P, V, U


def solve_osqp(prob) -> bool:
    """Giải QP bằng OSQP với dung sai tường minh của SPEC_cong_bang §7.
    Trả True nếu có nghiệm (optimal / optimal_inaccurate)."""
    try:
        prob.solve(solver=cp.OSQP, **OSQP_OPTS)
    except Exception:
        return False
    return prob.status in ("optimal", "optimal_inaccurate")


def conic_solver_choice() -> tuple:
    """Chọn solver nón cho QCQP/SOCP: ECOS nếu môi trường có (bài
    gốc Varun), ngược lại Clarabel — thay thế mà SPEC_cong_bang §7
    cho phép, kèm dung sai bằng số. Trả (solver, opts, tên)."""
    installed = cp.installed_solvers()
    if "ECOS" in installed:
        return cp.ECOS, dict(ECOS_OPTS), "ECOS"
    return cp.CLARABEL, dict(CLARABEL_OPTS), "CLARABEL"


def solve_conic(prob) -> tuple:
    """Giải bài toán nón bằng solver đã chọn. Trả (ok, tên solver)."""
    solver, opts, name = conic_solver_choice()
    try:
        prob.solve(solver=solver, **opts)
    except Exception:
        return False, name
    return prob.status in ("optimal", "optimal_inaccurate"), name


class FirstFeasibleTracker:
    """Ghi time_to_first_feasible (SPEC_cong_bang §6.2/§12): vòng
    đầu tiên và thời gian wall-clock (từ đầu lần gọi plan) tại đó
    quỹ đạo đạt khả thi theo ngưỡng cho trước."""

    def __init__(self, t_start: float):
        """Khởi tạo bộ theo dõi; t_start là time.perf_counter() lúc bắt
        đầu plan()."""
        self.t_start = t_start
        self.iteration = None
        self.time_s = None

    def update(self, iteration: int, violation: float, tol: float) -> bool:
        """Ghi nhận vòng `iteration` có vi phạm `violation` (m).

        Returns:
            True đúng một lần: ở vòng đầu tiên violation ≤ tol. Các vòng
            sau trả False và không ghi đè mốc đã lưu.
        """
        if self.iteration is None and violation <= tol:
            self.iteration = int(iteration)
            self.time_s = float(time.perf_counter() - self.t_start)
            return True
        return False

    def as_dict(self) -> dict:
        """Hai trường metadata của §12; None nếu chưa từng khả thi."""
        return dict(
            time_to_first_feasible=self.time_s,
            iterations_to_first_feasible=self.iteration,
        )
