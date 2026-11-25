# -*- coding: utf-8 -*-
"""Baseline mốc tham chiếu: MPC-QP tuyến tính hóa một lần.

Đặc tả: SPEC-BL-MPCQP v1.1. Đây
KHÔNG phải tái cài đặt một bài báo: đây là mốc tham chiếu dưới do
nhóm tự định nghĩa — MPC dùng ĐÚNG MỘT lần tuyến tính hóa ràng
buộc an toàn quanh quỹ đạo tham chiếu, giải ĐÚNG MỘT QP, và dừng.
Mọi cơ chế của ST-DCA/Bài 1 đều tắt ở đây: không vòng lặp DCA/CCP,
không biến dư, không phạt, không CBF, không tích lũy mặt cắt (mỗi
(m, k) đúng 1 mặt cắt tại điểm tham chiếu), không mặt cắt điểm
giữa, không A* seed, không khởi tạo lại khi thất bại.

Mặt cắt dùng đúng công thức mặt cắt (2) của Bài 1 tại điểm tham
chiếu q_k: (q_k − ô)ᵀ(p_k − ô)/‖q_k − ô‖ ≥ need_{m,k}, áp CỨNG.
Sàn cứng (SPEC_cong_bang §10). QP infeasible ⇒ trả chính quỹ đạo
tham chiếu q kèm cờ qp_infeasible (SPEC Bước 4): harness thực thi
bước đầu của q như mọi chu kỳ và bộ đánh giá gắn cờ chu kỳ thất
bại — điểm yếu có chủ đích của mốc tham chiếu phải lộ ra trong số
liệu, không được che bằng phanh dự phòng hay thử lại.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from baselines import common
from baselines.common import BaselineResult

CONFIG_ID = "SPEC-BL-MPCQP v1.1 main"


@dataclass
class MpcqpConfig:
    """Không có siêu tham số riêng (SPEC §2) — lớp này chỉ mang
    config_id để metadata đồng nhất với hai baseline còn lại."""

    pass


def _reference_trajectory(p0, goal, K, seed_P):
    """Bước 1 của SPEC: lạnh = nội suy thẳng p0 → goal theo chỉ số
    bước (như hạt giống mặc định của dca_planner); ấm = seed_P."""
    return common.initial_trajectory(p0, goal, K, seed_P)


def _derive_vel_acc_from_positions(q, dt: float, v0):
    """Suy (V, U) cho quỹ đạo tham chiếu q khi phải trả q (QP
    infeasible): V bằng hiệu hữu hạn trên lưới Δt (ép V[0] = v0 —
    điều kiện đầu là sự kiện, không phải lựa chọn), U = ΔV/Δt.
    Chỉ dùng cho đường dự phòng; đường tối ưu lấy V, U của QP."""
    V = np.gradient(q, dt, axis=0, edge_order=1)
    V[0] = np.asarray(v0, dtype=float)
    U = np.diff(V, axis=0) / dt
    return V, U


def plan(p0, v0, goal, obstacles, prm, seed_P=None, cfg: MpcqpConfig | None = None):
    """plan() của MPC-QP — giao diện khớp dca_planner.plan.

    success nội bộ = QP giải được (không phải ngưỡng biến dư —
    bảng ánh xạ cờ ở SPEC_cong_bang §9.1); total_slack = 0 khi QP
    tối ưu (không có biến dư trong phương pháp này); iterations = 1
    luôn luôn.
    """
    t_start = time.perf_counter()
    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    goal = np.asarray(goal, dtype=float)
    K, dt = prm.K, prm.dt
    O, R, RHO = common.forecast_arrays(obstacles, K)
    need = common.need_array(R, RHO, prm)
    checksum = common.forecast_checksum(O, R, RHO)
    M = O.shape[0]

    q = _reference_trajectory(p0, goal, K, seed_P)

    # Bước 2: đúng một mặt cắt cho mỗi (m, k), k = 1..K, tại q_k.
    P = cp.Variable((K + 1, 3))
    V = cp.Variable((K + 1, 3))
    U = cp.Variable((K, 3))
    cons: list = []
    common.add_dynamics(cons, P, V, U, p0, v0, dt)
    common.add_limits_hard_floor(cons, P, V, U, prm)
    n_cuts = 0
    n_degenerate = 0
    for m in range(M):
        for k in range(1, K + 1):
            o = O[m, k]
            dvec = q[k] - o
            nrm = float(np.linalg.norm(dvec))
            if nrm < 1e-9:
                # Điểm tham chiếu trùng tâm: bỏ cắt, ghi cờ (SPEC Bước 2).
                n_degenerate += 1
                continue
            unit = dvec / nrm
            cons.append(unit @ (P[k] - o) >= need[m, k])
            n_cuts += 1

    # Bước 3: đúng MỘT QP.
    prob = cp.Problem(cp.Minimize(common.objective_expr(P, U, goal, prm)), cons)
    ok = common.solve_osqp(prob)
    solve_time = time.perf_counter() - t_start

    if ok and P.value is not None:
        Pn, Vn, Un = (np.asarray(P.value), np.asarray(V.value), np.asarray(U.value))
        success = True
        qp_infeasible = False
        total_slack = 0.0
        obj_history = [common.eval_objective(Pn, Un, goal, prm)]
        out_P, out_V, out_U = Pn, Vn, Un
    else:
        # Bước 4: trả quỹ đạo tham chiếu q, KHÔNG thử lại/nới lỏng.
        Vn, Un = _derive_vel_acc_from_positions(q, dt, v0)
        success = False
        qp_infeasible = True
        total_slack = 0.0
        obj_history = []
        out_P, out_V, out_U = q, Vn, Un

    meta = dict(
        method="MPCQP",
        config_id=CONFIG_ID,
        internal_success_flag=success,
        internal_success_unit=None,
        internal_success_threshold=None,
        qp_infeasible=bool(qp_infeasible),
        degenerate_cut=bool(n_degenerate > 0),
        n_degenerate_cuts=int(n_degenerate),
        n_cuts=int(n_cuts),
        n_solver_calls=1,
        solver="OSQP",
        true_violation_sum_m=common.true_violation_sum(out_P, O, need),
        forecast_checksum=checksum,
        time_to_first_feasible=(
            solve_time
            if success
            and common.true_violation_sum(out_P, O, need) <= common.FEAS_TOL_M
            else None
        ),
        iterations_to_first_feasible=(
            1
            if success
            and common.true_violation_sum(out_P, O, need) <= common.FEAS_TOL_M
            else None
        ),
    )
    return BaselineResult(
        P=out_P,
        V=out_V,
        U=out_U,
        success=success,
        min_clearance=common.min_clearance_nodes(out_P, O, R),
        total_slack=float(total_slack),
        iterations=1,
        solve_time=solve_time,
        obj_history=obj_history,
        meta=meta,
    )
