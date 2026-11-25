# -*- coding: utf-8 -*-
"""
Bộ giải DCA dạng phạt cho bài toán một UAV tránh vật cản.

Bài toán: tìm quỹ đạo K bước (bước thời gian dt) cực tiểu hoá
    J = Σ_k [ α‖u_k‖² + β‖u_{k+1} − u_k‖² ] + γ‖p_K − p_đích‖²
với các ràng buộc
    (R1) động học tích phân kép  p_{k+1} = p_k + dt·v_k + ½dt²·u_k,
                                  v_{k+1} = v_k + dt·u_k;
    (R2) trạng thái đầu (p_0, v_0) cho trước;
    (R3) |v_k,i| ≤ v_max, |u_k,i| ≤ u_max (từng thành phần), cao độ ≥ z_floor;
    (R4) an toàn  ‖p_k − o_mk‖ ≥ need_mk  với mọi vật cản m, bước k ≥ 1,
         need_mk = r_mk + d_safe + extra_margin + ρ_mk.

(R4) là ràng buộc DC: need − ‖p − o‖ ≤ 0 (hằng số trừ hàm lồi). DCA thay
‖p − o‖ bằng tiếp tuyến của nó tại một điểm đã thăm q:
    ‖p − o‖ ≥ e·(p − o),   e = (q − o)/‖q − o‖,
nên ràng buộc tuyến tính e·(p − o) ≥ need là xấp xỉ TRONG của (R4):
điểm nào thoả nó thì thoả (R4). Mỗi ràng buộc được nới bởi một biến dư
s_mk ≥ 0 bị phạt τ·Σs trong hàm mục tiêu, nên bài toán con luôn có
nghiệm và là một QP lồi (giải bằng OSQP qua CVXPY). Thuật toán lặp:
giải QP → thêm điểm vừa tìm được vào tập điểm tuyến tính hoá → giải
lại; khi hết vòng mà vẫn còn vi phạm thì tăng τ (phạt chính xác).

Vật cản có thể là cầu hoặc ellipsoid phụ thuộc thời gian. Với ellipsoid,
‖p − o‖ được thay bằng a_min·g(p), trong đó g là hàm gauge của
ellipsoid (xem ellipsoid.py); mặt cắt có cùng dạng tuyến tính và thu về
đúng mặt cắt cầu khi ba bán trục bằng nhau.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from core.ellipsoid import ellipsoid_series
from core.ellipsoid import interpolate as _ell_interpolate

# Dung sai OSQP cho mọi QP con của plan(). Đây là nguồn duy nhất của các
# giá trị này: baselines/common.py import lại để baseline giải QP với đúng
# dung sai của Bài 1 (SPEC_cong_bang §7). Cố ý KHÔNG dùng mặc định 1e-3
# của OSQP.
OSQP_OPTS = dict(eps_abs=1e-4, eps_rel=1e-4, max_iter=20000, warm_start=True)


@dataclass
class PlannerParams:
    """Tham số của bộ giải DCA; giá trị mặc định là cấu hình đang dùng.

    Các trường K, dt, alpha, beta, gamma, v_max, u_max, z_floor, d_safe,
    extra_margin được các baseline đọc chung (SPEC_cong_bang), nên đổi ý
    nghĩa của chúng là đổi luật chơi chung.
    """

    K: int = 28  # số bước chân trời
    dt: float = 0.25  # bước thời gian (s)
    alpha: float = 1.0  # trọng số năng lượng ‖u‖²
    beta: float = 0.5  # trọng số độ trơn ‖u_{k+1} − u_k‖²
    gamma: float = 30.0  # trọng số bám đích ở cuối chân trời
    v_max: float = 3.0  # giới hạn vận tốc từng thành phần (m/s)
    u_max: float = 4.0  # giới hạn gia tốc từng thành phần (m/s^2)
    z_floor: float = 0.8  # cao độ tối thiểu (m), áp dạng phạt
    d_safe: float = 0.5  # khoảng cách an toàn yêu cầu tới bề mặt vật cản (m)
    tau0: float = 10.0  # hệ số phạt ban đầu
    tau_factor: float = 4.0  # mỗi lần tăng phạt: tau ← tau·tau_factor
    tau_max: float = 2.0e4  # trần hệ số phạt
    max_iter: int = 15  # số vòng DCA tối đa cho mỗi mức phạt
    slack_tol: float = 1e-3  # tổng vi phạm (m) được coi là khả thi
    max_lin_points: int = 6  # số điểm tuyến tính hoá giữ lại cho mỗi (m, k)
    reset_lin_on_tau: bool = False  # True: khi tăng phạt, chỉ giữ điểm hiện tại
    # (R4) chỉ được áp tại các nút k; giữa hai nút quỹ đạo là cung
    # parabol và có thể võng vào vật cản. midpoint_cuts thêm ràng buộc
    # (R4) tại điểm giữa mỗi đoạn để chặn hiện tượng này.
    midpoint_cuts: bool = True
    # Đệm hằng số cộng vào ngưỡng (R4). 0,03 m xấp xỉ độ võng tối đa
    # của cung parabol giữa hai nút: u_max·dt²/8 = 4·0,25²/8 ≈ 0,031 m.
    extra_margin: float = 0.03


@dataclass
class PlanResult:
    """Kết quả trả về của plan() (và của các baseline, qua BaselineResult)."""

    P: np.ndarray  # (K+1, 3) vị trí, m
    V: np.ndarray  # (K+1, 3) vận tốc, m/s
    U: np.ndarray  # (K, 3) gia tốc lệnh, m/s²; phía gọi thực thi U[0]
    success: bool  # True nếu tổng vi phạm cuối ≤ max(slack_tol, 1e-2)
    # Khe hở nhỏ nhất tới bề mặt các vật cản ĐẦU VÀO (m), không trừ d, ρ.
    min_clearance: float
    total_slack: float  # tổng vi phạm thật (R4) + sàn của quỹ đạo trả về (m)
    iterations: int  # tổng số QP đã giải
    solve_time: float  # thời gian chạy plan() (s)
    obj_history: list = field(default_factory=list)  # mục tiêu phạt thật sau mỗi QP


def normalize_obstacles(obstacles: list, K: int) -> tuple:
    """Chuẩn hoá vật cản tĩnh / động / biến đổi hình dạng về mảng theo thời gian.

    Đây là hàm chuẩn hoá DUY NHẤT của gói: planner và mọi baseline đều suy
    ra (O, R, RHO) qua hàm này để mảng dự báo đóng băng giống hệt nhau.

    Args:
        obstacles: danh sách vật cản, mỗi phần tử ở một dạng bên dưới.
        K: số bước chân trời; mảng trả về có K+1 nút theo thời gian.

    Returns:
        (O, R, RHO, ELL): O (M, K+1, 3) tâm (m), R (M, K+1) bán kính (m),
        RHO (M, K+1) biên bất định (m), ELL danh sách độ dài M gồm dãy
        Ellipsoid theo từng bước, hoặc None với vật cầu.

    Mỗi phần tử của obstacles là một trong ba dạng:
      - (tâm (3,), bán kính): vật cản tĩnh, ρ = 0;
      - dict {'centers': (K+1, 3) hoặc (3,), 'radii': (K+1,) hoặc số,
              'rho': (K+1,) hoặc số, tuỳ chọn}: vật cầu theo thời gian;
      - dict {'centers': ..., 'axes': (K+1, 3) hoặc (3,),
              'yaws': (K+1,) hoặc số (mặc định 0), 'rho': tuỳ chọn}:
              vật ellipsoid theo thời gian. Khi đó R[m] chứa a_min (bán
              trục nhỏ nhất) tại từng bước, vì ngưỡng của (R4*) ở đơn vị
              mét là a_min + d_safe + đệm + ρ (xem ellipsoid.Ellipsoid.cut).
    Giá trị dạng hằng (không có trục thời gian) được lặp cho mọi bước.
    """
    M = len(obstacles)
    O = np.zeros((M, K + 1, 3))
    R = np.zeros((M, K + 1))
    RHO = np.zeros((M, K + 1))
    ELL = [None] * M
    for m, ob in enumerate(obstacles):
        if isinstance(ob, dict) and "axes" in ob:
            # Vật ellipsoid
            O[m] = _series_3d(ob["centers"], K)
            AX = _series_3d(ob["axes"], K)
            YW = _series_1d(ob.get("yaws", 0.0), K)
            ELL[m] = ellipsoid_series(O[m], AX, YW)
            R[m] = np.min(AX, axis=1)
            if "rho" in ob and ob["rho"] is not None:
                RHO[m] = _series_1d(ob["rho"], K)
        elif isinstance(ob, dict):
            # Vật cầu theo thời gian
            O[m] = _series_3d(ob["centers"], K)
            R[m] = _series_1d(ob["radii"], K)
            if "rho" in ob and ob["rho"] is not None:
                RHO[m] = _series_1d(ob["rho"], K)
        else:
            # Vật cầu tĩnh (tâm, bán kính)
            o, r = ob
            O[m] = np.asarray(o, dtype=float)[None, :].repeat(K + 1, axis=0)
            R[m] = float(r)
    return O, R, RHO, ELL


def _series_3d(value, K: int) -> np.ndarray:
    """Đưa một đại lượng 3 chiều về dạng (K+1, 3).

    Giá trị đã có trục thời gian (K+1, 3) được giữ nguyên; giá trị hằng
    (3,) được lặp cho mọi bước.
    """
    a = np.asarray(value, dtype=float)
    if a.shape == (K + 1, 3):
        return a
    return np.repeat(a.reshape(1, 3), K + 1, axis=0)


def _series_1d(value, K: int) -> np.ndarray:
    """Đưa một đại lượng vô hướng về dạng (K+1,).

    Giá trị đã có trục thời gian (K+1,) được giữ nguyên; giá trị hằng
    được lặp cho mọi bước.
    """
    a = np.asarray(value, dtype=float)
    if a.shape == (K + 1,):
        return a
    return float(a) * np.ones(K + 1)


# Vật cản tại thời điểm giữa hai bước liên tiếp (dùng cho ràng buộc điểm giữa)
def _midpoint_arrays(O, R, RHO, ELL=None):
    """Tâm, bán kính và biên bất định tại điểm giữa mỗi đoạn [k, k+1],
    nội suy tuyến tính giữa hai nút (chính xác khi vật cản chuyển động
    với vận tốc không đổi). Mỗi mảng có K phần tử theo trục thời gian.

    Với vật ellipsoid: nội suy chính ellipsoid (tâm và bán trục tuyến
    tính, yaw theo góc đã mở gói — ellipsoid.interpolate), lấy a_min của
    ellipsoid giữa làm R_mid và trả thêm ELL_mid (None nếu không có vật
    ellipsoid nào)."""
    O_mid = 0.5 * (O[:, :-1] + O[:, 1:])
    R_mid = 0.5 * (R[:, :-1] + R[:, 1:])
    RHO_mid = 0.5 * (RHO[:, :-1] + RHO[:, 1:])
    ELL_mid = None
    if ELL is not None and any(e is not None for e in ELL):
        K = O.shape[1] - 1
        ELL_mid = [None] * len(ELL)
        for m, series in enumerate(ELL):
            if series is None:
                continue
            ELL_mid[m] = [
                _ell_interpolate(series[k], series[k + 1], 0.5) for k in range(K)
            ]
            R_mid[m] = np.array([e.a_min for e in ELL_mid[m]])
    return O_mid, R_mid, RHO_mid, ELL_mid


def _scaled_gauge_series(series, X):
    """a_min·g(x) tại các điểm X (N, 3), điểm thứ i đo với ellipsoid series[i]."""
    return np.array([series[i].scaled_gauge(X[i]) for i in range(len(series))])


# Hàm mục tiêu phạt tính bằng khoảng cách THẬT (không tuyến tính hoá)
def _true_penalized_objective(
    P, V, U, goal, O, R, RHO, prm: PlannerParams, tau: float, ELL=None
) -> float:
    """Trả (J + tau·pen, pen), trong đó pen = Σ max(0, need − khoảng cách
    thật) trên mọi vật cản, tại các nút k ≥ 1 và (nếu bật midpoint_cuts)
    tại điểm giữa mỗi đoạn. Với vật ellipsoid, khoảng cách thật là
    a_min·g(p) (đơn vị mét, bằng ‖p − o‖ khi ellipsoid là cầu).
    pen là thước đo vi phạm dùng cho điều kiện dừng của plan()."""
    K, dt = prm.K, prm.dt
    # Hàm mục tiêu gốc J
    cost = prm.alpha * float(np.sum(U**2))
    if K > 1:
        cost += prm.beta * float(np.sum(np.diff(U, axis=0) ** 2))
    cost += prm.gamma * float(np.sum((P[-1] - goal) ** 2))
    # Vi phạm (R4) tại các nút k = 1..K (nút 0 là trạng thái đầu, không ràng buộc)
    pen = 0.0
    for m in range(O.shape[0]):
        need = R[m, 1:] + prm.d_safe + prm.extra_margin + RHO[m, 1:]
        if ELL is not None and ELL[m] is not None:
            sg = _scaled_gauge_series(ELL[m][1:], P[1:])
            pen += float(np.sum(np.maximum(0.0, need - sg)))
        else:
            dist = np.linalg.norm(P[1:] - O[m, 1:], axis=1)
            pen += float(np.sum(np.maximum(0.0, need - dist)))
    # Vi phạm (R4) tại điểm giữa mỗi đoạn
    if prm.midpoint_cuts and O.shape[0]:
        O_mid, R_mid, RHO_mid, ELL_mid = _midpoint_arrays(O, R, RHO, ELL)
        # Vị trí tại t_k + dt/2 theo mô hình tích phân kép:
        # p_k + (dt/2)·v_k + ½(dt/2)²·u_k = p_k + 0,5·dt·v_k + 0,125·dt²·u_k
        P_mid = P[:-1] + 0.5 * dt * V[:-1] + 0.125 * dt * dt * U
        for m in range(O.shape[0]):
            need = R_mid[m] + prm.d_safe + prm.extra_margin + RHO_mid[m]
            if ELL_mid is not None and ELL_mid[m] is not None:
                sg = _scaled_gauge_series(ELL_mid[m], P_mid)
                pen += float(np.sum(np.maximum(0.0, need - sg)))
            else:
                dist = np.linalg.norm(P_mid - O_mid[m], axis=1)
                pen += float(np.sum(np.maximum(0.0, need - dist)))
    return cost + tau * pen, pen


# Hàm giải chính
def plan(
    p0: np.ndarray,
    v0: np.ndarray,
    goal: np.ndarray,
    obstacles: list,
    prm: PlannerParams,
    seed_P: np.ndarray | None = None,
) -> PlanResult:
    """Chạy DCA dạng phạt cho một lần lập kế hoạch.

    Args:
        p0: (3,) vị trí đầu, m.
        v0: (3,) vận tốc đầu, m/s.
        goal: (3,) vị trí đích, m.
        obstacles: danh sách vật cản, xem normalize_obstacles().
        prm: tham số bộ giải.
        seed_P: (K+1, 3) quỹ đạo khởi tạo, m (A* hoặc nghiệm chu kỳ
            trước). None hoặc sai kích thước → đường thẳng p0 → goal.

    Returns:
        PlanResult của quỹ đạo tốt nhất tìm được. Nếu không QP nào có
        nghiệm, trả quỹ đạo đứng yên tại p0 với success = False.
    """
    t_start = time.perf_counter()
    K, dt = prm.K, prm.dt
    M = len(obstacles)
    O, R, RHO, ELL = normalize_obstacles(obstacles, K)

    # --- Quỹ đạo khởi tạo: seed_P nếu đúng kích thước, ngược lại nội suy
    # thẳng p0 → đích
    if seed_P is not None and seed_P.shape == (K + 1, 3):
        P_cur = seed_P.copy()
    else:
        ts = np.linspace(0.0, 1.0, K + 1)[:, None]
        P_cur = p0[None, :] * (1 - ts) + goal[None, :] * ts

    # --- Tập điểm tuyến tính hoá q cho mỗi ràng buộc nút (m, k), k = 1..K.
    # Mỗi điểm q sinh một mặt cắt; ban đầu chỉ có điểm của quỹ đạo khởi tạo.
    lin_points: dict = {}
    for m in range(M):
        for k in range(1, K + 1):
            lin_points[(m, k)] = [P_cur[k].copy()]
    # --- Vật cản tại điểm giữa đoạn và tập điểm tuyến tính hoá riêng cho
    # các ràng buộc điểm giữa (m, k), k = 0..K-1 (điểm đầu: trung điểm hai nút).
    O_mid, R_mid, RHO_mid, ELL_mid = (
        _midpoint_arrays(O, R, RHO, ELL) if M else (None, None, None, None)
    )
    lin_points_mid: dict = {}
    if prm.midpoint_cuts:
        for m in range(M):
            for k in range(K):
                lin_points_mid[(m, k)] = [0.5 * (P_cur[k] + P_cur[k + 1])]

    tau = prm.tau0
    obj_history: list = []
    total_iter = 0
    P_best, V_best, U_best = None, None, None  # nghiệm QP gần nhất
    total_slack = np.inf

    # Vòng ngoài: các mức phạt tau. Vòng trong: các vòng DCA ở một mức phạt.
    while True:
        for _ in range(prm.max_iter):
            total_iter += 1
            # ---- Bài toán con lồi (QP), dựng lại từ đầu ở mỗi vòng ----
            U = cp.Variable((K, 3))
            P = cp.Variable((K + 1, 3))
            V = cp.Variable((K + 1, 3))
            # Biến dư (R4) tại các nút; cột k = 0 không có ràng buộc và
            # không vào hàm mục tiêu (chỉ để chỉ số khớp với k).
            S = cp.Variable((max(M, 1), K + 1)) if M else None

            # Trạng thái đầu (R2)
            cons = [P[0] == p0, V[0] == v0]

            # Động lực học tích phân kép (R1)
            for k in range(K):
                cons.append(P[k + 1] == P[k] + dt * V[k] + 0.5 * dt * dt * U[k])
                cons.append(V[k + 1] == V[k] + dt * U[k])

            # Giới hạn vận tốc và gia tốc (R3), dạng hộp theo từng thành phần.
            # V có dạng (K+1, 3) nên V[1:, :] là các bước k ≥ 1. Không chặn
            # V[0] vì v0 là trạng thái đo được: nếu |v0_i| > v_max thì chặn
            # V[0] sẽ làm QP vô nghiệm ngay từ đầu.
            cons.append(cp.abs(V[1:, :]) <= prm.v_max)
            cons.append(cp.abs(U) <= prm.u_max)

            # Cao độ sàn (R3), áp MỀM bằng biến dư Sfl có phạt như (R4).
            # Sàn cứng có thể mâu thuẫn với các mặt cắt sinh ra khi quỹ đạo
            # đi phía dưới vật cản ở các vòng đầu, khiến bộ giải không
            # vòng qua được; sàn mềm cho phép vi phạm tạm thời rồi bị đẩy
            # lên khi tau tăng.
            Sfl = cp.Variable(K + 1)
            cons.append(Sfl >= 0)
            cons.append(P[:, 2] + Sfl >= prm.z_floor)

            # Biến dư cho ràng buộc điểm giữa
            S_mid = cp.Variable((max(M, 1), K)) if (M and prm.midpoint_cuts) else None
            if M:
                cons.append(S >= 0)
                # ---- Mặt cắt (R4) tại các nút k = 1..K ----
                for m in range(M):
                    for k in range(1, K + 1):
                        o = O[m, k]
                        need = R[m, k] + prm.d_safe + prm.extra_margin + RHO[m, k]
                        if ELL[m] is not None:
                            # Ellipsoid: tiếp tuyến của a_min·g tại q, dạng
                            # w·p ≥ b − s (xem ellipsoid.Ellipsoid.cut); biến dư
                            # vẫn tính bằng mét như nhánh cầu.
                            for q in lin_points[(m, k)]:
                                cut = ELL[m][k].cut(q, need)
                                # q trùng tâm: gradient không xác định
                                if cut is None:
                                    continue
                                w, b = cut
                                cons.append(w @ P[k] >= b - S[m, k])
                            continue
                        # Cầu: e·(p − o) ≥ need − s với e là hướng đơn vị từ tâm tới q.
                        # Một biến dư S[m, k] dùng chung cho mọi mặt cắt của (m, k).
                        for q in lin_points[(m, k)]:
                            dvec = q - o
                            nrm = np.linalg.norm(dvec)
                            if nrm < 1e-9:  # q trùng tâm: bỏ qua
                                continue
                            unit = dvec / nrm
                            cons.append(unit @ (P[k] - o) >= need - S[m, k])
                # ---- Mặt cắt (R4) tại điểm giữa mỗi đoạn [k, k+1] ----
                if prm.midpoint_cuts:
                    cons.append(S_mid >= 0)
                    for m in range(M):
                        for k in range(K):
                            o = O_mid[m, k]
                            need = (
                                R_mid[m, k]
                                + prm.d_safe
                                + prm.extra_margin
                                + RHO_mid[m, k]
                            )
                            # Vị trí tại t_k + dt/2, là biểu thức afin của biến
                            p_mid = P[k] + 0.5 * dt * V[k] + 0.125 * dt * dt * U[k]
                            if ELL_mid is not None and ELL_mid[m] is not None:
                                for q in lin_points_mid[(m, k)]:
                                    cut = ELL_mid[m][k].cut(q, need)
                                    if cut is None:
                                        continue
                                    w, b = cut
                                    cons.append(w @ p_mid >= b - S_mid[m, k])
                                continue
                            for q in lin_points_mid[(m, k)]:
                                dvec = q - o
                                nrm = np.linalg.norm(dvec)
                                if nrm < 1e-9:
                                    continue
                                unit = dvec / nrm
                                cons.append(unit @ (p_mid - o) >= need - S_mid[m, k])
            # ---- Hàm mục tiêu: J + tau·(tổng mọi biến dư) ----
            obj = (
                prm.alpha * cp.sum_squares(U)
                + prm.beta * cp.sum_squares(U[1:] - U[:-1])
                + prm.gamma * cp.sum_squares(P[K] - goal)
            )
            obj += tau * cp.sum(Sfl)
            if M:
                obj += tau * cp.sum(S[:, 1:])
                if prm.midpoint_cuts:
                    obj += tau * cp.sum(S_mid)
            prob = cp.Problem(cp.Minimize(obj), cons)
            prob.solve(solver=cp.OSQP, **OSQP_OPTS)
            # Solver không trả nghiệm: bỏ mức phạt này, sang mức kế tiếp.
            if P.value is None:
                break
            P_new = np.asarray(P.value)
            U_new = np.asarray(U.value)
            V_new = np.asarray(V.value)

            # ---- Đánh giá nghiệm bằng khoảng cách thật ----
            obj_true, pen = _true_penalized_objective(
                P_new, V_new, U_new, goal, O, R, RHO, prm, tau, ELL
            )
            obj_history.append(obj_true)
            # Tổng vi phạm = vi phạm (R4) thật + vi phạm cao độ sàn
            floor_slack = float(np.sum(np.maximum(0.0, prm.z_floor - P_new[:, 2])))
            total_slack = pen + floor_slack
            P_best, V_best, U_best = P_new, V_new, U_new
            # Độ dịch lớn nhất của quỹ đạo so với vòng trước (m)
            moved = float(np.max(np.abs(P_new - P_cur)))

            # ---- Tích luỹ điểm tuyến tính hoá: thêm điểm mới, bỏ điểm cũ nhất
            # khi vượt max_lin_points ----
            for m in range(M):
                for k in range(1, K + 1):
                    pts = lin_points[(m, k)]
                    pts.append(P_new[k].copy())
                    if len(pts) > prm.max_lin_points:
                        del pts[0]
            if prm.midpoint_cuts:
                P_mid_new = P_new[:-1] + 0.5 * dt * V_new[:-1] + 0.125 * dt * dt * U_new
                for m in range(M):
                    for k in range(K):
                        pts = lin_points_mid[(m, k)]
                        pts.append(P_mid_new[k].copy())
                        if len(pts) > prm.max_lin_points:
                            del pts[0]
            P_cur = P_new
            # Dừng vòng trong khi quỹ đạo hầu như không đổi và (R4) đã thoả
            if moved < 1e-4 and pen <= prm.slack_tol:
                break

        # ---- Kết thúc khi khả thi, khi tau đã chạm trần, hoặc khi chưa có nghiệm nào;
        # ngược lại tăng phạt và lặp tiếp ----
        if total_slack <= prm.slack_tol or tau >= prm.tau_max or P_best is None:
            break
        tau = min(tau * prm.tau_factor, prm.tau_max)
        if prm.reset_lin_on_tau:
            # Bỏ các điểm tuyến tính hoá cũ, chỉ giữ điểm của quỹ đạo hiện tại
            for m in range(M):
                for k in range(1, K + 1):
                    lin_points[(m, k)] = [P_cur[k].copy()]
            if prm.midpoint_cuts:
                for m in range(M):
                    for k in range(K):
                        lin_points_mid[(m, k)] = [0.5 * (P_cur[k] + P_cur[k + 1])]

    solve_time = time.perf_counter() - t_start
    if P_best is None:  # không QP nào có nghiệm: trả quỹ đạo đứng yên tại p0
        P_best = np.repeat(p0[None, :], K + 1, axis=0)
        V_best = np.zeros((K + 1, 3))
        U_best = np.zeros((K, 3))

    # ---- Khe hở nhỏ nhất tới bề mặt các vật cản đầu vào (tâm, bán kính
    # truyền vào hàm), tại mọi nút k = 0..K. Không trừ d_safe, extra_margin, ρ.
    # Với ellipsoid dùng a_min·(g(p) − 1), một cận dưới của khoảng cách
    # Euclid tới bề mặt (bằng ‖p − o‖ − r khi là cầu).
    if M:
        clears = []
        for m in range(M):
            if ELL[m] is not None:
                clears.append(
                    min(
                        float(ELL[m][k].clearance_lower_bound(P_best[k]))
                        for k in range(K + 1)
                    )
                )
            else:
                dist = np.linalg.norm(P_best - O[m], axis=1)
                clears.append(float(np.min(dist - R[m])))
        min_clear = min(clears)
    else:
        min_clear = np.inf
    success = bool(total_slack <= max(prm.slack_tol, 1e-2))
    return PlanResult(
        P=P_best,
        V=V_best,
        U=U_best,
        success=success,
        min_clearance=min_clear,
        total_slack=float(total_slack),
        iterations=total_iter,
        solve_time=solve_time,
        obj_history=obj_history,
    )
