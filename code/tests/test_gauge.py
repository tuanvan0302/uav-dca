# -*- coding: utf-8 -*-
"""
Kiểm thử lớp 1 cho tầng hình dạng H2 (core/ellipsoid.py và nhánh gauge
của core/dca_planner.py):
  - hàm gauge: gradient dạng kín vs sai phân hữu hạn; tính lồi số
    học (bất đẳng thức Jensen + Hessian nửa xác định dương);
  - trường hợp riêng cầu: gauge và mặt cắt gauge thu về đúng mặt
    cắt của nhánh cầu (sai số < 1e-9);
  - mặt cắt là xấp xỉ trong: điểm thỏa mặt cắt => thỏa (R4*);
  - quy tắc đổi biên mét -> gauge (Chương 5): g(p) >= 1 + d/a_min
    => khe hở Euclid tới bề mặt >= d (kiểm chứng số học);
  - bộ giải ở chế độ ellipsoid: tương đương cầu ở mức bộ giải, kịch
    bản thanh dài, vật biến dạng ở chế độ sự thật nền.

Chạy:  python3 -m tests.test_gauge   (từ gốc code/)
       hoặc pytest tests/test_gauge.py
"""

import numpy as np

from core.astar_seed import astar_seed
from core.dca_planner import PlannerParams, plan
from core.ellipsoid import Ellipsoid, interpolate, metric_margin_to_gauge
from core.shape_forecast import bounding_sphere_tuples, truth_ellipsoid_obstacles

PRM = PlannerParams()
FAIL = []
RNG = np.random.default_rng(20261121)


def _check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def _rand_ellipsoid():
    return Ellipsoid(
        RNG.normal(0.0, 2.0, 3),
        np.exp(RNG.normal(0.0, 0.45, 3)),
        RNG.uniform(-np.pi, np.pi),
    )


# ----------------------------------------------------------------------
# 1. Gradient dạng kín
# ----------------------------------------------------------------------
def test_closed_form_gradient():
    h = 1e-6
    worst = 0.0
    for _ in range(30):
        ell = _rand_ellipsoid()
        x = ell.center + RNG.normal(0.0, 2.0, 3)
        ga = ell.grad(x)
        gf = np.zeros(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            gf[i] = (ell.gauge(x + e) - ell.gauge(x - e)) / (2.0 * h)
        worst = max(worst, float(np.max(np.abs(ga - gf))))
    _check(
        "∇g dạng kín khớp sai phân hữu hạn trung tâm",
        worst < 1e-6,
        f"(sai số lớn nhất {worst:.2e})",
    )


# ----------------------------------------------------------------------
# 2. Tính lồi số học của gauge
# ----------------------------------------------------------------------
def test_convexity():
    # Jensen: g(θx + (1−θ)y) ≤ θ g(x) + (1−θ) g(y)
    worst_j = -np.inf
    for _ in range(300):
        ell = _rand_ellipsoid()
        x = ell.center + RNG.normal(0.0, 2.0, 3)
        y = ell.center + RNG.normal(0.0, 2.0, 3)
        th = RNG.uniform(0.0, 1.0)
        worst_j = max(
            worst_j,
            ell.gauge(th * x + (1 - th) * y)
            - (th * ell.gauge(x) + (1 - th) * ell.gauge(y)),
        )
    _check(
        "Gauge lồi theo Jensen (300 cặp ngẫu nhiên)",
        worst_j <= 1e-9,
        f"(vi phạm lớn nhất {worst_j:.2e})",
    )
    # Hessian (sai phân hữu hạn) nửa xác định dương ngoài tâm
    h = 1e-4
    worst_e = np.inf
    for _ in range(15):
        ell = _rand_ellipsoid()
        x = ell.center + RNG.normal(0.0, 1.5, 3)
        H = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                ei = np.zeros(3)
                ei[i] = h
                ej = np.zeros(3)
                ej[j] = h
                H[i, j] = (
                    ell.gauge(x + ei + ej)
                    - ell.gauge(x + ei - ej)
                    - ell.gauge(x - ei + ej)
                    + ell.gauge(x - ei - ej)
                ) / (4 * h * h)
        worst_e = min(worst_e, float(np.min(np.linalg.eigvalsh(H))))
    _check(
        "Hessian của gauge nửa xác định dương (số học)",
        worst_e >= -1e-4,
        f"(trị riêng nhỏ nhất {worst_e:.2e})",
    )


# ----------------------------------------------------------------------
# 3. Trường hợp riêng cầu: gauge và mặt cắt
# ----------------------------------------------------------------------
def test_sphere_case():
    worst_g = 0.0
    worst_cut = 0.0
    for _ in range(50):
        o = RNG.normal(0.0, 2.0, 3)
        r = float(np.exp(RNG.normal(0.0, 0.4)))
        ell = Ellipsoid(o, [r, r, r], yaw=float(RNG.uniform(-np.pi, np.pi)))
        x = o + RNG.normal(0.0, 2.0, 3)
        worst_g = max(worst_g, abs(ell.scaled_gauge(x) - np.linalg.norm(x - o)))
        # Mặt cắt gauge vs mặt cắt cầu của code gốc:
        # gốc: unit·(p − o) >= need − s  <=>  unit·p >= need + unit·o − s
        q = o + RNG.normal(0.0, 2.0, 3)
        if np.linalg.norm(q - o) < 0.2:
            continue
        need = r + PRM.d_safe + PRM.extra_margin + 0.37
        w, b = ell.cut(q, need)
        unit = (q - o) / np.linalg.norm(q - o)
        b_ref = need + float(unit @ o)
        worst_cut = max(worst_cut, float(np.max(np.abs(w - unit))), abs(b - b_ref))
    _check(
        "a=b=c: a_min·g(x) = ‖x − o‖ (máy)",
        worst_g < 1e-12,
        f"(sai số lớn nhất {worst_g:.2e})",
    )
    _check(
        "a=b=c: mặt cắt gauge = mặt cắt cầu của code gốc",
        worst_cut < 1e-9,
        f"(sai số lớn nhất {worst_cut:.2e}, ngưỡng yêu cầu 1e-9)",
    )


# ----------------------------------------------------------------------
# 4. Mặt cắt là xấp xỉ trong của (R4*)
# ----------------------------------------------------------------------
def test_cut_is_inner_approximation():
    n_bad = 0
    n_tot = 0
    for _ in range(40):
        ell = _rand_ellipsoid()
        delta = 0.5 + 0.03 + 0.2  # d + đệm + rho (m)
        need = ell.a_min + delta
        # Điểm tuyến tính hoá nằm ngoài ellipsoid đã phồng
        for _ in range(6):
            d = RNG.normal(0.0, 1.0, 3)
            q = ell.center + ell.rotation() @ (
                ell.axes * d / np.linalg.norm(d) * (1.6 + RNG.uniform(0, 1))
            )
            cut = ell.cut(q, need)
            if cut is None:
                continue
            w, b = cut
            for _ in range(25):
                # Điểm mẫu: từ q đi theo hướng không vi phạm mặt cắt
                step = RNG.normal(0.0, 1.2, 3)
                if w @ step < 0:
                    step = step - 2.0 * (w @ step) / (w @ w) * w
                x = q + step * RNG.uniform(0.0, 3.0)
                if w @ x >= b - 1e-12:
                    n_tot += 1
                    if ell.scaled_gauge(x) < need - 1e-9:
                        n_bad += 1
    _check(
        "Thỏa mặt cắt => thỏa (R4*) (xấp xỉ trong)",
        n_bad == 0 and n_tot >= 200,
        f"({n_bad} vi phạm / {n_tot} điểm mẫu)",
    )


# ----------------------------------------------------------------------
# 5. Đổi biên mét -> gauge là điều kiện đủ (Chương 5)
# ----------------------------------------------------------------------
def _surface_points(ell, n_th=60, n_ph=30):
    th = np.linspace(0, 2 * np.pi, n_th)
    ph = np.linspace(0.05, np.pi - 0.05, n_ph)
    TH, PH = np.meshgrid(th, ph)
    d = np.stack(
        [np.sin(PH) * np.cos(TH), np.sin(PH) * np.sin(TH), np.cos(PH)], axis=-1
    ).reshape(-1, 3)
    return ell.center + (d * ell.axes) @ ell.rotation().T


def test_slack_unit_conversion():
    d = 0.5
    worst = np.inf
    for _ in range(8):
        ell = _rand_ellipsoid()
        surf = _surface_points(ell)
        R = ell.rotation()
        for j in range(3):  # tia dọc ba trục vật + vài tia ngẫu nhiên
            dirs = [R[:, j]]
            if j == 0:
                u = RNG.normal(0, 1, 3)
                dirs.append(u / np.linalg.norm(u))
            for u in dirs:
                gdir = np.linalg.norm((u @ R) / ell.axes)  # g(o + s·u) = s·gdir
                s = (1.0 + metric_margin_to_gauge(d, ell)) / gdir
                x = ell.center + s * u
                dist = float(np.min(np.linalg.norm(surf - x, axis=1)))
                worst = min(worst, dist)
    _check(
        "g(p) ≥ 1 + d/a_min => khe hở Euclid ≥ d (mẫu bề mặt dày)",
        worst >= d - 1e-6,
        f"(khe hở số học nhỏ nhất {worst:.4f} m, d = {d})",
    )


# ----------------------------------------------------------------------
# 6. Nội suy điểm giữa thu về code gốc khi là cầu
# ----------------------------------------------------------------------
def test_midpoint_interpolation():
    o = np.array([1.0, 2.0, 3.0])
    e1 = Ellipsoid(o, [1.1, 1.1, 1.1], 0.4)
    e2 = Ellipsoid(o + 1.0, [1.5, 1.5, 1.5], 0.4)
    em = interpolate(e1, e2, 0.5)
    _check(
        "Nội suy giữa: cầu -> bán kính trung bình như code gốc",
        np.max(np.abs(em.axes - 1.3)) < 1e-12
        and np.max(np.abs(em.center - (o + 0.5))) < 1e-12,
        f"(bán trục giữa {em.axes[0]:.12f}, kỳ vọng 1.3)",
    )


# ----------------------------------------------------------------------
# 7. Bộ giải ở chế độ ellipsoid
# ----------------------------------------------------------------------
def _ell_dict(o, r, yaw):
    return [
        dict(
            centers=np.repeat(o[None, :], PRM.K + 1, axis=0),
            axes=np.repeat(np.array([[r, r, r]]), PRM.K + 1, axis=0),
            yaws=np.full(PRM.K + 1, yaw),
            rho=np.zeros(PRM.K + 1),
        )
    ]


def test_solver_matches_sphere_branch():
    start = np.array([0.0, 0.0, 1.5])
    goal = np.array([10.0, 0.0, 1.5])
    # (a) Đối xứng chết, yaw = 0, không seed: hai nhánh phải cho quỹ
    # đạo trùng nhau ở mức 1e-6 — kể cả việc CÙNG kẹt ở cực trị địa
    # phương phía vòng sai (hành vi đã biết của bộ giải khi thiếu A*
    # seed với vật chắn giữa đường; không phải hồi quy của nhánh
    # ellipsoid). So sánh khe hở ở ca đối xứng này là vô nghĩa vì
    # nghiệm kẹt là dao sắc: nhiễu số học 1e-16 của phép quay khi
    # yaw ≠ 0 đủ làm DCA thoát kẹt sang một cực trị khả thi khác.
    o = np.array([5.0, 0.0, 1.5])
    r = 1.0
    res_t = plan(start, np.zeros(3), goal, [(o, r)], PRM)
    res_e = plan(start, np.zeros(3), goal, _ell_dict(o, r, 0.0), PRM)
    dp = float(np.max(np.abs(res_t.P - res_e.P)))
    _check(
        "Bộ giải: a=b=c, yaw=0 cho quỹ đạo trùng nhánh cầu (< 1e-6)",
        res_e.success == res_t.success and dp < 1e-6,
        f"(max |ΔP| = {dp:.2e}, cùng success = {res_t.success})",
    )
    # (b) Chế độ vận hành thật: vật lệch trục (phá đối xứng) + A*
    # seed; ellipsoid a=b=c với yaw ≠ 0 phải cho cùng nghiệm.
    o2 = np.array([5.0, 0.9, 1.5])
    seed_t = astar_seed(
        start, goal, [(o2, r)], PRM.K + 1, d_safe=PRM.d_safe, z_floor=PRM.z_floor
    )
    seed_e = astar_seed(
        start,
        goal,
        bounding_sphere_tuples(_ell_dict(o2, r, 0.7)),
        PRM.K + 1,
        d_safe=PRM.d_safe,
        z_floor=PRM.z_floor,
    )
    _check(
        "A* seed từ cầu bao ngoài trùng bit seed từ tuple cầu",
        seed_t is not None and seed_e is not None and np.array_equal(seed_t, seed_e),
    )
    res_t = plan(start, np.zeros(3), goal, [(o2, r)], PRM, seed_P=seed_t)
    res_e = plan(start, np.zeros(3), goal, _ell_dict(o2, r, 0.7), PRM, seed_P=seed_t)
    dc = abs(res_e.min_clearance - res_t.min_clearance)
    _check(
        "Bộ giải: ellipsoid a=b=c (yaw≠0, có seed) tương đương cầu",
        res_t.success and res_e.success and dc < 1e-6,
        f"(khe hở {res_e.min_clearance:.4f} so với {res_t.min_clearance:.4f} m,"
        f" |Δ| = {dc:.2e})",
    )


def test_solver_long_rod():
    # Thanh dài 4 m dọc trục y tại x = 5: ellipsoid bán trục
    # (0,15; 2,0; 0,15). Cầu bao ngoài r = 2,0 (cho A* seed).
    start = np.array([0.0, 0.0, 1.5])
    goal = np.array([10.0, 0.0, 1.5])
    ob = dict(
        centers=np.repeat(np.array([[5.0, 0.0, 1.5]]), PRM.K + 1, axis=0),
        axes=np.repeat(np.array([[0.15, 2.0, 0.15]]), PRM.K + 1, axis=0),
        yaws=np.zeros(PRM.K + 1),
        rho=np.zeros(PRM.K + 1),
    )
    seed = astar_seed(
        start,
        goal,
        bounding_sphere_tuples([ob]),
        PRM.K + 1,
        d_safe=PRM.d_safe,
        z_floor=PRM.z_floor,
    )
    _check("Thanh dài: A* trên cầu bao ngoài tìm được seed", seed is not None)
    res = plan(start, np.zeros(3), goal, [ob], PRM, seed_P=seed)
    _check(
        "Thanh dài: bộ giải (ellipsoid) báo thành công",
        res.success,
        f"(biến dư {res.total_slack:.2e}, thời gian giải {res.solve_time:.2f} s)",
    )
    _check(
        "Thanh dài: cận dưới khe hở >= biên an toàn",
        res.min_clearance >= PRM.d_safe - 1e-3,
        f"(cận dưới khe hở {res.min_clearance:.3f} m)",
    )


def test_solver_deforming_truth():
    # Vật biến dạng ở chế độ sự thật nền: tâm đứng yên, trục x phồng–xẹp
    # điều hòa (như D3 nhưng dị hướng), hai trục còn lại cố định.
    def fn(t):
        a1 = 1.0 + 0.35 * np.sin(2.0 * np.pi * t / 6.0)
        return np.array([5.0, 0.0, 1.5]), np.array([a1, 0.9, 0.9]), 0.0

    obs = truth_ellipsoid_obstacles([fn], 0.0, PRM.K, PRM.dt)
    start = np.array([0.0, 0.0, 1.5])
    goal = np.array([10.0, 0.0, 1.5])
    res = plan(start, np.zeros(3), goal, obs, PRM)
    _check(
        "Biến dạng (sự thật nền): bộ giải báo thành công",
        res.success,
        f"(biến dư {res.total_slack:.2e}, thời gian giải {res.solve_time:.2f} s)",
    )
    _check(
        "Biến dạng (sự thật nền): cận dưới khe hở >= biên an toàn",
        res.min_clearance >= PRM.d_safe - 1e-3,
        f"(cận dưới khe hở {res.min_clearance:.3f} m)",
    )


if __name__ == "__main__":
    test_closed_form_gradient()
    test_convexity()
    test_sphere_case()
    test_cut_is_inner_approximation()
    test_slack_unit_conversion()
    test_midpoint_interpolation()
    test_solver_matches_sphere_branch()
    test_solver_long_rod()
    test_solver_deforming_truth()
    print("\nH2 shape-layer tests:", "PASS" if not FAIL else f"FAIL {FAIL}")
