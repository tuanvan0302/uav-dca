# -*- coding: utf-8 -*-
"""
Kiểm thử lớp 1 cho dự báo hình dạng hai lớp (core/shape_forecast.py):
  - chế độ sự thật nền: lấy mẫu đúng, ρ = 0;
  - phồng điều hòa 1 trục: dự báo log-bán-trục thuần tại cuối chân
    trời chính xác và tốt hơn hẳn mốc giữ nguyên giá trị cuối
    (persistence);
  - lệch pha 2 trục: pha riêng theo trục được bắt đúng (tỉ lệ hai
    trục dự báo khớp tỉ lệ thật);
  - vật xoay: cổng tỉ lệ trục mở, yaw được dự báo, trục ngang được
    phồng theo σ_ψ;
  - vật gần cầu: cổng đóng — yaw không ước lượng, hai trục ngang
    được tròn hoá (an toàn theo yaw chưa biết);
  - guardrail ngoại suy: cấu hình mặc định n_hist = 24 và sự thật
    chu kỳ 12 s (cửa sổ ≲ 1 chu kỳ) rơi về chế độ dự phòng, sai số
    log-bán-trục bị chặn (không có guardrail thì tới ~10^2);
  - seed A* theo vùng quét: vật di chuyển vào đường bay không còn
    bị seed xuyên qua vùng chiếm dụng cuối chân trời (ca B1);
  - phồng yaw theo σ_ψ được ĐỊNH LƯỢNG là xấp xỉ bậc nhất, không
    phải bao phủ có chứng nhận; tròn hoá cổng đóng bao phủ tuyệt
    đối (ca B2);
  - định dạng đầu ra tương thích dca_planner.plan.

Đây là kiểm thử đơn vị nhanh, không phải đánh giá phân phối đa seed.

Chạy:  python3 -m tests.test_shape_forecast   (từ gốc code/)"""

import numpy as np

from core.astar_seed import astar_seed
from core.dca_planner import PlannerParams, plan
from core.ellipsoid import Ellipsoid
from core.shape_forecast import (
    MODE_FALLBACK,
    bounding_sphere_tuples,
    forecast_error_ellipsoid,
    kalman_ellipsoid_obstacles,
    sample_ellipsoid_truth,
    truth_ellipsoid_obstacles,
)

K, DT = 28, 0.25
T_NOW = 12.0
N_HIST = 48  # 12 s lịch sử = 2 chu kỳ biến dạng
FAIL = []


def _check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


# --- Các sự thật nền tổng hợp -------------------------------------------
def fn_harm1(t):
    """Phồng điều hòa 1 trục (trục x), hai trục còn lại cố định."""
    a1 = np.exp(0.30 * np.sin(2.0 * np.pi * t / 6.0))
    return np.array([5.0, 0.0, 1.5]), np.array([a1, 0.8, 0.8]), 0.0


def fn_phase2(t):
    """Hai trục phồng điều hòa lệch pha nhau 1,3 rad."""
    a1 = np.exp(0.30 * np.sin(2.0 * np.pi * t / 6.0))
    a2 = np.exp(0.30 * np.sin(2.0 * np.pi * t / 6.0 + 1.3))
    return np.array([5.0, 0.0, 1.5]), np.array([a1, a2, 0.9]), 0.2


def fn_rot(t):
    """Vật kéo dài đứng yên hình dạng, xoay đều quanh z."""
    return np.array([5.0, 0.0, 1.5]), np.array([1.4, 0.7, 0.9]), 0.12 * t


def fn_near_sphere(t):
    """Vật gần cầu (tỉ lệ trục ngang 1,05 < 1,15) đang xoay."""
    return np.array([5.0, 0.0, 1.5]), np.array([1.05, 1.0, 0.95]), 0.3 * t


def _forecast(fn, seed=42):
    rng = np.random.default_rng(seed)
    return kalman_ellipsoid_obstacles([fn], T_NOW, K, DT, rng, n_hist=N_HIST)[0]


def _pure_log_err(pred, fn):
    """Sai số log-bán-trục THUẦN (trước phồng/tròn hoá yaw) tại K."""
    true_a = fn(T_NOW + K * DT)[1]
    return np.abs(np.log(pred["axes_pure"][K]) - np.log(true_a))


# ----------------------------------------------------------------------
def test_truth_mode():
    obs = truth_ellipsoid_obstacles([fn_phase2], 0.0, K, DT)
    truth = sample_ellipsoid_truth([fn_phase2], 0.0, K, DT)
    ob = obs[0]
    ok = (
        np.all(ob["rho"] == 0.0)
        and np.max(np.abs(ob["centers"] - truth[0][0])) < 1e-12
        and np.max(np.abs(ob["axes"] - truth[0][1])) < 1e-12
        and np.max(np.abs(ob["yaws"] - truth[0][2])) < 1e-12
    )
    _check("Sự thật nền ellipsoid: lấy mẫu đúng, ρ = 0", ok)


def test_output_format():
    p = _forecast(fn_harm1)
    ok = (
        p["centers"].shape == (K + 1, 3)
        and p["axes"].shape == (K + 1, 3)
        and p["yaws"].shape == (K + 1,)
        and p["rho"].shape == (K + 1,)
        and np.all(p["axes"] > 0)
        and np.all(p["rho"] >= 0)
    )
    _check("Định dạng đầu ra: (K+1) bước, bán trục dương, ρ ≥ 0", ok)


def test_harmonic_single_axis():
    p = _forecast(fn_harm1)
    err = _pure_log_err(p, fn_harm1)
    base = abs(np.log(fn_harm1(T_NOW + K * DT)[1][0]) - np.log(fn_harm1(T_NOW)[1][0]))
    _check(
        "Điều hòa 1 trục: sai số log-bán-trục thuần tại K < 0,10",
        err[0] < 0.10,
        f"(trục biến dạng: {err[0]:.4f})",
    )
    _check(
        "Điều hòa 1 trục: tốt hơn hẳn persistence",
        err[0] < 0.5 * base,
        f"(dự báo {err[0]:.4f} so với persistence {base:.4f})",
    )
    _check(
        "Điều hòa 1 trục: hai trục tĩnh cũng được dự báo đúng",
        max(err[1], err[2]) < 0.15,
        f"(sai số {err[1]:.4f}, {err[2]:.4f})",
    )
    ec = forecast_error_ellipsoid([fn_harm1], [p], T_NOW, K, DT)["center"]
    _check(
        "Điều hòa 1 trục: sai số tâm cuối chân trời < 0,5 m (cổng M2)",
        ec < 0.5,
        f"({ec:.3f} m)",
    )


def test_two_axis_phase_shift():
    p = _forecast(fn_phase2)
    err = _pure_log_err(p, fn_phase2)
    _check(
        "Lệch pha 2 trục: cả hai trục đều dự báo đúng tại K",
        err[0] < 0.15 and err[1] < 0.15,
        f"(sai số {err[0]:.4f}, {err[1]:.4f})",
    )
    # Tỉ lệ hai trục mang thông tin pha: nếu hai trục bị ép dùng
    # chung một pha, tỉ lệ dự báo sẽ lệch xa tỉ lệ thật.
    true_ratio = np.log(
        fn_phase2(T_NOW + K * DT)[1][0] / fn_phase2(T_NOW + K * DT)[1][1]
    )
    fc_ratio = np.log(p["axes_pure"][K][0] / p["axes_pure"][K][1])
    _check(
        "Lệch pha 2 trục: pha riêng theo trục được giữ",
        abs(fc_ratio - true_ratio) < 0.15,
        f"(log-tỉ lệ dự báo {fc_ratio:.3f}, thật {true_ratio:.3f})",
    )


def test_rotation():
    p = _forecast(fn_rot)
    ey = forecast_error_ellipsoid([fn_rot], [p], T_NOW, K, DT)["yaw"]
    _check(
        "Xoay: cổng mở, sai số yaw tại K < 0,25 rad",
        ey < 0.25,
        f"({ey:.3f} rad, σ_ψ(K) = {p['sigma_yaw'][K]:.3f})",
    )
    _check(
        "Xoay: trục ngang được phồng theo σ_ψ (axes ≥ axes_pure)",
        np.all(p["axes"][:, :2] >= p["axes_pure"][:, :2] - 1e-12)
        and p["axes"][K][0] > p["axes_pure"][K][0] + 0.02,
        f"(a1 tại K: {p['axes'][K][0]:.3f} so với thuần {p['axes_pure'][K][0]:.3f})",
    )


def test_near_sphere_gate_closed():
    p = _forecast(fn_near_sphere)
    _check(
        "Gần cầu: cổng đóng — σ_ψ ≡ 0, không ước lượng yaw",
        np.all(p["sigma_yaw"] == 0.0),
    )
    _check(
        "Gần cầu: hai trục ngang được tròn hoá bằng nhau",
        np.allclose(p["axes"][:, 0], p["axes"][:, 1]),
        f"(a1(K) = {p['axes'][K][0]:.3f}, a2(K) = {p['axes'][K][1]:.3f})",
    )
    err = _pure_log_err(p, fn_near_sphere)
    _check(
        "Gần cầu: dự báo bán trục thuần vẫn bám đúng",
        float(np.max(err)) < 0.18,
        f"(sai số lớn nhất {np.max(err):.4f})",
    )


def fn_period12(t):
    """Biến dạng chu kỳ 12 s: cửa sổ 12 s chỉ chứa 1 chu kỳ."""
    a1 = np.exp(0.30 * np.sin(2.0 * np.pi * t / 12.0))
    return np.array([5.0, 0.0, 1.5]), np.array([a1, 0.8, 0.8]), 0.0


def test_guardrail_default_config():
    # Cấu hình MẶC ĐỊNH n_hist = 24 (6 s lịch sử = 1 chu kỳ biến dạng
    # 6 s): không có guardrail thì sai số log-bán-trục tới ~10^2.
    # Trục biến dạng phải rơi về DỰ PHÒNG và sai số bị chặn.
    for seed in (0, 5, 42):
        rng = np.random.default_rng(seed)
        p = kalman_ellipsoid_obstacles([fn_harm1], T_NOW, K, DT, rng)[0]
        err = _pure_log_err(p, fn_harm1)
        _check(
            f"Guardrail mặc định n_hist=24 (seed {seed}): "
            "trục biến dạng ở chế độ dự phòng, sai số log ≤ 1,0",
            int(p["axis_mode"][0]) == MODE_FALLBACK
            and float(np.max(err)) <= 1.0
            and np.all(np.isfinite(p["axes"]))
            and np.all(p["axes"] > 0),
            f"(chế độ {int(p['axis_mode'][0])}, sai số max {np.max(err):.3f})",
        )


def test_guardrail_period_12s():
    # Sự thật chu kỳ 12 s ở n_hist = 48: cửa sổ chứa đúng 1 chu kỳ
    # hiệu dụng — không có guardrail thì bán trục bị đánh giá thấp
    # ~6 lần tại K.
    for seed in (0, 5, 42):
        rng = np.random.default_rng(seed)
        p = kalman_ellipsoid_obstacles([fn_period12], T_NOW, K, DT, rng, n_hist=N_HIST)[
            0
        ]
        err = _pure_log_err(p, fn_period12)
        _check(
            f"Guardrail chu kỳ 12 s (seed {seed}): dự phòng, sai số ≤ 1,0",
            int(p["axis_mode"][0]) == MODE_FALLBACK and float(np.max(err)) <= 1.0,
            f"(chế độ {int(p['axis_mode'][0])}, sai số max {np.max(err):.3f})",
        )


def test_seed_swept_region_moving_obstacle():
    # Ca B1: vật di chuyển VÀO đường bay, tâm từ (5, -4, 1,5) tới
    # (5, 0, 1,5) trong chân trời, bán trục 1,0. Nếu chỉ dùng cầu tại
    # tâm bước 0, seed xuyên ĐÚNG tâm cuối chân trời (khoảng cách
    # 0,00 m). bounding_sphere_tuples dựng vùng quét (một cầu mỗi
    # bước), nên toàn bộ đường seed phải nằm ngoài r + d_safe quanh
    # MỌI tâm trên chân trời.
    c0 = np.array([5.0, -4.0, 1.5])
    c1 = np.array([5.0, 0.0, 1.5])
    centers = c0[None, :] + (c1 - c0)[None, :] * np.linspace(0.0, 1.0, K + 1)[:, None]
    ob = dict(
        centers=centers,
        axes=np.ones((K + 1, 3)),
        yaws=np.zeros(K + 1),
        rho=np.zeros(K + 1),
    )
    tuples = bounding_sphere_tuples([ob])
    _check(
        "Vùng quét A*: một cầu bao cho mỗi bước của chân trời",
        len(tuples) == K + 1,
        f"({len(tuples)} cầu)",
    )
    seed = astar_seed(
        np.array([0.0, 0.0, 1.5]),
        np.array([10.0, 0.0, 1.5]),
        tuples,
        K + 1,
        d_safe=0.5,
        z_floor=0.8,
    )
    if seed is None:
        _check("Vùng quét A*: tìm được seed cho vật di chuyển", False)
        return
    dense = []
    for i in range(len(seed) - 1):
        n = max(2, int(np.ceil(np.linalg.norm(seed[i + 1] - seed[i]) / 0.05)) + 1)
        dense.append(
            seed[i] + (seed[i + 1] - seed[i]) * np.linspace(0.0, 1.0, n)[:, None]
        )
    dense = np.vstack(dense)
    d_all = min(
        float(np.min(np.linalg.norm(dense - centers[k][None, :], axis=1)))
        for k in range(K + 1)
    )
    d_end = float(np.min(np.linalg.norm(dense - centers[K][None, :], axis=1)))
    _check(
        "Vùng quét A*: seed ngoài r + d_safe quanh mọi tâm chân trời",
        d_all >= 1.0 + 0.5 - 0.02,
        f"(khoảng cách nhỏ nhất {d_all:.2f} m, ngưỡng 1,50 m)",
    )
    _check(
        "Vùng quét A*: seed không xuyên tâm cuối chân trời",
        d_end >= 1.45,
        f"(khoảng cách {d_end:.2f} m)",
    )


def _fib_sphere(n=4000):
    i = np.arange(n)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y**2))
    return np.column_stack([r * np.cos(golden * i), y, r * np.sin(golden * i)])


def test_yaw_inflation_is_first_order():
    # Định lượng ca B2: phồng trục ngang theo
    # σ_ψ là XẤP XỈ BẬC NHẤT, không phải bao phủ có chứng nhận.
    # Test này ĐO và khoá hành vi đó bằng ngưỡng khai báo (không
    # khẳng định bao phủ): lấy mẫu biên dày (Fibonacci 4000 điểm)
    # của ellipsoid thật bị sai yaw δ, đếm tỉ lệ điểm nằm ngoài
    # ellipsoid dự báo đã phồng theo đúng quy tắc của module và độ
    # xuyên scaled-gauge lớn nhất. Ngưỡng là bao hồi quy quanh số
    # đo hiện tại, ghi rõ để lần sửa sau không lặng lẽ tệ đi.
    fib = _fib_sphere()

    def _rot(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _measure(axes, sigma, delta):
        s = np.sin(min(sigma, np.pi / 2.0))
        a_inf = np.array([axes[0] + axes[1] * s, axes[1] + axes[0] * s, axes[2]])
        ell = Ellipsoid(np.zeros(3), a_inf, 0.0)
        X = (axes * fib) @ _rot(delta).T
        g = ell.gauge(X)
        out = g > 1.0
        pen = float(np.max(ell.clearance_lower_bound(X[out]))) if np.any(out) else 0.0
        return float(np.mean(out)), pen

    sigma = 0.185  # σ_ψ(K) đo được của bộ dự báo (Q_YAW = 1e-4)
    axes_f3 = np.array([1.6, 0.45, 0.55])  # kiểu F3 giao thức: tỉ lệ ~3,6
    f1, p1 = _measure(axes_f3, sigma, sigma)
    _check(
        "Phồng yaw F3: ở δ = σ_ψ vẫn THỦNG (không bao phủ tuyệt đối)",
        0.005 < f1 <= 0.30 and p1 <= 0.05,
        f"(ngoài {100 * f1:.1f}%, xuyên tối đa {p1:.3f} m)",
    )
    f2, p2 = _measure(axes_f3, sigma, 2.0 * sigma)
    _check(
        "Phồng yaw F3: ở δ = 2σ_ψ thủng nặng hơn, vẫn trong bao khai báo",
        f2 <= 0.50 and p2 <= 0.20,
        f"(ngoài {100 * f2:.1f}%, xuyên tối đa {p2:.3f} m)",
    )
    axes_mild = np.array([1.6, 0.7, 0.9])
    f3, _ = _measure(axes_mild, sigma, 2.0 * sigma)
    _check(
        "Phồng yaw vật ít kéo dài: δ = 2σ_ψ cũng thủng trong bao khai báo",
        0.005 < f3 <= 0.40,
        f"(ngoài {100 * f3:.1f}%)",
    )
    # Đối ngược: tròn hoá khi cổng đóng là bao phủ TUYỆT ĐỐI theo
    # mọi yaw — tỉ lệ ngoài phải bằng 0 ở mọi δ.
    ell_round = Ellipsoid(np.zeros(3), np.array([1.6, 1.6, 0.55]), 0.0)
    ok_round = True
    for delta in (0.37, 1.0, 2.5):
        X = (axes_f3 * fib) @ _rot(delta).T
        ok_round &= bool(np.all(ell_round.gauge(X) <= 1.0 + 1e-12))
    _check("Tròn hoá cổng đóng: bao phủ tuyệt đối theo mọi yaw (đối ngược)", ok_round)


def test_forecast_plugs_into_planner():
    # Đầu ra dự báo (chế độ ước lượng) phải cắm thẳng được vào plan().
    p = _forecast(fn_harm1)
    prm = PlannerParams()
    res = plan(
        np.array([0.0, 0.0, 1.5]), np.zeros(3), np.array([10.0, 0.0, 1.5]), [p], prm
    )
    _check(
        "Dự báo ellipsoid cắm vào plan() chạy được",
        res.P.shape == (prm.K + 1, 3) and np.all(np.isfinite(res.P)),
        f"(success = {res.success}, biến dư {res.total_slack:.2e})",
    )


if __name__ == "__main__":
    test_truth_mode()
    test_output_format()
    test_harmonic_single_axis()
    test_two_axis_phase_shift()
    test_rotation()
    test_near_sphere_gate_closed()
    test_guardrail_default_config()
    test_guardrail_period_12s()
    test_seed_swept_region_moving_obstacle()
    test_yaw_inflation_is_first_order()
    test_forecast_plugs_into_planner()
    print("\nShape-forecast tests:", "PASS" if not FAIL else f"FAIL {FAIL}")
