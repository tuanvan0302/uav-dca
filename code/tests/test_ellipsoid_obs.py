# -*- coding: utf-8 -*-
"""
Test hook quan sát ellipsoid trong sim.sim_realtime — chế độ
obs_mode = "ellipsoid" của run_episode / get_obs:

  1. Hàm sự thật suy biến từ cầu: a = (r, r, r), yaw = 0, đúng tâm
     tại mọi t (khớp từng bit).
  2. Chế độ truth: quan sát trùng sự thật nền ellipsoid tại mọi
     bước của chân trời, ρ = 0.
  3. Chế độ frozen: quan sát là trạng thái tại thời điểm neo giữ
     nguyên trên cả chân trời, ρ = 0.
  4. Chế độ kalman KHÔNG nhiễu (mọi std đo = 0) trên F1 tĩnh: tâm
     và yaw trùng sự thật; bán trục tại k = 0 trùng sự thật; bán
     trục cuối chân trời phồng ĐÚNG công thức bậc nhất theo σ_yaw
     của tầng hình dạng (a1 + a2·sin(σ_ψ)) — phồng là bất định dự
     báo, không phải nhiễu đo. σ_yaw là đại lượng hiệp phương sai
     thuần: bất biến theo seed ở cùng tham số.
  5. Thống kê nhiễu của chế độ kalman (tham số khai của tầng hình
     dạng: MEAS_STD cho tâm, LOG_MEAS_STD cho log-bán-trục): tâm
     dự báo tại k = 0 không lệch, độ lệch chuẩn sau lọc nhỏ hơn σ
     đo khai; log-bán-trục trục dọc không lệch; hai trục ngang có
     lệch DƯƠNG do tròn hoá bằng trục lớn hơn khi cổng yaw đóng
     (lựa chọn an toàn đã ghi trong shape_forecast — kiểm tra dấu
     của lệch, không coi là lỗi).
  6. Tích hợp vòng lặp: episode ngắn qua run_episode ở chế độ
     ellipsoid ghi đúng config (obs_mode, ellipsoid_obs_source =
     "registry" cho họ F / "degenerate_from_sphere" cho họ D);
     mặc định vẫn là "sphere"; obs_mode lạ bị bác bỏ.

Chạy: python3 -m tests.test_ellipsoid_obs   (từ gốc code/)"""

from __future__ import annotations

import numpy as np

from core import shape_forecast as sf
from sim.scenarios_pkg import family_f
from sim.sim_realtime import degenerate_ellipsoid_fns, ellipsoid_obstacles, run_episode

K, DT = 28, 0.25


def test_degenerate_fns_exact():
    def mv(t):
        return np.array([1.0 + 0.5 * t, -2.0, 1.5]), 0.8 + 0.1 * np.sin(t)

    ell = degenerate_ellipsoid_fns([mv])
    for t in (0.0, 1.3, 7.9):
        c, a, psi = ell[0](t)
        ct, rt = mv(t)
        assert np.array_equal(c, ct), f"tâm lệch tại t={t}"
        assert np.array_equal(a, np.full(3, rt)), f"bán trục lệch tại t={t}"
        assert psi == 0.0
    print("  suy biến từ cầu: (tâm, (r,r,r), 0) khớp từng bit tại mọi t")


def test_truth_mode_matches_truth():
    fns = family_f.ellipsoid_truth("F3_xoay_yaw")
    rng = np.random.default_rng(0)
    obs = ellipsoid_obstacles("truth", fns, 2.0, K, DT, rng)
    truth = sf.sample_ellipsoid_truth(fns, 2.0, K, DT)
    (ct, at, yt) = truth[0]
    ob = obs[0]
    assert np.array_equal(ob["centers"], ct)
    assert np.array_equal(ob["axes"], at)
    assert np.array_equal(ob["yaws"], yt)
    assert np.all(ob["rho"] == 0.0)
    print("  truth: quan sát = sự thật ellipsoid mọi bước, ρ = 0")


def test_frozen_mode_matches_anchor():
    fns = family_f.ellipsoid_truth("F4_lech_pha")
    rng = np.random.default_rng(0)
    obs = ellipsoid_obstacles("frozen", fns, 3.0, K, DT, rng)
    c, a, psi = fns[0](3.0)
    ob = obs[0]
    assert np.all(ob["centers"] == c), "tâm frozen phải hằng = tâm tại neo"
    assert np.all(ob["axes"] == a), "bán trục frozen phải hằng = tại neo"
    assert np.all(ob["yaws"] == psi)
    assert np.all(ob["rho"] == 0.0)
    print("  frozen: trạng thái tại neo giữ nguyên cả chân trời, ρ = 0")


def test_kalman_zero_noise():
    fns = family_f.ellipsoid_truth("F1_thanh_dai")
    kw = dict(meas_std=0.0, log_meas_std=0.0, yaw_meas_std=0.0)
    ob = sf.kalman_ellipsoid_obstacles(fns, 5.0, K, DT, np.random.default_rng(0), **kw)[
        0
    ]
    c_true, a_true, _ = fns[0](5.0)
    assert np.max(np.abs(ob["centers"] - c_true)) < 1e-9, "tâm tĩnh phải trùng"
    assert np.max(np.abs(ob["yaws"])) < 1e-9, "yaw thật 0 phải dự báo 0"
    assert np.max(np.abs(ob["axes"][0] - a_true)) < 1e-9, "bán trục k=0"
    # Cuối chân trời: phồng bậc nhất theo σ_yaw (cổng yaw mở vì tỉ
    # lệ trục 2,0/0,15 > 1,15) — đúng công thức của tầng hình dạng.
    s = np.sin(ob["sigma_yaw"][K])
    expect = np.array([a_true[0] + a_true[1] * s, a_true[1] + a_true[0] * s, a_true[2]])
    assert np.max(np.abs(ob["axes"][K] - expect)) < 1e-9, (
        f"phồng yaw cuối chân trời sai công thức: {ob['axes'][K]} vs {expect}"
    )
    # σ_yaw là hiệp phương sai thuần: bất biến theo seed cùng tham số.
    ob2 = sf.kalman_ellipsoid_obstacles(
        fns, 5.0, K, DT, np.random.default_rng(999), **kw
    )[0]
    assert np.array_equal(ob["sigma_yaw"], ob2["sigma_yaw"])
    # Cùng seed -> tái lập từng bit.
    ob3 = sf.kalman_ellipsoid_obstacles(
        fns, 5.0, K, DT, np.random.default_rng(0), **kw
    )[0]
    for key in ("centers", "axes", "yaws", "rho"):
        assert np.array_equal(ob[key], ob3[key]), f"không tái lập: {key}"
    print(
        "  kalman không nhiễu: tâm/yaw/bán trục k=0 trùng sự thật; "
        "phồng cuối chân trời đúng a·sin(σ_yaw); σ_yaw bất biến theo seed"
    )


def test_kalman_noise_statistics():
    def ell_fn(t):
        return np.array([5.0, 0.0, 1.5]), np.ones(3), 0.0

    n_seed = 160
    cs = np.empty((n_seed, 3))
    la = np.empty((n_seed, 3))
    for s in range(n_seed):
        ob = sf.kalman_ellipsoid_obstacles(
            [ell_fn], 5.0, K, DT, np.random.default_rng(1000 + s)
        )[0]
        cs[s] = ob["centers"][0]
        la[s] = np.log(ob["axes"][0])
    c_std = cs.std(axis=0)
    c_mean_err = np.abs(cs.mean(axis=0) - np.array([5.0, 0.0, 1.5])).max()
    l_std = la.std(axis=0)
    l_mean = la.mean(axis=0)
    assert np.all(c_std > 0.005) and np.all(c_std < sf.MEAS_STD), (
        f"std tâm sau lọc phải ở (0, MEAS_STD): {c_std}"
    )
    assert c_mean_err < 0.004, f"tâm dự báo lệch hệ thống: {c_mean_err}"
    assert np.all(l_std > 0.005) and np.all(l_std < sf.LOG_MEAS_STD), (
        f"std log-bán-trục sau lọc phải ở (0, LOG_MEAS_STD): {l_std}"
    )
    assert abs(l_mean[2]) < 0.006, f"trục dọc phải không lệch: {l_mean[2]}"
    assert l_mean[0] > 0.005 and l_mean[1] > 0.005, (
        f"trục ngang phải lệch dương do tròn hoá max: {l_mean}"
    )
    print(
        f"  thống kê nhiễu ({n_seed} seed): std tâm {c_std.round(4)} "
        f"< MEAS_STD={sf.MEAS_STD}; std log-trục {l_std.round(4)} "
        f"< LOG_MEAS_STD={sf.LOG_MEAS_STD}; lệch tâm {c_mean_err:.4f}; "
        f"lệch log-trục ngang {l_mean[:2].round(4)} > 0 (tròn hoá), "
        f"dọc {l_mean[2]:.4f}"
    )


def test_episode_hook_integration():
    out = run_episode(
        "F1_thanh_dai",
        mode="kalman",
        seed=9000000,
        time_model="frozen",
        t_max=2.0,
        obs_mode="ellipsoid",
    )
    cfg = out["config"]
    assert cfg["obs_mode"] == "ellipsoid"
    assert cfg["ellipsoid_obs_source"] == "registry"
    assert len(out["cycles"]) >= 1 and out["metrics"]["status"]
    out_d = run_episode(
        "D1_cat_ngang",
        mode="kalman",
        seed=9000000,
        time_model="frozen",
        t_max=2.0,
        obs_mode="ellipsoid",
    )
    assert out_d["config"]["ellipsoid_obs_source"] == "degenerate_from_sphere"
    out_s = run_episode(
        "D1_cat_ngang", mode="kalman", seed=9000000, time_model="frozen", t_max=2.0
    )
    assert out_s["config"]["obs_mode"] == "sphere"
    assert out_s["config"]["ellipsoid_obs_source"] is None
    try:
        run_episode("D1_cat_ngang", obs_mode="lidar")
    except ValueError:
        pass
    else:
        raise AssertionError("obs_mode lạ phải bị bác bỏ")
    print(
        "  vòng lặp: F ghi source=registry, D ghi "
        "source=degenerate_from_sphere, mặc định sphere, lạ bị bác bỏ"
    )


if __name__ == "__main__":
    test_degenerate_fns_exact()
    test_truth_mode_matches_truth()
    test_frozen_mode_matches_anchor()
    test_kalman_zero_noise()
    test_kalman_noise_statistics()
    test_episode_hook_integration()
    print("All ellipsoid observation-hook tests passed.")
