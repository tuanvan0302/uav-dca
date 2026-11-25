# -*- coding: utf-8 -*-
"""
Kiểm thử core/method_adapter.py (nối A1–A3 vào runner).

Ca hồi quy chính: với đầu vào tĩnh dạng tuple (mode frozen), transform
sinh axes (3,); yaws phải là vô hướng 0,0. Nếu yaws là mảng (3,),
planner gọi float(yw) trên mảng 3 phần tử và A2/A3 crash TypeError.

Bao phủ: ba dạng đầu vào × A1/A2/A3:
  (a) tuple tĩnh  (đầu ra get_obs mode frozen);
  (b) dict dãy    (đầu ra kalman_obstacles: centers/radii/rho
                   dạng (K+1, ...)) — dùng quan sát kalman thật
                   của D1 (seed 9000000) cho khẳng định success;
  (c) dict tĩnh   (dict có radii vô hướng).
Khẳng định theo hai tầng:
  - Tầng transform (mọi dạng × mọi cấu hình): yaws khớp chiều
    thời gian của axes ((K+1,) khi axes (K+1, 3); vô hướng khi
    axes (3,)) — đây là ca hồi quy chính nêu trên.
  - Tầng plan(): không crash TypeError, total_slack hữu hạn.
    success=True chỉ khẳng định ở các ca đường ống hiện tại
    giải được: A2 trên mọi dạng tĩnh (biên legacy), và cả ba
    cấu hình trên quan sát kalman D1 (A1 đã chứng minh: 3 vòng
    lặp, slack 0). A1/A3 trên vật tĩnh chết tâm (S1) KHÔNG
    khẳng định success: với seed thẳng của chu kỳ 0, rho 2,89 m
    (need 4,59 m) làm DCA chạm trần 105 vòng với slack 33,6 m;
    với seed A* tính theo rho thì giải được (42 vòng, slack ~0),
    xem tests/test_astar_seed.py.

DV-H7 — trần T_cap + cờ drift_capped trên đường chạy thật (xem
core/method_adapter.py): ba test —
  (a) mặc định t_cap=None: rho lắp ráp khớp TỪNG BIT với công
      thức không trần tính tay qua safety_margin (window() không
      chặn) trên dãy solve_time kịch bản hoá có T_r tới 120 s;
      window_info phơi ra có T_r_eff == T_r, capped False; A2
      không có window_info.
  (b) T_cap nhỏ giả tạo 0,5 s: capped True đúng chu kỳ T_r
      vượt trần, T_r ghi ra vẫn là số không chặn, δ_trôi khớp
      công thức trần; run_summary có n_drift_capped /
      drift_capped_rate đúng.
  (c) episode A1 frozen qua runner chính thức với --t-cap 2.0:
      mọi bản ghi chu kỳ của log mang T_r / T_r_eff /
      drift_capped / T_cap; drift_capped ⟺ T_r > 2,0;
      T_r_eff = min(T_r, 2,0); drift_capped_rate trong
      run_summary khớp đếm tay từ log; provenance ghi
      t_cap_s = 2,0. Test (c) gọi experiments.experiment_runner, gói
      này nằm ngoài code/.

Chạy:  python3 -m tests.test_method_adapter   (từ gốc code/)
       hoặc pytest tests/test_method_adapter.py
"""

import numpy as np

from core.dca_planner import PlannerParams, plan
from core.method_adapter import MethodAdapter
from core.obstacle_predict import kalman_obstacles
from sim import dynamic_scenarios

K = 28
START = np.array([0.0, 0.0, 1.5])
GOAL = np.array([10.0, 0.0, 1.5])
CENTER = np.array([5.0, 0.0, 1.5])
RADIUS = 1.2
METHODS = ("A1", "A2", "A3")


def _prm(method):
    # Cùng quy ước với runner: A1/A3 gộp biên vào rho
    # nên extra_margin = 0; A2 giữ mặc định như A0.
    return (
        PlannerParams(extra_margin=0.0) if method in ("A1", "A3") else PlannerParams()
    )


def _transform(method, obstacles):
    adapter = MethodAdapter(method, eps=0.05)
    new_obs, _info = adapter.transform(obstacles, _prm(method))
    return new_obs


def _plan(method, new_obs):
    return plan(START, np.zeros(3), GOAL, new_obs, _prm(method))


# ---- dạng đầu vào ------------------------------------------------
def _tuple_static():
    return [(CENTER.copy(), RADIUS)]


def _dict_series_d1():
    """Quan sát kalman thật của D1 tại t=0, seed 9000000."""
    sc = dynamic_scenarios.dynamic_scenario("D1_cat_ngang")
    return kalman_obstacles(
        sc["truth_fns"], 0.0, K, 0.25, np.random.default_rng(9000000)
    )


def _dict_static():
    return [dict(centers=CENTER.copy(), radii=RADIUS, rho=0.0)]


def _assert_yaws_shape(method, new_obs):
    d = new_obs[0]
    axes = np.asarray(d["axes"]) if "axes" in d else None
    if method in ("A2", "A3"):
        assert axes is not None, f"{method}: thiếu axes"
        if axes.shape == (K + 1, 3):
            assert np.asarray(d["yaws"]).shape == (K + 1,), (
                f"{method}: yaws phải (K+1,), nhận {np.shape(d['yaws'])}"
            )
        elif axes.shape == (3,):
            assert np.ndim(d["yaws"]) == 0, (
                f"{method}: yaws phải vô hướng khi axes (3,), "
                f"nhận {np.shape(d['yaws'])}"
            )
            assert float(d["yaws"]) == 0.0
        else:
            raise AssertionError(f"{method}: axes lạ {axes.shape}")


def test_transform_yaws_shapes():
    """Tầng transform: yaws khớp chiều thời gian của axes, mọi dạng."""
    cases = [
        ("tuple_static", _tuple_static()),
        ("dict_series_D1", _dict_series_d1()),
        ("dict_static", _dict_static()),
    ]
    for _name, obs in cases:
        for method in METHODS:
            new_obs = _transform(method, obs)
            _assert_yaws_shape(method, new_obs)


def test_plan_no_crash_and_legacy_success():
    """Tầng plan(): không crash; A2/A3 success trên vật tĩnh;
    A1 success trên quan sát kalman D1."""
    # Vật tĩnh chết tâm (S1): chỉ A2 (biên legacy) phải success;
    # A1/A3 mang rho_traj 2,63 m nên chỉ khẳng định không crash +
    # slack hữu hạn (xem docstring).
    for name, obs in [("tuple_tinh", _tuple_static()), ("dict_tinh", _dict_static())]:
        for method in METHODS:
            new_obs = _transform(method, obs)
            res = _plan(method, new_obs)
            assert np.isfinite(res.total_slack), (name, method)
            if method == "A2":
                assert res.success, (name, method, res.total_slack, res.iterations)
    # Quan sát kalman D1: cả ba cấu hình đều success.
    for method in METHODS:
        new_obs = _transform(method, _dict_series_d1())
        res = _plan(method, new_obs)
        assert np.isfinite(res.total_slack), ("dict_day_D1", method)
        assert res.success, ("dict_day_D1", method, res.total_slack, res.iterations)


# ---- DV-H7: trần T_cap + cờ drift_capped -------------------------
from types import SimpleNamespace  # noqa: E402

from core import safety_margin as _mh  # noqa: E402


def _fake_result(solve_time, slack=0.0, iterations=3):
    return SimpleNamespace(
        iterations=iterations, total_slack=slack, solve_time=solve_time
    )


def _drive(adapter, obstacles, solve_times):
    """Chạy wrap_plan với plan() giả có solve_time kịch bản hoá.

    adapter._n_calls đặt trước = 1 để bỏ điểm kích hoạt chu kỳ 0
    của hạt giống DV-H8: t_seed = 0 tất định, dãy cập nhật đồng
    hồ H1 là đúng dãy solve_times (phép kiểm này về đồng hồ/trần,
    không phải hạt giống). Trả (rho đã đưa cho planner theo lần
    gọi — vật đầu tiên, window_info theo lần gọi)."""
    captured = []

    def fake_plan(p0, v0, goal, new_obs, prm, seed_P=None):
        ob0 = new_obs[0]
        captured.append(
            np.asarray(ob0["rho"], dtype=float).copy() if "rho" in ob0 else None
        )
        return _fake_result(solve_times[len(captured) - 1])

    adapter._n_calls = 1
    wrapped = adapter.wrap_plan(fake_plan)
    winfos = []
    for _ in solve_times:
        wrapped(START, np.zeros(3), GOAL, obstacles, _prm(adapter.method))
        winfos.append(
            None if adapter.last_window_info is None else dict(adapter.last_window_info)
        )
    return captured, winfos


def _expected_rho(adapter, centers, T_r_window):
    """rho lắp ráp tính tay qua safety_margin với cửa sổ cho trước."""
    prm = _prm(adapter.method)
    v_hat = _mh.estimate_speed(centers, prm.dt)
    dd = _mh.delta_drift(v_hat, adapter.a_max, T_r_window, _mh.TRACKING_ERROR_DEFAULT)
    ds = _mh.delta_sample(prm.u_max, adapter.a_max, prm.dt)
    return _mh.compose_rho_traj_for_planner(adapter.rho_traj, dd, ds, K=prm.K)


def test_tcap_default_off_bit_identical():
    """(a) Mặc định tắt: rho khớp từng bit với công thức không
    trần; window_info có T_r_eff == T_r, capped False, T_cap
    None ở mọi chu kỳ — kể cả khi T_r = 120,25 s."""
    obs = _dict_series_d1()
    centers = np.asarray(obs[0]["centers"], dtype=float)
    adapter = MethodAdapter("A1", eps=0.05)
    assert adapter.t_cap is None
    solve_times = [0.30, 5.0, 120.0, 0.20]
    captured, winfos = _drive(adapter, obs, solve_times)
    clock = _mh.ReplanClock(T_replan=_prm("A1").dt)
    for i, st in enumerate(solve_times):
        T_r = clock.window()
        assert np.array_equal(captured[i], _expected_rho(adapter, centers, T_r)), i
        assert winfos[i]["T_r"] == T_r
        assert winfos[i]["T_r_eff"] == T_r
        assert winfos[i]["capped"] is False
        assert winfos[i]["T_cap"] is None
        clock.update(st)  # t_seed = 0 (không kích hoạt — xem _drive)
    # A2 không có hạng tử trôi: không phơi window_info.
    a2 = MethodAdapter("A2")
    a2._n_calls = 1
    wrapped2 = a2.wrap_plan(lambda *a, **k: _fake_result(0.1))
    wrapped2(START, np.zeros(3), GOAL, obs, _prm("A2"))
    assert a2.last_window_info is None
    assert wrapped2.last_window_info is None


def test_tcap_artificial_cap():
    """(b) T_cap = 0,5 s giả tạo: T_r dãy [0,25; 0,35; 5,25] ->
    capped [False; False; True]; T_r ghi ra là số KHÔNG chặn;
    δ_trôi của chu kỳ bị chặn khớp công thức trần (cửa sổ 0,5);
    run_summary đếm đúng."""
    obs = _dict_series_d1()
    centers = np.asarray(obs[0]["centers"], dtype=float)
    adapter = MethodAdapter("A1", eps=0.05, t_cap=0.5)
    solve_times = [0.10, 5.0, 0.10]
    captured, winfos = _drive(adapter, obs, solve_times)
    T_rs = [0.25, 0.35, 5.25]
    for i, T_r in enumerate(T_rs):
        eff = min(T_r, 0.5)
        assert winfos[i]["T_r"] == T_r, i
        assert winfos[i]["T_r_eff"] == eff, i
        assert winfos[i]["capped"] is (T_r > 0.5), i
        assert winfos[i]["T_cap"] == 0.5
        assert np.array_equal(captured[i], _expected_rho(adapter, centers, eff)), i
    summ = adapter.summary()
    assert summ["n_drift_capped"] == 1
    assert summ["drift_capped_rate"] == 1 / 3
    assert len(summ["window_records"]) == 3
    assert adapter.provenance()["t_cap_s"] == 0.5


def test_runner_episode_tcap_log_fields():
    """(c) Episode A1 frozen qua runner chính thức với
    --t-cap 2.0: cờ drift_capped tồn tại trong log đường sản
    xuất ở MỌI chu kỳ và nhất quán số học với T_r/T_r_eff;
    tỉ lệ trong run_summary khớp đếm tay từ chính log đó."""
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    code_root = Path(__file__).resolve().parent.parent  # gốc code/
    with tempfile.TemporaryDirectory() as td:
        cmd = [
            sys.executable,
            "-m",
            "experiments.experiment_runner",
            "--method",
            "A1",
            "--scenario",
            "S1_mot_vat",
            "--seed",
            "9000000",
            "--mode",
            "frozen",
            "--time-model",
            "frozen",
            "--t-max",
            "3.0",
            "--t-cap",
            "2.0",
            "--outdir",
            td,
        ]
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, cwd=code_root
        )
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        logs = list(Path(td).glob("A1_S1_mot_vat_frozen_frozen_seed9000000*.json"))
        assert len(logs) == 1, [p.name for p in Path(td).iterdir()]
        log = json.loads(logs[0].read_text(encoding="utf-8"))
    cycles = log["cycles"]
    assert len(cycles) >= 4, len(cycles)
    n_cap = 0
    for c in cycles:
        for key in ("T_r", "T_r_eff", "drift_capped", "T_cap"):
            assert key in c, (key, c.get("cycle"))
        assert c["T_cap"] == 2.0
        assert c["drift_capped"] == (c["T_r"] > 2.0)
        assert c["T_r_eff"] == min(c["T_r"], 2.0)
        n_cap += bool(c["drift_capped"])
    prov = log["config"]["method_adapter"]
    assert prov["t_cap_s"] == 2.0
    rs = prov["run_summary"]
    assert len(rs["window_records"]) == len(cycles)
    assert rs["n_drift_capped"] == n_cap
    assert abs(rs["drift_capped_rate"] - n_cap / len(cycles)) < 1e-12


if __name__ == "__main__":
    tests = [
        test_transform_yaws_shapes,
        test_plan_no_crash_and_legacy_success,
        test_tcap_default_off_bit_identical,
        test_tcap_artificial_cap,
        test_runner_episode_tcap_log_fields,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:  # noqa: BLE001 - báo cáo rồi đếm lỗi
            failed += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
