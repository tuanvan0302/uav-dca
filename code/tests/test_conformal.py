# -*- coding: utf-8 -*-
"""
test_conformal.py — Kiểm thử conformal.py:
  1) Công thức quantile hữu hạn mẫu ⌈(n+1)(1−ε)⌉ trên mẫu nhỏ tính tay.
  2) Trên phân phối ĐÃ BIẾT (nửa chuẩn |N(0,1)|): bao phủ thực nghiệm
     khớp mức danh định 1−ε trong dung sai thống kê.
  3) Bảng conformal: xây dựng → lưu → nạp → tra cứu; dự phòng nhóm gộp.
  4) Điểm số mức quỹ đạo và biên Bonferroni: bao phủ ≥ danh định
     (trong dung sai), và biên Bonferroni từng bước ≥ biên từng bước
     thông thường tại cùng ε (tính đơn điệu theo mức quantile).
  5) Giao diện kênh hình dạng (shape_j) cho H2.

Chạy: python3 -m tests.test_conformal   (từ gốc code/)
      hoặc pytest tests/test_conformal.py
"""

import os
import tempfile

import numpy as np

from core import conformal as cf


def test_quantile_formula_hand_computed():
    s = np.arange(1, 11, dtype=float)  # n = 10: 1..10
    rho, rank, n = cf.conformal_quantile(s, 0.10)
    assert (rho, rank, n) == (10.0, 10, 10)  # ⌈11·0,9⌉ = 10
    rho, rank, n = cf.conformal_quantile(s, 0.50)
    assert (rho, rank) == (6.0, 6)  # ⌈11·0,5⌉ = 6
    # n = 10, ε = 0,05: ⌈11·0,95⌉ = 11 > n → CLAMP về S_(n) = 10
    # (giao thức §4.2), không trả +inf.
    rho, rank, n = cf.conformal_quantile(s, 0.05)
    assert rho == 10.0 and rank == 11
    # n = 19 là cỡ tối thiểu cho ε = 0,05: ⌈20·0,95⌉ = 19
    rho, rank, n = cf.conformal_quantile(np.arange(19.0), 0.05)
    assert rank == 19 and rho == 18.0


def _coverage_tol(p, n, n_cal=None, z=3.0):
    """Dung sai bao phủ: gồm phương sai lấy mẫu của tập ĐÁNH GIÁ (n)
    và phương sai của chính quantile hiệu chuẩn (n_cal) — bao phủ có
    điều kiện trên tập hiệu chuẩn dao động quanh 1−ε với độ lệch
    ~sqrt(ε(1−ε)/n_cal) (phân phối Beta của thống kê thứ tự)."""
    var = p * (1 - p) / n
    if n_cal:
        var += p * (1 - p) / n_cal
    return z * np.sqrt(var)


def test_synthetic_known_distribution_coverage():
    rng = np.random.default_rng(20261121)
    n_cal, n_eval = 4000, 20000
    cal = np.abs(rng.normal(0.0, 1.0, n_cal))  # |N(0,1)|
    ev = np.abs(rng.normal(0.0, 1.0, n_eval))
    for eps in (0.05, 0.10):
        rho, _, _ = cf.conformal_quantile(cal, eps)
        cov = float((ev <= rho).mean())
        tol = _coverage_tol(1 - eps, n_eval, n_cal=n_cal)
        # Conformal: bao phủ ≥ 1−ε (hữu hạn mẫu), và không bảo thủ quá
        assert cov >= 1 - eps - tol, (eps, cov)
        assert cov <= 1 - eps + tol + 0.005, (eps, cov)


def _synthetic_recorder(n_runs=300, K=28, seed=7):
    """Recorder tổng hợp: sai số tăng theo k (mô phỏng chân trời)."""
    rng = np.random.default_rng(seed)
    rec = cf.ScoreRecorder(K=K)
    ks = np.arange(K + 1)
    for _ in range(n_runs):
        base_c = rng.uniform(4.0, 6.0, 3)
        base_r = rng.uniform(0.8, 1.2)
        ct = np.tile(base_c, (K + 1, 1))
        rt = np.full(K + 1, base_r)
        cp = ct + rng.normal(0, 0.02, (K + 1, 3)) * (1 + ks[:, None] / K)
        rp = rt + rng.normal(0, 0.01, K + 1) * (1 + ks / K)
        rec.record_forecast(ct, rt, cp, rp, "age_mid")
    return rec


def test_table_build_save_load_lookup():
    rec = _synthetic_recorder()
    tab = cf.ConformalTable.build(rec)
    arr = tab.rho_array("comb", "age_mid", 0.05)
    assert arr.shape == (29,)
    assert np.all(np.isfinite(arr))
    # ρ tăng theo k (sai số tổng hợp tăng theo chân trời)
    assert arr[-1] > arr[0]
    # ε nhỏ hơn → biên lớn hơn hoặc bằng
    assert np.all(
        tab.rho_array("comb", "age_mid", 0.05) >= tab.rho_array("comb", "age_mid", 0.10)
    )
    # Lưu/nạp khớp nhau
    with tempfile.TemporaryDirectory() as d:
        prefix = os.path.join(d, "tab")
        tab.save(prefix)
        tab2 = cf.ConformalTable.load(prefix)
        assert np.allclose(tab2.rho_array("comb", "age_mid", 0.05), arr)
        assert tab2.rho_traj("age_mid", 0.05) == tab.rho_traj("age_mid", 0.05)
    # Nhóm tuổi không tồn tại → dự phòng nhóm gộp "__all__"
    arr_pool = tab.rho_array("comb", "age_khong_ton_tai", 0.05)
    assert np.allclose(arr_pool, tab.rho_array("comb", cf.POOLED, 0.05))


def test_traj_and_bonferroni_coverage():
    K = 28
    rec = _synthetic_recorder(n_runs=800, K=K, seed=11)
    tab = cf.ConformalTable.build(rec)
    # Biên Bonferroni từng bước ≥ biên từng bước thường tại cùng ε
    assert np.all(
        tab.rho_bonf_array("age_mid", 0.05) >= tab.rho_array("comb", "age_mid", 0.05)
    )
    # Bao phủ mức quỹ đạo trên mẫu đánh giá tổng hợp độc lập
    rng = np.random.default_rng(99)
    ks = np.arange(K + 1)
    n_eval, ok_traj, ok_bonf = 2000, 0, 0
    rho_t = tab.rho_traj("age_mid", 0.05)
    rho_b = tab.rho_bonf_array("age_mid", 0.05)
    for _ in range(n_eval):
        e = np.abs(rng.normal(0, 0.02, K + 1)) * (1 + ks / K) + np.abs(
            rng.normal(0, 0.01, K + 1)
        ) * (1 + ks / K)
        ok_traj += bool(e.max() <= rho_t)
        ok_bonf += bool(np.all(e <= rho_b))
    tol = _coverage_tol(0.95, n_eval, n_cal=800)
    assert ok_traj / n_eval >= 0.95 - tol
    assert ok_bonf / n_eval >= 0.95 - tol


def test_shape_channel_interface():
    K = 28
    rec = cf.ScoreRecorder(K=K)
    rng = np.random.default_rng(3)
    lt = np.log(np.tile([1.5, 0.8, 0.8], (K + 1, 1)))
    lp = lt + rng.normal(0, 0.03, (K + 1, 3))
    for _ in range(60):
        rec.record_shape_errors(lt, lp + rng.normal(0, 0.01, (K + 1, 3)), "age_old")
    tab = cf.ConformalTable.build(rec)
    v = tab.rho("shape_0", 10, "age_old", 0.10)
    assert np.isfinite(v) and v > 0


def test_comb_score_is_sum_not_max():
    # Giao thức §4.2: s_k = e_vt,k + e_ht,k — TỔNG các kênh theo mét.
    # Dạng max sẽ đánh giá thấp: ví dụ max ghi 0,30 m khi độ dời bề mặt
    # thật ≈ 0,61 m. Test khóa dạng tổng.
    K = 4
    rec = cf.ScoreRecorder(K=K)
    ct = np.zeros((K + 1, 3))
    cp = ct.copy()
    cp[:, 0] = 0.30  # e_pos = 0,30 mọi k
    rt = np.full(K + 1, 1.0)
    rp = rt + 0.10  # e_rad = 0,10 mọi k
    s = rec.record_forecast(ct, rt, cp, rp, "age_new")
    comb = rec.stack("comb", "age_new")[0]
    assert np.allclose(comb, 0.40)  # 0,30 + 0,10 (tổng)
    assert abs(s - 0.40) < 1e-12


def test_roundtrip_fallback_on_loaded_table():
    # Sau save→load, fallback về nhóm gộp "__all__" phải hoạt động cho
    # MỌI kênh, kể cả các kênh step.
    rec = _synthetic_recorder()
    tab = cf.ConformalTable.build(rec)
    with tempfile.TemporaryDirectory() as d:
        prefix = os.path.join(d, "tab")
        tab.save(prefix)
        tab2 = cf.ConformalTable.load(prefix)
    # Khóa step nạp từ đĩa khớp hệt khóa trong bộ nhớ
    assert set(tab2.step.keys()) == set(tab.step.keys())
    assert ("comb", cf.POOLED) in tab2.step
    for ch in ("pos", "rad", "comb"):
        arr = tab2.rho_array(ch, "age_khong_co", 0.05)  # không KeyError
        assert np.allclose(arr, tab.rho_array(ch, cf.POOLED, 0.05))
        v = tab2.rho(ch, 7, "age_khong_co", 0.10)
        assert np.isfinite(v)
    # traj và bonf cũng fallback trên bảng đã nạp
    assert tab2.rho_traj("age_khong_co", 0.05) == tab.rho_traj(cf.POOLED, 0.05)
    assert np.allclose(
        tab2.rho_bonf_array("age_khong_co", 0.05), tab.rho_bonf_array(cf.POOLED, 0.05)
    )


def test_calibration_family_exclusion():
    # §4.3: tham số hiệu chuẩn trong khoảng đóng băng, KHÔNG nằm
    # trong hộp ±0,05 quanh vectơ đánh giá, không trùng vectơ đó.
    instances, audit = cf.draw_calibration_instances(12, seed=123)
    assert len(instances) == 36
    for inst in instances:
        spec = cf.CAL_FAMILIES[inst["family"]]
        for k, (lo, hi) in spec["ranges"].items():
            assert lo <= inst["params"][k] <= hi
        assert not cf.in_exclusion_box(inst["params"], spec)
        assert inst["params"] != spec["eval_vector"]
    # Hộp loại trừ bắt đúng vectơ đánh giá
    for spec in cf.CAL_FAMILIES.values():
        assert cf.in_exclusion_box(spec["eval_vector"], spec)


def test_shape_surface_yaw_interface():
    # Giao diện §4.2 cho tầng hình dạng: kênh bề mặt theo mét TRỰC
    # TIẾP |a − â|, kênh yaw theo DÂY CUNG a_max·2·sin(|Δψ|/2),
    # comb_full = pos + shape_surf + yaw (TỔNG các kênh).
    K = 28
    rec = cf.ScoreRecorder(K=K)
    ct = np.tile([5.0, 0.0, 1.5], (K + 1, 1))
    axes_t = np.tile([1.6, 0.5, 0.5], (K + 1, 1))
    axes_p = axes_t.copy()
    yaw_t = np.zeros(K + 1)
    yaw_p = np.full(K + 1, np.deg2rad(10.0))  # lệch yaw 10°
    s = rec.record_shape_forecast(
        ct, ct.copy(), axes_t, axes_p, yaw_t, yaw_p, "age_mid"
    )
    yaw_row = rec.stack("yaw", "age_mid")[0]
    # e_yaw = a_max·2·sin(10°/2) = 1,6·2·sin(5°) ≈ 0,2789 m tại mọi k
    assert np.allclose(yaw_row, 1.6 * 2.0 * np.sin(np.deg2rad(10.0) / 2.0))
    assert np.allclose(rec.stack("shape_surf", "age_mid")[0], 0.0)
    assert abs(s - 1.6 * 2.0 * np.sin(np.deg2rad(10.0) / 2.0)) < 1e-12
    tab = cf.ConformalTable.build(rec)
    assert np.isfinite(tab.rho("comb_full", 3, "age_mid", 0.10))
    # e_ht trực tiếp theo mét: â = (1,8; 0,5; 0,5) → |Δa| lớn nhất
    # = 0,2 m (dạng log â·|Δlog a| sẽ cho 1,8·log(1,8/1,6) ≈ 0,212,
    # nên test phân biệt được hai dạng); comb_full là TỔNG.
    rec2 = cf.ScoreRecorder(K=K)
    axes_p2 = np.tile([1.8, 0.5, 0.5], (K + 1, 1))
    s2 = rec2.record_shape_forecast(
        ct, ct.copy(), axes_t, axes_p2, yaw_t, yaw_t.copy(), "age_mid"
    )
    assert np.allclose(rec2.stack("shape_surf", "age_mid")[0], 0.2)
    assert abs(s2 - 0.2) < 1e-12  # pos 0 + surf 0,2 + yaw 0


def test_shape_comb_full_is_sum_multichannel():
    # Phép ghép TỔNG của comb_full phải được khóa khi ≥ 2 kênh khác 0
    # ĐỒNG THỜI — hai ca của
    # test_shape_surface_yaw_interface đều chỉ có một kênh khác 0
    # nên dạng max ≡ dạng tổng trên chúng (kiểm đột biến: đổi
    # e_full sang max không làm test nào trượt). Ca này dùng cả
    # ba kênh cùng khác 0, giá trị kỳ vọng tính tay:
    #   e_pos  = 0,30 (dịch tâm 0,30 m theo x)
    #   e_surf = 0,20 (|1,8 − 1,6|, trực tiếp theo mét)
    #   e_yaw  = 1,6·2·sin(5°) ≈ 0,2789 (lệch yaw 10°, a_max thật 1,6)
    #   TỔNG   = 0,30 + 0,20 + 0,2789 ≈ 0,7789 ở mọi k;
    # dạng max sẽ cho 0,30 — khác hẳn, test này phải trượt nếu
    # comb_full bị đổi về max.
    K = 4
    ct = np.tile([5.0, 0.0, 1.5], (K + 1, 1))
    cp = ct.copy()
    cp[:, 0] += 0.30  # e_pos = 0,30 mọi k
    axes_t = np.tile([1.6, 0.5, 0.5], (K + 1, 1))
    axes_p = np.tile([1.8, 0.5, 0.5], (K + 1, 1))  # e_surf = 0,20
    yaw_t = np.zeros(K + 1)
    yaw_p = np.full(K + 1, np.deg2rad(10.0))
    e_yaw = 1.6 * 2.0 * np.sin(np.deg2rad(10.0) / 2.0)
    # Ca 3 kênh cùng khác 0: pos + surf + yaw
    rec = cf.ScoreRecorder(K=K)
    s = rec.record_shape_forecast(ct, cp, axes_t, axes_p, yaw_t, yaw_p, "age_mid")
    assert np.allclose(rec.stack("pos", "age_mid")[0], 0.30)
    assert np.allclose(rec.stack("shape_surf", "age_mid")[0], 0.20)
    assert np.allclose(rec.stack("yaw", "age_mid")[0], e_yaw)
    assert np.allclose(rec.stack("comb_full", "age_mid")[0], 0.30 + 0.20 + e_yaw)
    assert abs(s - (0.30 + 0.20 + e_yaw)) < 1e-12
    # Ca 2 kênh cùng khác 0: pos + surf, yaw = 0 → TỔNG = 0,50
    # (dạng max cho 0,30 — vẫn phân biệt được).
    rec2 = cf.ScoreRecorder(K=K)
    s2 = rec2.record_shape_forecast(
        ct, cp, axes_t, axes_p, yaw_t, yaw_t.copy(), "age_mid"
    )
    assert np.allclose(rec2.stack("comb_full", "age_mid")[0], 0.50)
    assert abs(s2 - 0.50) < 1e-12


def test_clopper_pearson_sanity():
    lo, hi = cf.clopper_pearson_ci(95, 100)
    assert lo < 0.95 < hi
    lo, hi = cf.clopper_pearson_ci(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = cf.clopper_pearson_ci(100, 100)
    assert hi == 1.0 and 0.95 < lo < 1.0
    # CP rộng hơn hoặc bằng Wilson ở cùng dữ liệu (bảo thủ hơn)
    wlo, whi = cf.wilson_ci(95, 100)
    lo, hi = cf.clopper_pearson_ci(95, 100)
    assert hi - lo >= whi - wlo - 1e-9


if __name__ == "__main__":
    fns = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")
