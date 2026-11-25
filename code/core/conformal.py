# -*- coding: utf-8 -*-
"""
conformal.py — hiệu chuẩn split conformal cho hạng tử bất định ρ(ε) của
biên an toàn ST-DCA (xem core/safety_margin.py).

Ý tưởng: chạy bộ dự báo trên một tập hiệu chuẩn tách riêng, ghi lại các
điểm số không phù hợp s (sai số dự báo, đơn vị mét), rồi lấy ρ(ε) là
quantile hữu hạn mẫu bậc ⌈(n+1)(1−ε)⌉ của n điểm số. Nếu dữ liệu hiệu
chuẩn và dữ liệu lúc chạy trao đổi được (exchangeable) thì
P(s ≤ ρ(ε)) ≥ 1 − ε.

Điểm số tại bước k:
  - Vật cản cầu (record_forecast):
        s_k = e_vt,k + e_ht,k,   e_vt = ‖ô − o‖,  e_ht = |r̂ − r|.
    Dùng tổng vì: nếu planner bảo đảm ‖p − ô‖ ≥ r̂ + d + ρ và s_k ≤ ρ thì
    theo bất đẳng thức tam giác
        ‖p − o‖ ≥ ‖p − ô‖ − e_vt ≥ r̂ + d + ρ − e_vt ≥ r + d + ρ − s_k ≥ r + d.
  - Vật cản ellipsoid (record_shape_forecast):
        e_ht  = max_j |a_j − â_j|,
        e_yaw = a_max · 2·sin(|ψ − ψ̂|/2)  (độ dài dây cung khi quay một
                điểm cách tâm a_max một góc |ψ − ψ̂|),
        s_k   = e_vt + e_ht + e_yaw.
    Module chỉ cung cấp hàm ghi các kênh này; đường ống hiệu chuẩn hiện
    chạy cho vật cản cầu.

Các loại quantile trong bảng:
  - step : quantile riêng cho từng bước k, ρ_k(ε).
  - traj : quantile của điểm số mức quỹ đạo S = max_k s_k (và max qua các
           vật cản của cùng một lần chạy). Dùng cùng ρ_traj(ε) cho mọi k
           thì mọi bước được phủ đồng thời với xác suất ≥ 1 − ε. Đây là giá
           trị planner dùng.
  - bonf : quantile từng bước ở mức ε/(K+1) (hiệu chỉnh Bonferroni), một
           cách khác để phủ đồng thời mọi k.

Tập hiệu chuẩn: tham số kịch bản được rút ngẫu nhiên từ các khoảng cố định
của ba họ một vật cản cầu (cắt ngang, đối đầu, biến dạng). Lần rút nào rơi
vào hộp ±0,05 (toạ độ chuẩn hoá về [0, 1]) quanh vectơ tham số của kịch
bản đánh giá tương ứng (D1/D2/D3) thì bị loại, để tập hiệu chuẩn không
chứa kịch bản đánh giá.

Luồng ngẫu nhiên: mỗi ô (thực thể × t_now × n_hist, hoặc kịch bản × t_now
× n_hist × seed khi đánh giá) có luồng RNG riêng suy ra bằng SeedSequence,
nên các ô độc lập với nhau.

Nhóm tuổi: ρ được tính riêng cho từng nhóm tuổi theo dõi của vật cản (mới
≤ 2, trung 3–8, lâu > 8 lần phát hiện). Ở đây số lần đo lịch sử n_hist
đóng vai trò tuổi: 2 → age_new; 4, 8 → age_mid; 12 → age_old. Nhóm
"__all__" gộp mọi nhóm.

Khoảng tin cậy cho tỉ lệ bao phủ: Clopper–Pearson. Với bao phủ từng bước,
các bước k trong cùng một hàng dự báo tương quan với nhau nên dùng thêm
bootstrap theo cụm hàng; với bao phủ mức lần chạy, dùng thêm bootstrap
theo cụm seed đánh giá.

Lưu trữ: bảng được lưu thành <prefix>.npz (các mảng) và <prefix>.json
(K, các mức ε, cỡ mẫu, meta). Khoá mảng có dạng "step::<kênh>::<nhóm>",
"traj::<nhóm>", "bonf::<nhóm>".

Phụ thuộc: core.obstacle_predict (bộ dự báo Kalman) và
sim.dynamic_scenarios (sự thật nền của các kịch bản).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as _stats

from core import obstacle_predict as _op
from sim import dynamic_scenarios as _ds

EPS_LEVELS = (0.05, 0.10)  # các mức ε có trong bảng
POOLED = "__all__"  # tên nhóm gộp mọi nhóm tuổi
PROTOCOL_VERSION = "1.2"  # phiên bản giao thức thực nghiệm, ghi vào log

# Ánh xạ n_hist (số lần đo lịch sử cấp cho bộ dự báo) -> nhóm tuổi theo
# dõi (mới ≤ 2, trung 3–8, lâu > 8 lần phát hiện liên tiếp).
AGE_GROUPS = {2: "age_new", 4: "age_mid", 8: "age_mid", 12: "age_old"}

# Gốc seed: luồng hiệu chuẩn dùng CAL_STREAM_BASE = 2_000_000; seed đánh giá
# thứ i là 1_000_000 + 7_919·i.
CAL_STREAM_BASE = 2_000_000
EVAL_SEED_BASE = 1_000_000
EVAL_SEED_STEP = 7_919
BOOTSTRAP_SEED = 424242  # seed cố định cho bootstrap (CI tái lập được)


def eval_seed(index: int) -> int:
    """Seed đánh giá thứ i: 1_000_000 + 7_919·i."""
    return EVAL_SEED_BASE + EVAL_SEED_STEP * int(index)


def age_group_of(n_hist: int) -> str:
    """Tên nhóm tuổi ứng với n_hist; báo lỗi nếu n_hist không có trong
    AGE_GROUPS."""
    if n_hist not in AGE_GROUPS:
        raise ValueError(
            f"n_hist={n_hist} has no age group; valid: {sorted(AGE_GROUPS)}"
        )
    return AGE_GROUPS[n_hist]


def _cell_rng(*entropy) -> np.random.Generator:
    """Luồng RNG riêng của một ô, suy ra xác định từ các số nguyên
    entropy (seed gốc và chỉ số của ô)."""
    return np.random.default_rng(np.random.SeedSequence([int(e) for e in entropy]))


# ------------------------------------------------------------------
# Quantile conformal + khoảng tin cậy
# ------------------------------------------------------------------
def conformal_quantile(scores, eps: float):
    """Quantile split conformal hữu hạn mẫu.

    Sắp xếp điểm số S_(1) ≤ ... ≤ S_(n), lấy rank = ⌈(n+1)(1−ε)⌉ và trả
    (rho = S_(rank), rank, n). Khi n quá nhỏ so với ε thì rank > n; theo lý
    thuyết khi đó ρ = +∞, còn hàm này trả S_(n) (điểm số lớn nhất), tức là
    không còn bảo đảm 1 − ε.
    """
    s = np.sort(np.asarray(scores, dtype=float).ravel())
    n = int(s.size)
    if n == 0:
        raise ValueError("empty score set")
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must lie in (0, 1)")
    rank = int(np.ceil((n + 1) * (1.0 - eps)))
    rank = max(rank, 1)
    if rank > n:
        return float(s[n - 1]), rank, n
    return float(s[rank - 1]), rank, n


def clopper_pearson_ci(n_success: int, n_total: int, alpha: float = 0.05):
    """Khoảng tin cậy Clopper–Pearson (chính xác) mức 1 − alpha cho tỉ lệ
    n_success/n_total, tính từ phân vị của phân phối beta. Dùng cho mọi
    tỉ lệ bao phủ trong module."""
    if n_total == 0:
        return (float("nan"), float("nan"))
    k, n = int(n_success), int(n_total)
    lo = 0.0 if k == 0 else float(_stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def wilson_ci(n_success: int, n_total: int, z: float = 1.96):
    """Khoảng tin cậy Wilson (mặc định 95%) cho một tỉ lệ. Các tỉ lệ bao
    phủ trong module dùng clopper_pearson_ci, không dùng hàm này."""
    if n_total == 0:
        return (float("nan"), float("nan"))
    p = n_success / n_total
    den = 1.0 + z**2 / n_total
    cen = (p + z**2 / (2 * n_total)) / den
    half = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / den
    return (float(max(0.0, cen - half)), float(min(1.0, cen + half)))


def bootstrap_pooled_coverage_ci(
    row_ok_counts, row_n_points: int, B: int = 2000, seed: int = BOOTSTRAP_SEED
):
    """CI bootstrap 95% theo cụm hàng cho bao phủ gộp mọi bước k.

    Mỗi hàng (một lần dự báo cho một vật cản, gồm row_n_points bước k) là
    một cụm: các bước trong cùng hàng tương quan, còn các hàng độc lập vì
    mỗi ô có luồng RNG riêng. row_ok_counts[i] = số bước được phủ trong
    hàng i. Mỗi lần lặp rút lại n hàng có hoàn lại."""
    counts = np.asarray(row_ok_counts, dtype=float)
    n = counts.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        reps[b] = counts[idx].sum() / (n * row_n_points)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return (float(lo), float(hi))


def bootstrap_cluster_ratio_ci(clusters, B: int = 2000, seed: int = BOOTSTRAP_SEED):
    """CI bootstrap 95% theo cụm cho một tỉ lệ gộp. clusters là danh sách
    (số đạt, cỡ cụm), ví dụ mỗi cụm là một seed đánh giá. Mỗi lần lặp rút
    lại các cụm có hoàn lại rồi tính tỉ lệ gộp."""
    cl = [(float(k), float(n)) for k, n in clusters if n > 0]
    if not cl:
        return (float("nan"), float("nan"))
    ks = np.array([c[0] for c in cl])
    ns = np.array([c[1] for c in cl])
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(cl), size=len(cl))
        reps[b] = ks[idx].sum() / ns[idx].sum()
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return (float(lo), float(hi))


def bootstrap_quantile_ci(
    scores, eps: float, B: int = 1000, seed: int = BOOTSTRAP_SEED
):
    """CI bootstrap 95% của chính giá trị ρ(ε): cho biết ρ dao động bao
    nhiêu do tập hiệu chuẩn có cỡ hữu hạn."""
    s = np.asarray(scores, dtype=float).ravel()
    n = s.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        sample = s[rng.integers(0, n, size=n)]
        reps[b], _, _ = conformal_quantile(sample, eps)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return (float(lo), float(hi))


# ------------------------------------------------------------------
# Ghi điểm số
# ------------------------------------------------------------------
@dataclass
class ScoreRecorder:
    """Thu thập điểm số không phù hợp (nonconformity scores).

    step_scores[(kênh, nhóm)]: danh sách hàng (K+1,), mỗi hàng là điểm số
        theo k của một lần dự báo cho một vật cản.
    traj_scores[nhóm]: danh sách điểm số mức quỹ đạo S = max_k s_k. Khi một
        lần chạy có nhiều vật cản, phía gọi lấy thêm max qua các vật cản và
        ghi một giá trị cho cả lần chạy bằng record_traj_score().
    Mỗi giá trị được ghi vào cả nhóm tuổi của nó và nhóm gộp POOLED.
    """

    K: int
    step_scores: dict = field(default_factory=dict)
    traj_scores: dict = field(default_factory=dict)

    def _append(self, key, row):
        # Thêm một hàng (K+1,) vào step_scores[key], kiểm tra kích thước.
        row = np.asarray(row, dtype=float)
        if row.shape != (self.K + 1,):
            raise ValueError(
                f"score row must have shape ({self.K + 1},), got {row.shape}"
            )
        self.step_scores.setdefault(key, []).append(row)

    def record_forecast(
        self,
        centers_true,
        radii_true,
        centers_pred,
        radii_pred,
        age_group: str,
        record_traj: bool = True,
    ) -> float:
        """Ghi một lần dự báo cho một vật cản cầu. Các kênh:
          pos  = ‖ô − o‖,
          rad  = |r̂ − r|,
          comb = pos + rad   (điểm số dùng để hiệu chuẩn).
        Nếu record_traj, ghi luôn S = max_k comb làm điểm số mức quỹ đạo.
        Trả S của vật cản này."""
        ct = np.asarray(centers_true, dtype=float)
        cp = np.asarray(centers_pred, dtype=float)
        rt = np.asarray(radii_true, dtype=float)
        rp = np.asarray(radii_pred, dtype=float)
        e_pos = np.linalg.norm(cp - ct, axis=1)
        e_rad = np.abs(rp - rt)
        e_comb = e_pos + e_rad
        for grp in (age_group, POOLED):
            self._append(("pos", grp), e_pos)
            self._append(("rad", grp), e_rad)
            self._append(("comb", grp), e_comb)
        if record_traj:
            self.record_traj_score(age_group, float(e_comb.max()))
        return float(e_comb.max())

    def record_traj_score(self, age_group: str, s_run: float) -> None:
        """Ghi điểm số mức quỹ đạo của một lần chạy:
        S = max qua mọi vật cản và mọi k của điểm số ghép."""
        for grp in (age_group, POOLED):
            self.traj_scores.setdefault(grp, []).append(float(s_run))

    def record_shape_forecast(
        self,
        centers_true,
        centers_pred,
        axes_true,
        axes_pred,
        yaw_true,
        yaw_pred,
        age_group: str,
        record_traj: bool = True,
    ) -> float:
        """Ghi một lần dự báo cho một vật cản ellipsoid.
        axes_*: (K+1, 3) bán trục thật/dự báo; yaw_*: (K+1,) yaw (rad).
        Các kênh (đơn vị mét):
          pos        = ‖ô − o‖,
          shape_surf = max_j |a_j − â_j|,
          yaw        = a_max(thật) · 2·sin(|ψ − ψ̂|/2)  (dây cung),
          comb_full  = pos + shape_surf + yaw.
        Nếu record_traj, ghi max_k comb_full vào traj_scores dưới khoá
        "full|<nhóm>". Trả max_k comb_full của vật cản này."""
        ct = np.asarray(centers_true, dtype=float)
        cp = np.asarray(centers_pred, dtype=float)
        at = np.asarray(axes_true, dtype=float)
        ap = np.asarray(axes_pred, dtype=float)
        yt = np.asarray(yaw_true, dtype=float)
        yp = np.asarray(yaw_pred, dtype=float)
        if at.shape != (self.K + 1, 3) or ap.shape != at.shape:
            raise ValueError("axes must have shape (K+1, 3)")
        e_pos = np.linalg.norm(cp - ct, axis=1)
        e_surf = np.max(np.abs(at - ap), axis=1)
        e_yaw = np.max(at, axis=1) * 2.0 * np.sin(np.abs(yt - yp) / 2.0)
        e_full = e_pos + e_surf + e_yaw
        for grp in (age_group, POOLED):
            self._append(("pos", grp), e_pos)
            self._append(("shape_surf", grp), e_surf)
            self._append(("yaw", grp), e_yaw)
            self._append(("comb_full", grp), e_full)
        if record_traj:
            for grp in (age_group, POOLED):
                self.traj_scores.setdefault(f"full|{grp}", []).append(
                    float(e_full.max())
                )
        return float(e_full.max())

    def record_shape_errors(self, log_axes_true, log_axes_pred, age_group: str) -> None:
        """Ghi kênh shape_j = |log â_j − log a_j| cho từng trục j (không thứ
        nguyên). Các kênh theo mét của ellipsoid nằm ở
        record_shape_forecast()."""
        lt = np.asarray(log_axes_true, dtype=float)
        lp = np.asarray(log_axes_pred, dtype=float)
        if lt.shape != (self.K + 1, 3) or lp.shape != lt.shape:
            raise ValueError("log_axes must have shape (K+1, 3)")
        for j in range(3):
            err = np.abs(lp[:, j] - lt[:, j])
            for grp in (age_group, POOLED):
                self._append((f"shape_{j}", grp), err)

    def stack(self, channel: str, group: str) -> np.ndarray:
        """Ghép các hàng của (kênh, nhóm) thành ma trận (số hàng, K+1)."""
        rows = self.step_scores.get((channel, group), [])
        if not rows:
            return np.empty((0, self.K + 1))
        return np.vstack(rows)


# ------------------------------------------------------------------
# Họ kịch bản hiệu chuẩn (một vật cản cầu, tham số rút ngẫu nhiên).
# Mỗi hàm dựng fn(t) -> (tâm (3,), bán kính).
# ------------------------------------------------------------------
def _crossing_fn(v, y0, r):
    # Cắt ngang: vật đi theo trục y với vận tốc v, xuất phát tại y0.
    def fn(t):
        return np.array([5.0, y0 + v * t, 1.5]), float(r)

    return fn


def _headon_fn(v, off, r):
    # Đối đầu: vật đi ngược trục x từ x = 11 với vận tốc v, lệch ngang off.
    def fn(t):
        return np.array([11.0 - v * t, off, 1.5]), float(r)

    return fn


def _deform_fn(r0, amp, period):
    # Biến dạng: vật đứng yên, bán kính r0 + amp·sin(2πt/period).
    def fn(t):
        return (
            np.array([5.0, 0.0, 1.5]),
            float(r0 + amp * np.sin(2.0 * np.pi * t / period)),
        )

    return fn


# Mỗi họ: khoảng rút đều của từng tham số (ranges), vectơ tham số của kịch
# bản đánh giá tương ứng (eval_vector, dùng cho hộp loại trừ) và hàm dựng
# truth_fn từ bộ tham số (make).
CAL_FAMILIES = {
    "crossing": dict(
        ranges=dict(v=(0.6, 1.4), y0=(-4.5, -2.5), r=(0.8, 1.2)),
        eval_vector=dict(v=0.9, y0=-3.5, r=1.0),  # D1
        make=lambda p: _crossing_fn(p["v"], p["y0"], p["r"]),
    ),
    "headon": dict(
        ranges=dict(v=(0.8, 1.4), off=(0.5, 1.3), r=(0.8, 1.2)),
        eval_vector=dict(v=1.1, off=0.9, r=0.9),  # D2
        make=lambda p: _headon_fn(p["v"], p["off"], p["r"]),
    ),
    "deform": dict(
        ranges=dict(r0=(0.8, 1.2), amp=(0.20, 0.45), period=(5.0, 8.0)),
        eval_vector=dict(r0=1.0, amp=0.35, period=6.0),  # D3
        make=lambda p: _deform_fn(p["r0"], p["amp"], p["period"]),
    ),
}
EXCLUSION_RADIUS = 0.05  # nửa cạnh hộp loại trừ, theo toạ độ chuẩn hoá


def _normalized(params, ranges):
    # Đưa từng tham số về [0, 1] theo khoảng rút của nó.
    return {k: (params[k] - lo) / (hi - lo) for k, (lo, hi) in ranges.items()}


def in_exclusion_box(params, family_spec) -> bool:
    """True nếu mọi tham số (đã chuẩn hoá) đều cách tham số tương ứng của
    vectơ đánh giá không quá EXCLUSION_RADIUS, tức vectơ nằm trong hộp
    loại trừ."""
    pn = _normalized(params, family_spec["ranges"])
    en = _normalized(family_spec["eval_vector"], family_spec["ranges"])
    return all(abs(pn[k] - en[k]) <= EXCLUSION_RADIUS for k in pn)


def draw_calibration_instances(
    n_per_family: int, seed: int = CAL_STREAM_BASE, families=CAL_FAMILIES
):
    """Rút n_per_family thực thể hiệu chuẩn cho mỗi họ; lần rút nào rơi vào
    hộp loại trừ thì rút lại.

    Trả (instances, audit):
      instances: danh sách dict(instance_id, family, params, truth_fns);
      audit[họ]: dict(accepted, rejected), số lần rút được nhận / bị loại."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed)]))
    instances, audit = [], {}
    next_id = 0
    for fam_name, spec in families.items():
        accepted, rejected = [], 0
        while len(accepted) < n_per_family:
            params = {
                k: float(rng.uniform(lo, hi)) for k, (lo, hi) in spec["ranges"].items()
            }
            if in_exclusion_box(params, spec):
                rejected += 1
                continue
            accepted.append(params)
        for params in accepted:
            instances.append(
                dict(
                    instance_id=next_id,
                    family=fam_name,
                    params=params,
                    truth_fns=[spec["make"](params)],
                )
            )
            next_id += 1
        audit[fam_name] = dict(accepted=len(accepted), rejected=rejected)
    # Kiểm tra lại: không thực thể nào nằm trong hộp loại trừ hay trùng
    # vectơ đánh giá.
    for inst in instances:
        spec = families[inst["family"]]
        assert not in_exclusion_box(inst["params"], spec)
        assert inst["params"] != spec["eval_vector"]
    return instances, audit


def collect_instance_scores(
    instances, t_nows, n_hists, K=28, dt=0.25, stream_base: int = CAL_STREAM_BASE
):
    """Chạy bộ dự báo Kalman trên các thực thể hiệu chuẩn và ghi điểm số.
    Mỗi ô (thực thể × t_now × n_hist) dùng một luồng RNG riêng.
    Trả (ScoreRecorder, counts), counts[nhóm] = số lần chạy của nhóm tuổi."""
    rec = ScoreRecorder(K=K)
    counts = {}
    for inst in instances:
        truth_fns = inst["truth_fns"]
        for it, t_now in enumerate(t_nows):
            truth = _ds.sample_truth(truth_fns, t_now, K, dt)
            for ih, n_hist in enumerate(n_hists):
                grp = age_group_of(n_hist)
                rng = _cell_rng(stream_base, inst["instance_id"], it, ih)
                # Dự báo, ghi điểm số từng vật cản, rồi ghi S của cả lần
                # chạy (max qua các vật cản).
                pred = _op.kalman_obstacles(truth_fns, t_now, K, dt, rng, n_hist=n_hist)
                s_run = 0.0
                for m in range(len(truth_fns)):
                    s_m = rec.record_forecast(
                        truth[m][0],
                        truth[m][1],
                        pred[m]["centers"],
                        pred[m]["radii"],
                        grp,
                        record_traj=False,
                    )
                    s_run = max(s_run, s_m)
                rec.record_traj_score(grp, s_run)
                counts[grp] = counts.get(grp, 0) + 1
    return rec, counts


def collect_scores(
    scenario_names, t_nows, n_hists, seeds, K=28, dt=0.25
) -> ScoreRecorder:
    """Thu điểm số trên các kịch bản có tên (ví dụ D1–D4). Mỗi ô
    (kịch bản × t_now × n_hist × seed) dùng luồng RNG suy ra từ
    (seed, chỉ số ô). Các kịch bản này là tập đánh giá, nên bảng hiệu
    chuẩn được xây từ collect_instance_scores, không từ hàm này."""
    rec = ScoreRecorder(K=K)
    for isc, name in enumerate(scenario_names):
        truth_fns = _ds.dynamic_scenario(name)["truth_fns"]
        for it, t_now in enumerate(t_nows):
            truth = _ds.sample_truth(truth_fns, t_now, K, dt)
            for ih, n_hist in enumerate(n_hists):
                grp = age_group_of(n_hist)
                for seed in seeds:
                    rng = _cell_rng(seed, isc, it, ih)
                    pred = _op.kalman_obstacles(
                        truth_fns, t_now, K, dt, rng, n_hist=n_hist
                    )
                    s_run = 0.0
                    for m in range(len(truth_fns)):
                        s_m = rec.record_forecast(
                            truth[m][0],
                            truth[m][1],
                            pred[m]["centers"],
                            pred[m]["radii"],
                            grp,
                            record_traj=False,
                        )
                        s_run = max(s_run, s_m)
                    rec.record_traj_score(grp, s_run)
    return rec


# ------------------------------------------------------------------
# Bảng conformal: xây dựng, lưu, nạp, tra cứu O(1)
# ------------------------------------------------------------------
class ConformalTable:
    """Bảng ρ đã hiệu chuẩn, tra cứu O(1).

      step[(kênh, nhóm)] : (n_eps, K+1), quantile theo từng bước k;
      traj[nhóm]         : (n_eps,), quantile của điểm số mức quỹ đạo;
                           khoá "full::<nhóm>" cho kênh comb_full;
      bonf[nhóm]         : (n_eps, K+1), quantile kênh comb từng bước ở
                           mức ε/(K+1);
      counts             : cỡ mẫu dùng cho từng mục;
      meta               : thông tin kèm theo (do phía gọi cung cấp).
    Hàng thứ i của các mảng ứng với eps_levels[i]."""

    def __init__(self, K, eps_levels=EPS_LEVELS):
        """Tạo bảng rỗng cho chân trời K và các mức ε; điền bằng build()
        hoặc load()."""
        self.K = int(K)
        self.eps_levels = tuple(float(e) for e in eps_levels)
        self.step: dict = {}
        self.traj: dict = {}
        self.bonf: dict = {}
        self.counts: dict = {}
        self.meta: dict = {}

    @classmethod
    def build(
        cls, rec: ScoreRecorder, eps_levels=EPS_LEVELS, meta: dict | None = None
    ) -> ConformalTable:
        """Tính mọi quantile từ các điểm số trong rec."""
        tab = cls(K=rec.K, eps_levels=eps_levels)
        # Quantile từng bước cho mọi (kênh, nhóm).
        groups = sorted({g for (_, g) in rec.step_scores})
        channels = sorted({c for (c, _) in rec.step_scores})
        for ch in channels:
            for grp in groups:
                mat = rec.stack(ch, grp)
                if mat.shape[0] == 0:
                    continue
                arr = np.empty((len(tab.eps_levels), tab.K + 1))
                for ie, eps in enumerate(tab.eps_levels):
                    for k in range(tab.K + 1):
                        arr[ie, k], _, n = conformal_quantile(mat[:, k], eps)
                    tab.counts[f"step|{ch}|{grp}"] = int(mat.shape[0])
                tab.step[(ch, grp)] = arr
        for key, scores in rec.traj_scores.items():
            # Quantile mức quỹ đạo. Khoá "full|<nhóm>" (từ
            # record_shape_forecast) được lưu thành traj["full::<nhóm>"].
            if key.startswith("full|"):
                grp = key.split("|", 1)[1]
                arr = np.empty(len(tab.eps_levels))
                for ie, eps in enumerate(tab.eps_levels):
                    arr[ie], _, n = conformal_quantile(scores, eps)
                tab.traj[f"full::{grp}"] = arr
                tab.counts[f"trajfull|{grp}"] = int(len(scores))
                continue
            arr = np.empty(len(tab.eps_levels))
            for ie, eps in enumerate(tab.eps_levels):
                arr[ie], _, n = conformal_quantile(scores, eps)
            tab.traj[key] = arr
            tab.counts[f"traj|{key}"] = int(len(scores))
        # Bonferroni: quantile kênh comb tại từng k ở mức ε/(K+1).
        for grp in groups:
            mat = rec.stack("comb", grp)
            if mat.shape[0] == 0:
                continue
            arr = np.empty((len(tab.eps_levels), tab.K + 1))
            for ie, eps in enumerate(tab.eps_levels):
                eps_step = eps / (tab.K + 1)
                for k in range(tab.K + 1):
                    arr[ie, k], _, _ = conformal_quantile(mat[:, k], eps_step)
            tab.bonf[grp] = arr
        tab.meta = dict(meta or {})
        return tab

    # ---- lưu / nạp (npz + json) ----
    # Khoá mảng: "step::<kênh>::<nhóm>", "traj::<nhóm>", "bonf::<nhóm>".
    # Dùng "::" làm dấu tách vì tên kênh có thể chứa "_" và tên nhóm
    # "__all__" chứa "__".
    def save(self, path_prefix: str) -> None:
        """Ghi <path_prefix>.npz (các mảng) và <path_prefix>.json (K, các
        mức ε, counts, meta)."""
        arrays = {}
        for (ch, grp), arr in self.step.items():
            arrays[f"step::{ch}::{grp}"] = arr
        for grp, arr in self.traj.items():
            arrays[f"traj::{grp}"] = arr
        for grp, arr in self.bonf.items():
            arrays[f"bonf::{grp}"] = arr
        np.savez_compressed(path_prefix + ".npz", **arrays)
        with open(path_prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(
                dict(
                    K=self.K,
                    eps_levels=list(self.eps_levels),
                    counts=self.counts,
                    meta=self.meta,
                ),
                f,
                ensure_ascii=False,
                indent=1,
            )

    @classmethod
    def load(cls, path_prefix: str) -> ConformalTable:
        """Nạp bảng đã lưu bằng save()."""
        with open(path_prefix + ".json", encoding="utf-8") as f:
            head = json.load(f)
        tab = cls(K=head["K"], eps_levels=head["eps_levels"])
        tab.counts = head.get("counts", {})
        tab.meta = head.get("meta", {})
        with np.load(path_prefix + ".npz") as z:
            for key in z.files:
                parts = key.split("::")
                kind = parts[0]
                if kind == "step" and len(parts) == 3:
                    tab.step[(parts[1], parts[2])] = z[key]
                elif kind == "traj" and len(parts) >= 2:
                    # "traj::full::<nhóm>" giữ nguyên phần sau "traj::".
                    tab.traj["::".join(parts[1:])] = z[key]
                elif kind == "bonf" and len(parts) == 2:
                    tab.bonf[parts[1]] = z[key]
                else:
                    raise KeyError(f"unknown table key: {key!r}")
        return tab

    # ---- tra cứu O(1) ----
    # Nếu nhóm tuổi yêu cầu không có trong bảng, các hàm tra cứu dùng nhóm
    # gộp POOLED. ε phải là một trong eps_levels.
    def _eps_idx(self, eps: float) -> int:
        if eps not in self.eps_levels:
            raise ValueError(f"eps={eps} not in table (available: {self.eps_levels})")
        return self.eps_levels.index(eps)

    def rho(self, channel: str, k: int, age_group: str, eps: float) -> float:
        """ρ_k(ε) của một kênh tại bước k."""
        if (channel, age_group) in self.step:
            key = (channel, age_group)
        elif (channel, POOLED) in self.step:
            key = (channel, POOLED)
        else:
            raise KeyError(f"channel {channel!r} not in table")
        return float(self.step[key][self._eps_idx(eps), int(k)])

    def rho_array(self, channel: str, age_group: str, eps: float) -> np.ndarray:
        """Mảng (K+1,) ρ_k(ε) của một kênh."""
        if (channel, age_group) in self.step:
            key = (channel, age_group)
        elif (channel, POOLED) in self.step:
            key = (channel, POOLED)
        else:
            raise KeyError(f"channel {channel!r} not in table")
        return self.step[key][self._eps_idx(eps)].copy()

    def rho_traj(self, age_group: str, eps: float) -> float:
        """ρ_traj(ε): quantile mức quỹ đạo, giá trị planner dùng."""
        grp = age_group if age_group in self.traj else POOLED
        return float(self.traj[grp][self._eps_idx(eps)])

    def rho_bonf_array(self, age_group: str, eps: float) -> np.ndarray:
        """Mảng (K+1,) quantile Bonferroni của kênh comb."""
        grp = age_group if age_group in self.bonf else POOLED
        return self.bonf[grp][self._eps_idx(eps)].copy()


# ------------------------------------------------------------------
# Đo bao phủ trên các kịch bản đánh giá (D1–D4), khác phân phối với
# tập hiệu chuẩn
# ------------------------------------------------------------------
def validate_coverage(
    table: ConformalTable,
    scenario_names,
    t_nows,
    n_hists,
    eval_seed_indices=range(20),
    K=28,
    dt=0.25,
    bootstrap_B: int = 2000,
) -> dict:
    """Đo tỉ lệ bao phủ của bảng conformal trên các kịch bản đánh giá cố
    định. Mỗi ô (kịch bản × t_now × n_hist × seed_eval(i)) có luồng RNG
    riêng suy ra từ (seed_eval(i), chỉ số ô).

    "Được phủ" nghĩa là điểm số ≤ ρ tra từ bảng. Với mỗi ε, kết quả gồm:
      - step[kênh] (pos, rad, comb): bao phủ gộp mọi (hàng, k), CI
        Clopper–Pearson (coi các điểm như độc lập), CI bootstrap theo cụm
        hàng, bao phủ nhỏ nhất theo k, bao phủ theo k và theo (k × nhóm
        tuổi), số điểm và số hàng;
      - traj_run: tỉ lệ lần chạy có max_{m,k} comb ≤ ρ_traj, kèm CI
        Clopper–Pearson, CI bootstrap theo cụm seed và tỉ lệ theo kịch bản;
      - traj_obstacle: như trên nhưng tính theo từng vật cản;
      - bonferroni: tỉ lệ vật cản có comb_k ≤ ρ_bonf,k với mọi k;
      - legacy_kappa2 / legacy_kappa2_plus_delta: để so sánh, đo bao phủ của
        ρ_κ = κ·σ (κ = 2) do bộ dự báo Kalman trả về, theo điều kiện
        e_pos + e_rad ≤ ρ_κ (+ 0,03 m ở biến thể plus_delta), cả theo từng
        điểm và theo lần chạy. ρ_κ ở đây không bị chặn trên.
    """
    # Bước 1: chạy dự báo cho mọi ô, lưu sai số của từng vật cản.
    runs = []  # mỗi phần tử: dict(grp, scenario, seed_idx, entries[m])
    for isc, name in enumerate(scenario_names):
        truth_fns = _ds.dynamic_scenario(name)["truth_fns"]
        for it, t_now in enumerate(t_nows):
            truth = _ds.sample_truth(truth_fns, t_now, K, dt)
            for ih, n_hist in enumerate(n_hists):
                grp = age_group_of(n_hist)
                for i in eval_seed_indices:
                    rng = _cell_rng(eval_seed(i), isc, it, ih)
                    pred = _op.kalman_obstacles(
                        truth_fns, t_now, K, dt, rng, n_hist=n_hist
                    )
                    entries = []
                    for m in range(len(truth_fns)):
                        e_pos = np.linalg.norm(pred[m]["centers"] - truth[m][0], axis=1)
                        e_rad = np.abs(pred[m]["radii"] - truth[m][1])
                        # Với vật cầu, điểm số ghép là e_pos + e_rad.
                        e_sum = e_pos + e_rad
                        entries.append(
                            dict(
                                e_pos=e_pos,
                                e_rad=e_rad,
                                e_comb=e_sum,
                                e_sum=e_sum,
                                rho_kappa=np.asarray(pred[m]["rho"]),
                            )
                        )
                    runs.append(
                        dict(grp=grp, scenario=name, seed_idx=int(i), entries=entries)
                    )
    # Bước 2: với mỗi ε, so sai số với ρ trong bảng và tính các tỉ lệ.
    n_runs = len(runs)
    all_entries = [e for r in runs for e in r["entries"]]
    n_obs = len(all_entries)
    out = dict(
        n_runs=n_runs,
        n_obstacle_runs=n_obs,
        K=K,
        n_seed_clusters=len({r["seed_idx"] for r in runs}),
        note_levels="step: cluster unit = one obstacle's forecast "
        "row; traj_run: unit = one run; coverage CI = "
        "Clopper-Pearson; primary per-step CI = "
        "row-cluster bootstrap.",
    )
    for eps in table.eps_levels:
        res = {}
        # Bao phủ từng bước của các kênh pos, rad, comb.
        for ch, key in (("pos", "e_pos"), ("rad", "e_rad"), ("comb", "e_comb")):
            ok = np.zeros(K + 1, dtype=int)  # số điểm được phủ theo k
            tot = np.zeros(K + 1, dtype=int)  # tổng số điểm theo k
            row_ok_counts = []
            per_k_age = {}
            for r in runs:
                bound = table.rho_array(ch, r["grp"], eps)
                for ent in r["entries"]:
                    okk = ent[key] <= bound
                    ok += okk.astype(int)
                    tot += 1
                    row_ok_counts.append(int(okk.sum()))
                    d = per_k_age.setdefault(r["grp"], np.zeros(K + 1, dtype=int))
                    d += okk.astype(int)
            n_ok, n_tot = int(ok.sum()), int(tot.sum())
            cp = clopper_pearson_ci(n_ok, n_tot)
            bc = bootstrap_pooled_coverage_ci(row_ok_counts, K + 1, B=bootstrap_B)
            age_n = {}
            for r in runs:
                age_n[r["grp"]] = age_n.get(r["grp"], 0) + len(r["entries"])
            per_k_age_cov = {g: (per_k_age[g] / age_n[g]).tolist() for g in per_k_age}
            res[ch] = dict(
                coverage=n_ok / n_tot,
                clopper_pearson95=list(cp),
                cluster_bootstrap95=list(bc),
                min_over_k=float((ok / tot).min()),
                coverage_by_k=(ok / tot).tolist(),
                coverage_by_k_age=per_k_age_cov,
                n_points=n_tot,
                n_rows=n_obs,
                ci_note="CP is nominal (points within a row are "
                "correlated); the primary CI is the row-cluster "
                "bootstrap.",
            )
        # Mức quỹ đạo theo lần chạy: max qua mọi vật cản và mọi k.
        ok_run_flags = []
        for r in runs:
            mx = max(float(e["e_comb"].max()) for e in r["entries"])
            ok_run_flags.append(mx <= table.rho_traj(r["grp"], eps))
        n_ok = int(sum(ok_run_flags))
        cp = clopper_pearson_ci(n_ok, n_runs)
        seed_clusters = {}
        for r, okf in zip(runs, ok_run_flags):
            a, b = seed_clusters.get(r["seed_idx"], (0, 0))
            seed_clusters[r["seed_idx"]] = (a + int(okf), b + 1)
        bc = bootstrap_cluster_ratio_ci(list(seed_clusters.values()), B=bootstrap_B)
        by_scen = {}
        for r, okf in zip(runs, ok_run_flags):
            a, b = by_scen.get(r["scenario"], (0, 0))
            by_scen[r["scenario"]] = (a + int(okf), b + 1)
        res["traj_run"] = dict(
            coverage=n_ok / n_runs,
            clopper_pearson95=list(cp),
            seed_cluster_bootstrap95=list(bc),
            n_runs=n_runs,
            by_scenario={
                s: dict(coverage=a / b, n_runs=b) for s, (a, b) in by_scen.items()
            },
        )
        # Mức quỹ đạo theo từng vật cản.
        ok_traj = 0
        for r in runs:
            bound = table.rho_traj(r["grp"], eps)
            ok_traj += sum(1 for e in r["entries"] if float(e["e_comb"].max()) <= bound)
        cp = clopper_pearson_ci(ok_traj, n_obs)
        res["traj_obstacle"] = dict(
            coverage=ok_traj / n_obs, clopper_pearson95=list(cp)
        )
        # Bonferroni: một vật cản được phủ nếu được phủ tại mọi k.
        ok_b = 0
        for r in runs:
            bound = table.rho_bonf_array(r["grp"], eps)
            ok_b += sum(1 for e in r["entries"] if bool(np.all(e["e_comb"] <= bound)))
        cp = clopper_pearson_ci(ok_b, n_obs)
        res["bonferroni"] = dict(coverage=ok_b / n_obs, clopper_pearson95=list(cp))
        # So sánh với ρ_κ (κ = 2) của bộ dự báo, có và không có δ = 0,03 m.
        for tag, extra in (("legacy_kappa2", 0.0), ("legacy_kappa2_plus_delta", 0.03)):
            row_ok_counts = []
            n_ok = 0
            for e in all_entries:
                okk = e["e_sum"] <= (e["rho_kappa"] + extra)
                n_ok += int(okk.sum())
                row_ok_counts.append(int(okk.sum()))
            n_tot = n_obs * (K + 1)
            cp = clopper_pearson_ci(n_ok, n_tot)
            bc = bootstrap_pooled_coverage_ci(row_ok_counts, K + 1, B=bootstrap_B)
            ok_runs = 0
            for r in runs:
                if all(
                    bool(np.all(e["e_sum"] <= e["rho_kappa"] + extra))
                    for e in r["entries"]
                ):
                    ok_runs += 1
            cp_run = clopper_pearson_ci(ok_runs, n_runs)
            res[tag] = dict(
                point_coverage=n_ok / n_tot,
                point_clopper_pearson95=list(cp),
                point_cluster_bootstrap95=list(bc),
                run_coverage=ok_runs / n_runs,
                run_clopper_pearson95=list(cp_run),
                n_points=n_tot,
                n_runs=n_runs,
                note="rho_kappa is not capped (favourable upper bound "
                "for legacy); extra=0.03 in the plus_delta variant "
                "is the constant delta of paper 1.",
            )
        out[f"eps={eps}"] = res
    return out
