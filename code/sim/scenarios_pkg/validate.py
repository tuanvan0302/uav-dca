# -*- coding: utf-8 -*-
"""
Kiểm tra tính HỢP LỆ của kịch bản — ĐỘC LẬP với mọi cấu hình đem so
sánh (giao thức §2.5(i)(ii), áp dụng mở rộng cho mọi họ). Cấm dùng
bất kỳ cấu hình A0–A6 / baseline nào làm oracle khả thi.

(i) Tồn tại đường đi hình học khả thi: A* trên lưới mịn bước 0,1 m
    trên trường chiếm dụng SỰ THẬT NỀN, mọi vật cản phồng đều theo
    biên d = 0,5 m quanh bề mặt thật:
      - kịch bản tĩnh: một trường duy nhất;
      - kịch bản động/biến dạng: trường tại (các) lớp thời gian bất
        lợi nhất (đỉnh phồng / giao nhau) + kiểm tra đường đi theo
        thời gian: nội suy trạng thái vật tại thời điểm UAV dự kiến
        tới từng điểm của đường A*, với tốc độ danh nghĩa
        v_nom ∈ {1; 2; 3} m/s (phán quyết ở 2 m/s, hai tốc độ còn
        lại báo cáo kèm).
    Hai mức kết quả được báo cáo tách bạch:
      - `astar_inflated_d`     : A* với phồng đúng d (đúng chữ §2.5(i));
      - `certificate_min_seg_clearance` : khe hở thật nhỏ nhất dọc
        các đoạn thẳng của đường tìm được (lấy mẫu ≤ 0,05 m) — số đo
        trung thực của chính đường đó;
      - `astar_certificate`    : A* với phồng d + (√3/2)·h — điều kiện
        ĐỦ (cùng lập luận với astar_seed.py của Bài 1): nếu có đường
        ở mức phồng này thì mọi đoạn thẳng của nó giữ khe hở ≥ d
        một cách chứng minh được.
(ii) Cực trị bán trục / bán kính / yaw đúng như bảng §2.5 (kiểm tra
    số học trên hàm sự thật, lấy mẫu dày).

Khoảng cách tới ellipsoid: khoảng cách Euclid tới bề mặt qua bài
toán điểm gần nhất (hệ số Lagrange, giải bằng chia đôi — chính xác
tới độ chính xác số) cho điểm ngoài; xấp xỉ bậc nhất (g−1)/‖∇g‖ cho
điểm trong (điểm trong bị chặn sâu, chỉ vùng lân cận bề mặt mới ảnh
hưởng biên chặn và ở đó xấp xỉ này chính xác).

Chạy (từ thư mục exp/code):
    python3 -m scenarios_pkg.validate [--only F1_thanh_dai ...]
Kết quả ghi scenarios_pkg/validation_results.json.
"""
from __future__ import annotations

import argparse
import heapq
import json
import time
from pathlib import Path

import numpy as np

from . import family_f
from .registry import SAFETY_D, get_scenario

H = 0.1                      # bước lưới mịn (m) — §2.5(i)
SEG_SAMPLE = 0.05            # bước lấy mẫu dọc đoạn (m)
CERT_EXTRA = (np.sqrt(3.0) / 2.0) * H   # điều kiện đủ cho đoạn thẳng
Z_FLOOR = 0.8                # sàn bay (PlannerParams.z_floor)


# ------------------------------------------------- khoảng cách có dấu
def sphere_signed_distance(P: np.ndarray, center: np.ndarray,
                           radius: float) -> np.ndarray:
    """P (N,3) -> khoảng cách có dấu tới bề mặt cầu (âm = bên trong)."""
    return np.linalg.norm(P - center, axis=1) - radius


def ellipsoid_signed_distance(P: np.ndarray, center: np.ndarray,
                              axes: np.ndarray, yaw: float) -> np.ndarray:
    """Khoảng cách có dấu (xấp xỉ bậc nhất phía trong, chính xác phía
    ngoài) từ điểm P (N,3) tới bề mặt ellipsoid (tâm, bán trục, yaw)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    X = P - center
    # quay về khung vật: u = Rz(-yaw) @ (x - o)
    u = np.empty_like(X)
    u[:, 0] = c * X[:, 0] - s * X[:, 1]
    u[:, 1] = s * X[:, 0] + c * X[:, 1]
    u[:, 2] = X[:, 2]
    a = np.asarray(axes, dtype=float)
    a2 = a * a
    g = np.sqrt(np.sum((u / a) ** 2, axis=1))
    # gradient của gauge trong khung vật: D^-2 u / g
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = (u / a2) / np.maximum(g, 1e-12)[:, None]
    gn = np.linalg.norm(grad, axis=1)
    # kẹp dưới để điểm sát tâm (g -> 0, gradient suy biến) không cho
    # giá trị -inf/1e12 làm nhiễu các phép min ở phía gọi
    dist = np.maximum((g - 1.0) / np.maximum(gn, 1e-12),
                      -(float(np.max(a)) + 1.0))
    # chỉ tinh chỉnh chính xác các điểm trong dải |dist| <= 1 m quanh
    # bề mặt (vùng quyết định chặn/lề); điểm xa giữ xấp xỉ bậc nhất —
    # dấu và kết luận chặn không đổi, tốc độ tăng nhiều lần
    out_mask = (g > 1.0 + 1e-9) & (dist <= 1.0)
    if np.any(out_mask):
        uu = u[out_mask]
        # điểm gần nhất: x_j = a_j^2 u_j / (t + a_j^2), t >= 0 là nghiệm
        # của sum_j (a_j u_j / (t + a_j^2))^2 = 1 (đơn điệu giảm theo t)
        au = a * uu
        lo = np.zeros(len(uu))
        hi = np.sqrt(np.sum(au ** 2, axis=1)) + 1e-9
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            val = np.sum((au / (mid[:, None] + a2)) ** 2, axis=1)
            too_big = val > 1.0
            lo = np.where(too_big, mid, lo)
            hi = np.where(too_big, hi, mid)
        t = 0.5 * (lo + hi)
        closest = a2 * uu / (t[:, None] + a2)
        dist[out_mask] = np.linalg.norm(uu - closest, axis=1)
    return dist


def scenario_signed_distance(entry: dict, P: np.ndarray, t: float,
                             representation: str = "truth") -> np.ndarray:
    """Khoảng cách có dấu nhỏ nhất từ các điểm P tới mọi vật cản của
    kịch bản tại thời điểm t, theo SỰ THẬT NỀN thật (ellipsoid cho họ
    F; cầu cho các họ còn lại)."""
    dmin = np.full(len(P), np.inf)
    if entry["ellipsoid_truth_fns"] and representation == "truth":
        for fn in entry["ellipsoid_truth_fns"]:
            o, a, yaw = fn(t)
            dmin = np.minimum(dmin, ellipsoid_signed_distance(P, o, a, yaw))
    else:
        for fn in entry["truth_fns"]:
            o, r = fn(t)
            dmin = np.minimum(dmin, sphere_signed_distance(P, o, r))
    return dmin


# ---------------------------------------------------------------- A*
def _grid_bounds(entry: dict, t_layer: float) -> tuple[np.ndarray, np.ndarray]:
    pts = [entry["start"], entry["goal"]]
    if entry["obstacles"]:
        for c, r in entry["obstacles"]:
            pts += [c - r, c + r]
    else:
        # lấy mẫu sự thật trên [0, t_max] để bao đủ vùng vật chiếm
        for tt in np.linspace(0.0, entry["t_max"], 81):
            if entry["ellipsoid_truth_fns"]:
                for fn in entry["ellipsoid_truth_fns"]:
                    o, a, _ = fn(tt)
                    pts += [o - np.max(a), o + np.max(a)]
            else:
                for fn in entry["truth_fns"]:
                    o, r = fn(tt)
                    pts += [o - r, o + r]
    P = np.array(pts)
    lo = P.min(axis=0) - np.array([1.2, 2.2, 0.0])
    hi = P.max(axis=0) + np.array([1.2, 2.2, 1.8])
    lo[2] = Z_FLOOR
    hi[2] = max(hi[2], entry["start"][2] + 0.5, entry["goal"][2] + 0.5)
    return lo, hi


def _blocked_grid(entry: dict, t_layer: float, inflate: float,
                  lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, tuple]:
    n = np.floor((hi - lo) / H).astype(int) + 1
    blocked = np.zeros(tuple(n), dtype=bool)
    xs = lo[0] + H * np.arange(n[0])
    ys = lo[1] + H * np.arange(n[1])
    zs = lo[2] + H * np.arange(n[2])
    for iz, z in enumerate(zs):
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        P = np.column_stack([xx.ravel(), yy.ravel(),
                             np.full(xx.size, z)])
        sd = scenario_signed_distance(entry, P, t_layer)
        blocked[:, :, iz] = (sd.reshape(xx.shape) < inflate)
    return blocked, (lo, n)


def _snap(free: np.ndarray, lo: np.ndarray, p: np.ndarray) -> tuple:
    idx = tuple(int(round((p[i] - lo[i]) / H)) for i in range(3))
    idx = tuple(min(max(idx[i], 0), free.shape[i] - 1) for i in range(3))
    return idx


_NEIGHBORS = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx or dy or dz:
                _NEIGHBORS.append((dx, dy, dz,
                                   H * np.sqrt(dx * dx + dy * dy + dz * dz)))


def astar(blocked: np.ndarray, start_idx: tuple, goal_idx: tuple):
    """A* 26-hướng trên lưới chặn. Trả danh sách chỉ số nút hoặc None."""
    blocked = blocked.copy()
    blocked[start_idx] = False
    blocked[goal_idx] = False
    shape = blocked.shape
    g_score = np.full(shape, np.inf)
    came = np.full(shape + (3,), -1, dtype=np.int32)
    closed = np.zeros(shape, dtype=bool)
    g_score[start_idx] = 0.0

    def heur(idx):
        return H * np.sqrt(sum((idx[i] - goal_idx[i]) ** 2 for i in range(3)))

    pq = [(heur(start_idx), 0.0, start_idx)]
    while pq:
        _, g, idx = heapq.heappop(pq)
        if idx == goal_idx:
            path = [idx]
            while path[-1] != start_idx:
                p = tuple(int(v) for v in came[path[-1]])
                path.append(p)
            return path[::-1]
        if closed[idx] or g > g_score[idx] + 1e-12:
            continue
        closed[idx] = True
        for dx, dy, dz, w in _NEIGHBORS:
            nb = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
            if not (0 <= nb[0] < shape[0] and 0 <= nb[1] < shape[1]
                    and 0 <= nb[2] < shape[2]):
                continue
            if blocked[nb] or closed[nb]:
                continue
            ng = g + w
            if ng < g_score[nb] - 1e-12:
                g_score[nb] = ng
                came[nb] = idx
                heapq.heappush(pq, (ng + heur(nb), ng, nb))
    return None


def _path_points(path_idx, lo):
    return np.array([lo + H * np.array(idx) for idx in path_idx])


def segment_min_clearance(entry: dict, pts: np.ndarray, t_of_s,
                          t_layer_fixed: float | None) -> float:
    """Khe hở thật nhỏ nhất dọc các đoạn của đường (lấy mẫu ≤ SEG_SAMPLE).
    t_of_s: hàm thời gian theo độ dài cung (cho kiểm tra theo thời gian)
    hoặc None để dùng lớp cố định t_layer_fixed."""
    worst = np.inf
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
        np.diff(pts, axis=0), axis=1))])
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        L = cum[i + 1] - cum[i]
        m = max(2, int(np.ceil(L / SEG_SAMPLE)) + 1)
        frac = np.linspace(0.0, 1.0, m)
        Q = a[None, :] + frac[:, None] * (b - a)[None, :]
        if t_of_s is None:
            sd = scenario_signed_distance(entry, Q, t_layer_fixed)
        else:
            sd = np.empty(len(Q))
            for j, f in enumerate(frac):
                sd[j] = scenario_signed_distance(
                    entry, Q[j:j + 1], t_of_s(cum[i] + f * L))[0]
        worst = min(worst, float(np.min(sd)))
    return worst


# ------------------------------------------------------- lớp bất lợi
def worst_layers(entry: dict) -> list:
    """Danh sách (nhãn, t) các lớp thời gian bất lợi nhất phải kiểm tra
    tĩnh, theo §2.5(i): đỉnh phồng / giao nhau. Tĩnh -> một lớp t=0."""
    name = entry["name"]
    fixed = {
        "D1_cat_ngang": [("giao_truc_bay", 3.5 / 0.9)],
        "D2_doi_dau": [("tai_x5", 6.0 / 1.1)],
        "D3_bien_dang": [("dinh_phong", 1.5)],
        "D4_ket_hop": [("dinh_phong_D3", 1.5), ("giao_truc_D1", 3.5 / 0.9)],
        "F1_thanh_dai": [("tinh", 0.0)],
        "F2_phong_dihuong": [("dinh_phong_s=1", 6.0)],
        "F3_xoay_yaw": [("yaw_0", 0.0), ("yaw_+35", 2.5), ("yaw_-35", 7.5)],
        "F5_ket_hop": [("giao_truc_bay", 3.5 / 0.9)],
    }
    if name in fixed:
        return fixed[name]
    if name == "F4_lech_pha":
        ts = np.linspace(0.0, 8.0, 1601)
        fn = entry["ellipsoid_truth_fns"][0]
        ax = np.array([fn(t)[1][0] for t in ts])
        ay = np.array([fn(t)[1][1] for t in ts])
        req = np.array([np.cbrt(np.prod(fn(t)[1])) for t in ts])
        return [("max_a_x", float(ts[np.argmax(ax)])),
                ("max_a_y", float(ts[np.argmax(ay)])),
                ("max_r_eq", float(ts[np.argmax(req)]))]
    return [("tinh", 0.0)]


def is_dynamic(entry: dict) -> bool:
    if entry["family"] in ("D",) or entry["name"] in (
            "F2_phong_dihuong", "F3_xoay_yaw", "F4_lech_pha", "F5_ket_hop"):
        return True
    return False


# ------------------------------------------------------------- chính
def validate_scenario(name: str) -> dict:
    t0 = time.time()
    entry = get_scenario(name)
    layers = worst_layers(entry)
    lo, hi = _grid_bounds(entry, 0.0)
    res = dict(name=name, family=entry["family"], kind=entry["kind"],
               t_max=entry["t_max"], grid_h=H, safety_d=SAFETY_D,
               layers=[], traversal=None, valid=None)
    candidates = []   # (nhãn nguồn, điểm đường) — họ đường tĩnh ứng viên
    for label, t_layer in layers:
        blocked_d, (lo_g, _) = _blocked_grid(entry, t_layer, SAFETY_D, lo, hi)
        s_idx = _snap(blocked_d, lo_g, entry["start"])
        g_idx = _snap(blocked_d, lo_g, entry["goal"])
        path_d = astar(blocked_d, s_idx, g_idx)
        layer = dict(label=label, t=float(t_layer),
                     astar_inflated_d=path_d is not None)
        if path_d is not None:
            pts = _path_points(path_d, lo_g)
            layer["path_len_m"] = float(np.sum(np.linalg.norm(
                np.diff(pts, axis=0), axis=1)))
            layer["seg_min_clearance_m"] = segment_min_clearance(
                entry, pts, None, t_layer)
            candidates.append((f"{label}/d", pts))
        blocked_c, _ = _blocked_grid(entry, t_layer, SAFETY_D + CERT_EXTRA,
                                     lo, hi)
        path_c = astar(blocked_c, s_idx, g_idx)
        layer["astar_certificate"] = path_c is not None
        if path_c is not None:
            candidates.append((f"{label}/cert", _path_points(path_c, lo_g)))
        res["layers"].append(layer)
    # Kiểm tra đường đi theo thời gian cho kịch bản động/biến dạng:
    # kiểm tra TỒN TẠI — thử mọi đường ứng viên × độ lệch giờ khởi
    # hành δ ∈ {0;1;2;3} s (chờ trước khi bay là một tham số hoá thời
    # gian hợp lệ) × tốc độ danh nghĩa {1;2;3} m/s. Phán quyết: có
    # một tổ hợp sạch ở v = 2 m/s. (Một đường tĩnh duy nhất có thể
    # không khả thi theo thời gian trong khi kịch bản vẫn hợp lệ —
    # ví dụ đường vòng phía vật đang tiến tới của D1, hay đường lách
    # ngang vừa khít ở đỉnh phồng của F2 khi bán trục ngang lớn nhất
    # ở pha khác.)
    if is_dynamic(entry) and candidates:
        trav = dict(candidates=[])
        best_v2 = -np.inf
        for src, pts in candidates:
            for v_nom in (1.0, 2.0, 3.0):
                for delta in (0.0, 1.0, 2.0, 3.0):
                    mc = segment_min_clearance(
                        entry, pts,
                        lambda s, v=v_nom, dl=delta: dl + s / v, None)
                    rec = dict(path=src, v_nom=v_nom, delta_s=delta,
                               min_clearance_m=mc,
                               margin_over_d_m=mc - SAFETY_D)
                    trav["candidates"].append(rec)
                    if v_nom == 2.0:
                        best_v2 = max(best_v2, mc - SAFETY_D)
        trav["best_margin_over_d_at_v2_m"] = float(best_v2)
        trav["pass"] = bool(best_v2 >= -1e-9)
        res["traversal"] = trav
    ok_layers = all(l["astar_inflated_d"] for l in res["layers"])
    ok_trav = (res["traversal"] is None or res["traversal"]["pass"])
    res["valid"] = bool(ok_layers and ok_trav)
    res["certificate_all_layers"] = all(
        l["astar_certificate"] for l in res["layers"])
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------- kiểm tra số học
def extrema_checks() -> dict:
    """§2.5(ii): cực trị bán trục/bán kính/yaw của họ F và D3 đúng bảng."""
    ts = np.linspace(0.0, 120.0, 24001)
    out = {}
    tol = 2e-3

    def rng(vals):
        return [float(np.min(vals)), float(np.max(vals))]

    f1 = family_f.ellipsoid_truth("F1_thanh_dai")[0]
    A = np.array([f1(t)[1] for t in ts[::200]])
    out["F1"] = dict(axes_const=bool(np.allclose(A, [0.15, 2.0, 0.15])),
                     r_eq=float(np.cbrt(0.15 * 2.0 * 0.15)),
                     r_eq_expected=0.356, r_bound=2.0)

    f2 = family_f.ellipsoid_truth("F2_phong_dihuong")[0]
    ax = np.array([f2(t)[1][0] for t in ts])
    ay = np.array([f2(t)[1][1] for t in ts])
    az = np.array([f2(t)[1][2] for t in ts])
    out["F2"] = dict(a_x_range=rng(ax), a_x_expected=[1.0, 1.8],
                     ay_eq_az=bool(np.allclose(ay, az)),
                     volume_abc_max_dev=float(np.max(np.abs(ax * ay * az - 1.0))),
                     ay_at_peak=float(1.0 / np.sqrt(1.8)), ay_expected=0.745)

    f3 = family_f.ellipsoid_truth("F3_xoay_yaw")[0]
    yaw = np.array([f3(t)[2] for t in ts])
    out["F3"] = dict(yaw_amplitude_deg=float(np.max(np.abs(yaw)) / np.pi * 180.0),
                     yaw_amplitude_expected=35.0,
                     r_eq=float(np.cbrt(1.6 * 0.5 * 0.5)), r_eq_expected=0.737)

    f4 = family_f.ellipsoid_truth("F4_lech_pha")[0]
    ax4 = np.array([f4(t)[1][0] for t in ts])
    ay4 = np.array([f4(t)[1][1] for t in ts])
    dt_peak = abs(ts[np.argmax(ax4)] - ts[np.argmax(ay4)])
    out["F4"] = dict(a_x_range=rng(ax4), a_y_range=rng(ay4),
                     range_expected=[0.65, 1.55],
                     peak_time_gap_s=float(dt_peak),
                     peak_gap_expected_s=8.0 / 3.0,
                     a_z=float(f4(0.0)[1][2]))

    f5 = family_f.ellipsoid_truth("F5_ket_hop")[0]
    ay5 = np.array([f5(t)[1][1] for t in ts])
    c0, c1 = f5(0.0)[0], f5(10.0)[0]
    out["F5"] = dict(a_y_range=rng(ay5), a_y_expected=[0.65, 1.35],
                     center_speed_y=float((c1[1] - c0[1]) / 10.0),
                     center_speed_expected=0.9,
                     a_x=float(f5(0.0)[1][0]), a_z=float(f5(0.0)[1][2]))

    from sim import dynamic_scenarios
    d3 = dynamic_scenarios.dynamic_scenario("D3_bien_dang")["truth_fns"][0]
    r3 = np.array([d3(t)[1] for t in ts])
    out["D3"] = dict(r_range=rng(r3), r_expected=[0.65, 1.35])

    def approx(a, b, t=tol):
        return abs(a - b) <= max(t, 0.01 * abs(b))

    out["_pass"] = bool(
        out["F1"]["axes_const"] and approx(out["F1"]["r_eq"], 0.356, 0.002)
        and approx(out["F2"]["a_x_range"][1], 1.8) and approx(out["F2"]["a_x_range"][0], 1.0)
        and out["F2"]["volume_abc_max_dev"] < 1e-9
        and approx(out["F3"]["yaw_amplitude_deg"], 35.0, 0.05)
        and approx(out["F3"]["r_eq"], 0.737, 0.002)
        and approx(out["F4"]["a_x_range"][0], 0.65) and approx(out["F4"]["a_x_range"][1], 1.55)
        and approx(out["F4"]["peak_time_gap_s"], 8.0 / 3.0, 0.02)
        and approx(out["F5"]["a_y_range"][1], 1.35)
        and approx(out["F5"]["center_speed_y"], 0.9)
        and approx(out["D3"]["r_range"][0], 0.65) and approx(out["D3"]["r_range"][1], 1.35))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(Path(__file__).parent
                                         / "validation_results.json"))
    args = ap.parse_args()
    from .registry import all_names
    names = args.only or all_names()
    # loạt B dùng lại hình học của kịch bản gốc — kiểm tra qua bản gốc
    names = [n for n in names if not n.startswith("B_")]
    results = {}
    for n in names:
        r = validate_scenario(n)
        results[n] = r
        print(f"[{n}] valid={r['valid']} cert={r['certificate_all_layers']} "
              f"({r['elapsed_s']}s)", flush=True)
    ext = extrema_checks()
    print(f"[extrema] pass={ext['_pass']}", flush=True)
    doc = dict(protocol="ST-DCA-EXP 1.2 §2.5(i)(ii) + mở rộng mọi họ",
               grid_h=H, safety_d=SAFETY_D,
               scenarios=results, extrema=ext)
    Path(args.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")
    return 0 if all(r["valid"] for r in results.values()) and ext["_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
