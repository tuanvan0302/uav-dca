# -*- coding: utf-8 -*-
"""
Sổ đăng ký kịch bản THỐNG NHẤT cho gói benchmark ST-DCA (giao thức
1.2, §2). Mọi họ kịch bản được chọn theo TÊN qua một cửa duy nhất:

  - Họ S  (tĩnh)              : scenarios.py của Bài 1 + S1L mới (§2.1)
  - Họ D  (động / biến dạng)  : dynamic_scenarios.py của Bài 1 (§2.2)
  - Loạt A (kiểu FlightBench) : benchmark_scenarios.random_field (§2.3)
  - Loạt B (kiểu FLYINGTRUST) : nền tảng × {S2, S4, D1} (§2.4)
  - Họ F  (ellipsoid biến dạng): scenarios_pkg/family_f.py (§2.5)

`get_scenario(name)` trả dict chuẩn:
    name, family ('S'|'D'|'A'|'B'|'F'), kind, start, goal, t_max,
    truth_fns   : [t -> (tâm (3,), bán kính)] — giao diện cầu mà
                  sim_realtime / obstacle_predict tiêu thụ. Với họ F
                  đây là hình chiếu cầu thể-tích-tương-đương theo
                  §2.7/Q6 (giao diện của A0/A1), ghi rõ ở khóa
                  `sphere_representation`.
    obstacles   : [(tâm, r)] nếu tĩnh (kind='static_spheres'), else None
    ellipsoid_truth_fns : [t -> (tâm, bán trục (3,), yaw)] — chỉ họ F
    truth_fns_bounding  : hình chiếu cầu bao a_max (A0b) — chỉ họ F
    platform    : (tên, v_max, u_max) — chỉ loạt B
    meta        : thông tin nguồn gốc / tham số sinh.

`resolve_for_runner(name)` trả ĐÚNG dict mà sim_realtime.run_episode
tiêu thụ ({name, start, goal, truth_fns}): tra dynamic_scenarios
trước (họ D giữ nguyên hành vi cũ), sau đó tới sổ này. Đây là hàm mà
đường chạy cần trỏ vào để chọn được mọi họ theo tên (xem README.md
của gói, mục "Tích hợp runner").
"""
from __future__ import annotations

import numpy as np

from sim import benchmark_scenarios
from sim import dynamic_scenarios
from sim import scenarios as static_scenarios

from . import family_f

SAFETY_D = 0.5  # biên an toàn d (m) — §0 giao thức


# ---------------------------------------------------------------- S
def _static_entry(name: str, sc: dict, family: str = "S",
                  meta: dict | None = None) -> dict:
    obs = [(np.asarray(c, dtype=float), float(r)) for c, r in sc["obstacles"]]
    fns = []
    for c, r in obs:
        def make(c=c, r=r):
            def g(t):
                return c.copy(), r
            return g
        fns.append(make())
    return dict(name=name, family=family, kind="static_spheres",
                start=np.asarray(sc["start"], dtype=float),
                goal=np.asarray(sc["goal"], dtype=float),
                t_max=60.0, truth_fns=fns, obstacles=obs,
                ellipsoid_truth_fns=None, truth_fns_bounding=None,
                platform=None, meta=meta or {})


def _s1l_entry() -> dict:
    """S1L (§2.1, mới — chỉ cho A4): như S1 nhưng điểm đầu (0, 0, 0,9),
    thấp hơn tâm vật cản — ca kẹt phía vòng sai kinh điển của Bài 1."""
    base = static_scenarios.scenario("S1_mot_vat")
    sc = dict(start=np.array([0.0, 0.0, 0.9]), goal=base["goal"],
              obstacles=base["obstacles"])
    return _static_entry("S1L_thap", sc,
                         meta=dict(derived_from="S1_mot_vat",
                                   note="điểm đầu z=0,9 < tâm vật z=1,5 (§2.1)"))


# ---------------------------------------------------------------- D
def _dynamic_entry(name: str) -> dict:
    sc = dynamic_scenarios.dynamic_scenario(name)
    return dict(name=name, family="D", kind="dynamic_spheres",
                start=sc["start"], goal=sc["goal"], t_max=40.0,
                truth_fns=sc["truth_fns"], obstacles=None,
                ellipsoid_truth_fns=None, truth_fns_bounding=None,
                platform=None, meta=dict(source="dynamic_scenarios.py"))


# ---------------------------------------------------------------- A
def series_a_name(level: int, rep: int) -> str:
    return f"A_L{level}_r{rep}"


def _series_a_entry(level: int, rep: int) -> dict:
    sc = benchmark_scenarios.random_field(level, rep)
    return _static_entry(
        series_a_name(level, rep), sc, family="A",
        meta=dict(level=level, rep=rep,
                  geometry_seed=1000 + 100 * level + rep,
                  n_obstacles=benchmark_scenarios.LEVEL_COUNTS[level],
                  source="benchmark_scenarios.random_field (§2.3)"))


# ---------------------------------------------------------------- B
_SERIES_B_BASES = ("S2_hanh_lang", "S4_ngau_nhien", "D1_cat_ngang")


def series_b_name(platform_idx: int, base: str) -> str:
    pname = benchmark_scenarios.PLATFORMS[platform_idx][0]
    short = base.split("_")[0]  # S2 / S4 / D1
    return f"B_{pname}_{short}"


def _series_b_entry(platform_idx: int, base: str) -> dict:
    pname, v_max, u_max = benchmark_scenarios.PLATFORMS[platform_idx]
    if base.startswith("D"):
        e = _dynamic_entry(base)
    else:
        e = _static_entry(base, static_scenarios.scenario(base))
    e = dict(e)
    e.update(name=series_b_name(platform_idx, base), family="B",
             t_max=60.0, platform=(pname, v_max, u_max),
             meta=dict(base_scenario=base, mode="kalman",
                       source="benchmark_scenarios.PLATFORMS (§2.4)"))
    return e


# ---------------------------------------------------------------- F
def _family_f_entry(name: str) -> dict:
    return dict(name=name, family="F", kind="ellipsoid",
                start=family_f.START.copy(), goal=family_f.GOAL.copy(),
                t_max=family_f.T_MAX,
                truth_fns=family_f.sphere_eq_fns(name),
                obstacles=None,
                ellipsoid_truth_fns=family_f.ellipsoid_truth(name),
                truth_fns_bounding=family_f.sphere_bound_fns(name),
                platform=None,
                sphere_representation="volume_equivalent (§2.7/Q6)",
                meta=dict(source="scenarios_pkg/family_f.py (§2.5)"))


def _f1_ms_entry() -> dict:
    obs = family_f.f1_ms_spheres()
    sc = dict(start=family_f.START, goal=family_f.GOAL, obstacles=obs)
    e = _static_entry("F1-MS", sc, family="F",
                      meta=dict(derived_from="F1_thanh_dai",
                                note="ô minh họa multi-sphere §2.5: 9 cầu con "
                                     "r=0,28 m thay thanh F1; chỉ cho A3, "
                                     "10 seed phụ, không tham gia §5.3"))
    e["kind"] = "multisphere"
    e["ellipsoid_truth_fns"] = family_f.ellipsoid_truth("F1_thanh_dai")
    return e


# ------------------------------------------------------------- API
def all_names() -> list:
    names = list(static_scenarios.ALL) + ["S1L_thap"]
    names += dynamic_scenarios.all_dynamic_names()
    names += [series_a_name(lv, rep) for lv in range(1, 9) for rep in range(3)]
    names += [series_b_name(pi, b) for pi in range(3) for b in _SERIES_B_BASES]
    names += list(family_f.ALL_F) + ["F1-MS"]
    return names


def get_scenario(name: str) -> dict:
    """Dict kịch bản chuẩn theo tên (xem docstring đầu module)."""
    if name in static_scenarios.ALL:
        return _static_entry(name, static_scenarios.scenario(name))
    if name == "S1L_thap":
        return _s1l_entry()
    if name in dynamic_scenarios.all_dynamic_names():
        return _dynamic_entry(name)
    if name.startswith("A_L"):
        parts = name[3:].split("_r")
        return _series_a_entry(int(parts[0]), int(parts[1]))
    if name.startswith("B_"):
        for pi in range(3):
            for b in _SERIES_B_BASES:
                if name == series_b_name(pi, b):
                    return _series_b_entry(pi, b)
        raise KeyError(name)
    if name in family_f.ALL_F:
        return _family_f_entry(name)
    if name == "F1-MS":
        return _f1_ms_entry()
    raise KeyError(f"scenario_not_found:{name}")


def resolve_for_runner(name: str) -> dict:
    """Dict {name, start, goal, truth_fns} cho sim_realtime.run_episode.

    Họ D tra qua dynamic_scenarios như hành vi hiện tại; mọi họ còn
    lại qua sổ đăng ký này (truth_fns của họ F là hình chiếu cầu
    thể-tích-tương-đương §2.7 — giao diện A0/A1)."""
    try:
        return dynamic_scenarios.dynamic_scenario(name)
    except KeyError:
        pass
    e = get_scenario(name)
    return dict(name=e["name"], start=e["start"], goal=e["goal"],
                truth_fns=e["truth_fns"])
