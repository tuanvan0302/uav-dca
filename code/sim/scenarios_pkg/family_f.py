# -*- coding: utf-8 -*-
"""
Họ F — bộ kịch bản vật cản ELLIPSOID biến dạng (đóng góp kịch bản của
đề tài; chưa có benchmark công khai cho vật biến dạng — Ch.8 báo cáo).

Định nghĩa đóng băng tại §2.5 của giao thức thực nghiệm ST-DCA 1.2
(~/workspace/uav_dca/exp/protocol/GIAO_THUC_THUC_NGHIEM.md). Mọi kịch
bản F: điểm đầu (0, 0, 1,5), đích (10, 0, 1,5), t_max = 60 s, MỘT vật
ellipsoid với sự thật nền (o(t), a(t), ψ(t)):

  - o(t)   : tâm (3,)
  - a(t)   : bộ ba bán trục (a_x, a_y, a_z) trong KHUNG VẬT (trục x, y
             ngang; trục z dọc), khớp quy ước của ellipsoid.py —
             ellipsoid tại thời điểm t là Ellipsoid(o, a, ψ) với phép
             quay Rz(ψ) quanh trục z.
  - ψ(t)   : yaw quanh trục z (rad).

Hàm sự thật ở đây là HÀM TRUY VẤN ĐƯỢC tại mọi t (không phải bảng lấy
mẫu): `ellipsoid_truth(name)` trả danh sách các hàm t -> (o, a, ψ).

Các hình chiếu cầu (cho cấu hình chỉ hiểu cầu):
  - `sphere_eq_fns(name)`    : cầu thể-tích-tương-đương r_eq = (abc)^(1/3)
                               tại tâm thật — giao diện của A0/A1 theo
                               quyết định Q6 (§2.7 giao thức).
  - `sphere_bound_fns(name)` : cầu bao r = a_max(t) — giao diện của A0b
                               (bắt buộc trên F1, F3 theo §2.7).
Cả hai đều là hàm t -> (tâm (3,), bán kính) khớp giao diện truth_fns
của dynamic_scenarios.py / sim_realtime.py.

Ô minh họa multi-sphere F1-MS (§2.5): `f1_ms_spheres()` trả 9 cầu con
tĩnh biểu diễn thay thế thanh của F1.
"""
from __future__ import annotations

import numpy as np

START = np.array([0.0, 0.0, 1.5])
GOAL = np.array([10.0, 0.0, 1.5])
T_MAX = 60.0

_DEG = np.pi / 180.0


# ---------------------------------------------------------------- F1
def _f1(t: float):
    """Thanh dài tĩnh chắn ngang đường bay: tâm (5; 0; 1,5), bán trục
    (0,15; 2,0; 0,15) — dài 4 m dọc y, dày 0,3 m theo x và z; ψ = 0."""
    return (np.array([5.0, 0.0, 1.5]),
            np.array([0.15, 2.0, 0.15]),
            0.0)


# ---------------------------------------------------------------- F2
def _f2(t: float):
    """Phồng dị hướng bảo toàn thể tích: vật tĩnh tại (5, 0, 1,5);
    s(t) = (1 − cos(2πt/12))/2, chu kỳ 12 s; a_x = 1 + 0,8·s (1,0→1,8);
    a_y = a_z = 1/√a_x (≈0,745 tại đỉnh); ψ = 0. Thể tích abc = 1 ∀t."""
    s = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / 12.0))
    ax = 1.0 + 0.8 * s
    ay = 1.0 / np.sqrt(ax)
    return (np.array([5.0, 0.0, 1.5]),
            np.array([ax, ay, ay]),
            0.0)


# ---------------------------------------------------------------- F3
def _f3(t: float):
    """Vật xoay yaw: tĩnh tại (5, 0, 1,5); bán trục hằng (1,6; 0,5; 0,5);
    ψ(t) = 35°·sin(2πt/10), chu kỳ 10 s."""
    return (np.array([5.0, 0.0, 1.5]),
            np.array([1.6, 0.5, 0.5]),
            35.0 * _DEG * np.sin(2.0 * np.pi * t / 10.0))


# ---------------------------------------------------------------- F4
def _f4(t: float):
    """Phồng lệch pha hai trục: tĩnh tại (5, 0, 1,5); chu kỳ 8 s;
    a_x = 1,1 + 0,45·sin(2πt/8); a_y = 1,1 + 0,45·sin(2πt/8 − 2π/3)
    (lệch pha 120°); a_z = 0,9; ψ = 0."""
    ph = 2.0 * np.pi * t / 8.0
    ax = 1.1 + 0.45 * np.sin(ph)
    ay = 1.1 + 0.45 * np.sin(ph - 2.0 * np.pi / 3.0)
    return (np.array([5.0, 0.0, 1.5]),
            np.array([ax, ay, 0.9]),
            0.0)


# ---------------------------------------------------------------- F5
def _f5(t: float):
    """Biến dạng + di chuyển đồng thời: tâm như D1, o(t) = (5; −3,5 +
    0,9·t; 1,5); a_y(t) = 1,0 + 0,35·sin(2πt/6) (chu kỳ 6 s, dao động
    dọc hướng chuyển động), a_x = a_z = 0,9; ψ = 0."""
    return (np.array([5.0, -3.5 + 0.9 * t, 1.5]),
            np.array([0.9, 1.0 + 0.35 * np.sin(2.0 * np.pi * t / 6.0), 0.9]),
            0.0)


_ELLIPSOID_TRUTH = {
    "F1_thanh_dai": _f1,
    "F2_phong_dihuong": _f2,
    "F3_xoay_yaw": _f3,
    "F4_lech_pha": _f4,
    "F5_ket_hop": _f5,
}

ALL_F = list(_ELLIPSOID_TRUTH.keys())


def ellipsoid_truth(name: str) -> list:
    """Danh sách hàm sự thật ellipsoid t -> (tâm (3,), bán trục (3,),
    yaw rad) của kịch bản F `name` (hiện mọi kịch bản F có 1 vật)."""
    return [_ELLIPSOID_TRUTH[name]]


def sphere_eq_fns(name: str) -> list:
    """Hình chiếu cầu thể-tích-tương-đương (§2.7/Q6): hàm t -> (tâm,
    r_eq(t) = (a·b·c)^(1/3)). Giao diện A0/A1 trên họ F."""
    out = []
    for fn in ellipsoid_truth(name):
        def make(fn):
            def g(t):
                o, a, _ = fn(t)
                return o, float(np.cbrt(a[0] * a[1] * a[2]))
            return g
        out.append(make(fn))
    return out


def sphere_bound_fns(name: str) -> list:
    """Hình chiếu cầu bao (§2.7, cấu hình A0b): hàm t -> (tâm,
    r(t) = a_max(t))."""
    out = []
    for fn in ellipsoid_truth(name):
        def make(fn):
            def g(t):
                o, a, _ = fn(t)
                return o, float(np.max(a))
            return g
        out.append(make(fn))
    return out


# ------------------------------------------------------------- F1-MS
def f1_ms_spheres() -> list:
    """Ô minh họa multi-sphere F1-MS (§2.5): thanh của F1 được biểu
    diễn thay thế bằng hợp 9 cầu con tĩnh, tâm (5; y; 1,5) với
    y ∈ {−2; −1,5; …; 2}, bán kính 0,28 m mỗi cầu (bao phủ thanh dày
    0,3 m, dài 4 m với độ chồng lấp liên tục). Trả danh sách
    (tâm (3,), bán kính)."""
    return [(np.array([5.0, y, 1.5]), 0.28)
            for y in np.linspace(-2.0, 2.0, 9)]


def f1_ms_truth_fns() -> list:
    """9 cầu con của F1-MS dưới dạng hàm sự thật cầu t -> (tâm, r)."""
    out = []
    for c, r in f1_ms_spheres():
        def make(c=c, r=r):
            def g(t):
                return c.copy(), r
            return g
        out.append(make())
    return out
