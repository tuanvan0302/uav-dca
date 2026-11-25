# -*- coding: utf-8 -*-
"""
Hình học ellipsoid phụ thuộc thời gian — tầng hình dạng (G) của ST-DCA.

Một vật cản ellipsoid tại một thời điểm được cho bởi:
  - tâm o ∈ R^3,
  - ba bán trục a = (a1, a2, a3), a_j > 0 (a1, a2 là hai trục ngang trong
    khung vật, a3 là trục dọc),
  - góc yaw ψ quanh trục z. Ellipsoid chỉ được quay quanh z, nên ma trận
    quay là R = Rz(ψ).

Hàm gauge:
    g(x) = ‖ D⁻¹ Rᵀ (x − o) ‖,   D = diag(a).
g là chuẩn Euclid của một ánh xạ afin nên lồi; g(x) = 1 trên bề mặt,
< 1 bên trong, > 1 bên ngoài. Gradient dạng kín:
    ∇g(x) = R D⁻² Rᵀ (x − o) / g(x)        (x ≠ o).

Ràng buộc an toàn tổng quát (R4*):  g(p_k) ≥ 1 + b_{m,k}, với b ở đơn vị
gauge. Một biên mét δ được đổi sang đơn vị gauge bằng δ / a_min,
a_min = min(a). Lý do: với mọi x có g(x) ≤ 1,
    g(p) − g(x) ≤ ‖D⁻¹ Rᵀ (p − x)‖ ≤ ‖p − x‖ / a_min,
nên ‖p − x‖ ≥ a_min · (g(p) − 1). Vậy g(p) ≥ 1 + δ/a_min kéo theo khoảng
cách Euclid từ p tới ellipsoid ≥ δ.

Mặt cắt dùng trong bộ giải (core/dca_planner.py) được nhân với a_min để
biến dư và dung sai có đơn vị mét:
    a_min · [ g(q) + ∇g(q)ᵀ (p − q) ]  ≥  need − s,
    need = a_min + δ_mét.
Khi a1 = a2 = a3 = r thì a_min·g(x) = ‖x − o‖ và a_min·∇g(q) = (q−o)/‖q−o‖,
nên mặt cắt trùng với mặt cắt của vật cản cầu
(unit · (p − o) ≥ need − s). tests/test_gauge.py kiểm tra điều này với
sai số < 1e-9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rotation_z(yaw: float) -> np.ndarray:
    """Ma trận quay quanh trục z một góc yaw (rad)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Ellipsoid:
    """Một ellipsoid tại một thời điểm: tâm, bán trục, yaw."""

    center: np.ndarray  # (3,) tâm o
    axes: np.ndarray  # (3,) bán trục (a1, a2, a3), mọi phần tử > 0
    yaw: float = 0.0  # rad, góc quay quanh z

    def __post_init__(self):
        """Chuẩn hoá kiểu dữ liệu và kiểm tra bán trục dương."""
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        self.axes = np.asarray(self.axes, dtype=float).reshape(3)
        if np.any(self.axes <= 0):
            raise ValueError("ellipsoid semi-axes must be positive")
        self.yaw = float(self.yaw)

    # ---- Đại lượng suy ra -------------------------------------------------
    @property
    def a_min(self) -> float:
        """Bán trục nhỏ nhất; hệ số đổi giữa đơn vị gauge và mét."""
        return float(np.min(self.axes))

    @property
    def a_max(self) -> float:
        """Bán trục lớn nhất."""
        return float(np.max(self.axes))

    @property
    def bounding_radius(self) -> float:
        """Bán kính hình cầu bao ngoài ellipsoid (= a_max). Dùng cho A* seed
        và các phép kiểm tra thô."""
        return self.a_max

    def rotation(self) -> np.ndarray:
        """Ma trận quay R = Rz(yaw) từ khung vật sang khung thế giới."""
        return rotation_z(self.yaw)

    # ---- Gauge và gradient --------------------------------------------------
    def _to_body(self, X: np.ndarray) -> np.ndarray:
        """Đổi điểm sang khung vật: y = Rᵀ (x − o).
        X: (..., 3) -> (..., 3).

        Với X xếp theo hàng, Rᵀ(x − o) viết thành (x − o) @ R."""
        X = np.asarray(X, dtype=float)
        return (X - self.center) @ self.rotation()

    def gauge(self, X: np.ndarray) -> np.ndarray:
        """g(x) = ‖D⁻¹ y‖ với y là toạ độ khung vật.
        Nhận một điểm (3,) -> số thực, hoặc mảng điểm (N, 3) -> (N,)."""
        y = self._to_body(X)
        z = y / self.axes  # D⁻¹ y
        g = np.sqrt(np.sum(z * z, axis=-1))
        return float(g) if np.ndim(X) == 1 else g

    def scaled_gauge(self, X: np.ndarray) -> np.ndarray:
        """a_min · g(x): gauge đổi sang đơn vị mét. Với vật cản cầu, đại
        lượng này đúng bằng khoảng cách tới tâm."""
        return self.a_min * self.gauge(X)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """∇g(x) = R D⁻² Rᵀ (x − o) / g(x). x: (3,) -> (3,).
        Không xác định tại tâm (g = 0)."""
        x = np.asarray(x, dtype=float).reshape(3)
        R = self.rotation()
        y = R.T @ (x - self.center)  # toạ độ khung vật
        g = float(np.sqrt(np.sum((y / self.axes) ** 2)))
        if g < 1e-12:
            raise ValueError("gauge gradient is undefined at the ellipsoid center")
        return R @ (y / self.axes**2) / g

    # ---- Mặt cắt cho bộ giải DCA --------------------------------------------
    def cut(self, q: np.ndarray, need: float):
        """Mặt cắt tuyến tính hoá tại điểm q, trả về (w, b) sao cho ràng buộc
        trong QP là
            w · p ≥ b − s.
        Đây là xấp xỉ trong của ràng buộc a_min·g(p) ≥ need (need ở đơn vị
        mét, = a_min + biên mét). Trả None nếu q gần như trùng tâm
        (g(q) < 1e-9), vì khi đó gradient không xác định.

        Suy ra: g lồi nên tiếp tuyến tại q nằm dưới g:
            g(p) ≥ g(q) + ∇g(q)ᵀ (p − q).
        Vì vậy chỉ cần a_min·[g(q) + ∇g(q)ᵀ(p − q)] ≥ need − s. Nhân a_min
        và chuyển các hằng số sang vế phải:
            (a_min ∇g(q)) · p ≥ need − a_min·g(q) + (a_min ∇g(q)) · q − s.
        """
        q = np.asarray(q, dtype=float).reshape(3)
        gq = self.gauge(q)
        if gq < 1e-9:
            return None
        w = self.a_min * self.grad(q)
        b = float(need) - self.a_min * gq + float(w @ q)
        return w, b

    def violation(self, X: np.ndarray, need: float) -> np.ndarray:
        """Mức vi phạm thật của ràng buộc a_min·g(x) ≥ need, tính bằng mét:
        max(0, need − a_min·g(x)). Nhận một điểm hoặc mảng điểm."""
        return np.maximum(0.0, float(need) - self.scaled_gauge(X))

    def clearance_lower_bound(self, X: np.ndarray) -> np.ndarray:
        """Cận dưới của khe hở Euclid từ x tới bề mặt ellipsoid:
        a_min · (g(x) − 1) ≤ dist(x, ellipsoid) (xem docstring module).
        Với vật cản cầu, giá trị này đúng bằng ‖x − o‖ − r."""
        return self.a_min * (self.gauge(X) - 1.0)


def ellipsoid_series(centers: np.ndarray, axes: np.ndarray, yaws: np.ndarray) -> list:
    """Dựng danh sách Ellipsoid, mỗi phần tử ứng với một bước k của chân
    trời, từ các mảng centers (K+1, 3), axes (K+1, 3), yaws (K+1,)."""
    centers = np.asarray(centers, dtype=float)
    axes = np.asarray(axes, dtype=float)
    yaws = np.asarray(yaws, dtype=float)
    return [
        Ellipsoid(centers[k], axes[k], float(yaws[k])) for k in range(centers.shape[0])
    ]


def interpolate(ell_a: Ellipsoid, ell_b: Ellipsoid, frac: float = 0.5) -> Ellipsoid:
    """Ellipsoid tại thời điểm nằm giữa hai nút, frac ∈ [0, 1]
    (frac = 0 cho ell_a, frac = 1 cho ell_b).

    Tâm và bán trục được nội suy tuyến tính. Với vật cản cầu, cách này cho
    đúng bán kính trung bình (r_a + r_b)/2 như phía vật cản cầu dùng cho mặt
    cắt điểm giữa; nội suy theo log của bán trục thì không.

    Yaw được nội suy theo hiệu góc đã đưa về (−π, π] để tránh nhảy 2π."""
    c = (1.0 - frac) * ell_a.center + frac * ell_b.center
    a = (1.0 - frac) * ell_a.axes + frac * ell_b.axes
    dpsi = (ell_b.yaw - ell_a.yaw + np.pi) % (2.0 * np.pi) - np.pi
    yaw = ell_a.yaw + frac * dpsi
    return Ellipsoid(c, a, yaw)


def metric_margin_to_gauge(margin_m: float, ell: Ellipsoid) -> float:
    """Đổi biên mét δ sang đơn vị gauge: δ / a_min. Bộ giải làm việc trực
    tiếp với dạng đã nhân a_min (xem cut()), nên hàm này chủ yếu dùng để
    kiểm tra."""
    return float(margin_m) / ell.a_min
