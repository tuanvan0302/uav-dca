# -*- coding: utf-8 -*-
"""
safety_margin.py — biên an toàn (M) của ST-DCA.

Đây là nửa "biên" của hạng mục H1 trong báo cáo ST-DCA (H1 = biên an
toàn + hiệu chuẩn conformal; nửa conformal nằm ở core/conformal.py). Các
cấu hình A1 và A3 dùng biên này; A0 và A2 dùng chế độ legacy.

Biên an toàn hiệu dụng của ràng buộc (vật cản m, bước k):

    b_{m,k} = d + δ_trôi_m + δ_mẫu_m + ρ_{m,k}(ε)

Các hạng tử (đơn vị mét; với vật cản ellipsoid, planner tự đổi sang đơn
vị gauge, xem core/ellipsoid.py):

  - d          : khoảng cách an toàn yêu cầu (mặc định 0,5 m).
  - δ_trôi_m   : hạng tử TRÔI. Trong khoảng trễ T_r từ lúc đo tới lúc kế
                 hoạch được thực thi, vật cản vẫn di chuyển:
                     δ_trôi = v̂_m · T_r + ½ · a_max_m · T_r² + e_bám
                 với
                     T_r = T_replan + T_giải(chu kỳ trước).
                 Dùng thời gian giải đo được của chu kỳ LIỀN TRƯỚC vì thời
                 gian giải của chu kỳ hiện tại chưa biết khi đang dựng bài
                 toán.
  - δ_mẫu_m    : hạng tử MẪU. Ràng buộc chỉ được kiểm tra tại các nút; giữa
                 hai nút, độ lệch tương đối giữa quỹ đạo UAV và vật cản là
                 một cung parabol, võng tối đa
                     δ_mẫu = (u_max + a_max_m) · Δt² / 8.
  - ρ_{m,k}(ε) : hạng tử BẤT ĐỊNH của dự báo, hiệu chuẩn bằng conformal
                 prediction. Giá trị được tra từ bảng do core/conformal.py
                 cung cấp; module này chỉ cộng các hạng tử lại.

Cách đưa biên vào planner: core/dca_planner.py dùng ngưỡng
need = R + d_safe + extra_margin + RHO[m, k], trong đó RHO là mảng (K+1,)
đi kèm mỗi vật cản và extra_margin là một số dùng chung. Vì δ_trôi_m và
δ_mẫu_m phụ thuộc vật cản, chúng được cộng vào mảng rho của từng vật cản
và extra_margin được đặt bằng 0:
  - compose_rho_traj_for_planner(): dùng ρ_traj(ε), một quantile mức quỹ
    đạo (một số cho mỗi nhóm tuổi và ε), áp như nhau cho mọi k. Đây là
    đường dùng khi chạy.
  - compose_rho_for_planner(): dùng mảng ρ_k(ε) theo từng bước, chỉ để
    chẩn đoán.

Chế độ "legacy" (MarginConfig(mode="legacy")) tái tạo cấu hình biên cũ
cho các ablation A0/A2: extra_margin = 0,03 m hằng số và
ρ_k = κ·sqrt(σ_vị_trí² + σ_bán_kính²) với κ = 2 (xem legacy_rho).

Trần cửa sổ trễ: δ_trôi có thể được tính trên cửa sổ hiệu lực
T_r^hl = min(T_r, T_cap). Trần chỉ áp vào δ_trôi; T_r đo được vẫn được giữ
nguyên để ghi log. ReplanClock nhận T_cap (mặc định None = không chặn) và
qua window_info() cho biết chu kỳ hiện tại có bị chặn hay không
(T_r > T_cap).

e_bám (sai số bám quỹ đạo của bộ điều khiển, trong code là
tracking_error) hiện chưa có số đo: TRACKING_ERROR_MEASURED = False và
giá trị mặc định là 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------
# Hằng số mặc định
# ------------------------------------------------------------------
TRACKING_ERROR_MEASURED = False  # True khi e_bám được đặt theo số đo thực tế
TRACKING_ERROR_DEFAULT = 0.0  # m; giá trị thay thế khi chưa có số đo

D_SAFE_DEFAULT = 0.5  # d (m), bằng PlannerParams.d_safe
LEGACY_DELTA = 0.03  # δ hằng số của chế độ legacy (m)
LEGACY_KAPPA = 2.0  # hệ số κ của ρ legacy (bội số độ lệch chuẩn)

# Trần mặc định của cửa sổ trễ: 8 × chu kỳ replan danh định 0,25 s.
T_CAP_DRAFT = 2.0  # s


@dataclass
class MarginConfig:
    """Cấu hình biên an toàn.

    mode = "h1"     : biên gồm d + δ_trôi + δ_mẫu + ρ(ε) (mặc định).
    mode = "legacy" : biên hằng δ = 0,03 m và ρ theo κ (xem legacy_rho),
                      dùng cho ablation A0/A2.
    """

    mode: str = "h1"
    d_safe: float = D_SAFE_DEFAULT
    u_max: float = 4.0  # m/s², bằng PlannerParams.u_max
    dt: float = 0.25  # s, bằng PlannerParams.dt
    T_replan: float = 0.25  # s, chu kỳ lập lại kế hoạch danh định
    # m, sai số bám e_bám của bộ điều khiển (xem TRACKING_ERROR_MEASURED)
    tracking_error: float = TRACKING_ERROR_DEFAULT
    legacy_delta: float = LEGACY_DELTA
    legacy_kappa: float = LEGACY_KAPPA

    def __post_init__(self):
        """Từ chối mode lạ ngay khi tạo cấu hình."""
        if self.mode not in ("h1", "legacy"):
            raise ValueError(f"invalid mode: {self.mode!r}")


# ------------------------------------------------------------------
# Cửa sổ trễ và hai hạng tử xác định
# ------------------------------------------------------------------
def latency_window(T_replan: float, prev_solve_time: float | None) -> float:
    """T_r = T_replan + T_giải(chu kỳ trước).

    prev_solve_time là thời gian giải đo được của chu kỳ liền trước. Ở chu
    kỳ đầu tiên chưa có giá trị này (None), khi đó T_r = T_replan.
    """
    if prev_solve_time is None:
        return float(T_replan)
    if prev_solve_time < 0:
        raise ValueError("prev_solve_time must be non-negative")
    return float(T_replan) + float(prev_solve_time)


def effective_latency_window(
    T_replan: float, prev_solve_time: float | None, T_cap: float | None
) -> float:
    """Cửa sổ hiệu lực dùng để tính δ_trôi:

        T_r^hl = min(T_r, T_cap),  T_r = latency_window(...)

    T_cap = None nghĩa là không chặn, trả về đúng T_r.
    """
    T_r = latency_window(T_replan, prev_solve_time)
    if T_cap is None:
        return T_r
    if T_cap <= 0:
        raise ValueError("T_cap must be positive")
    return min(T_r, float(T_cap))


class ReplanClock:
    """Theo dõi cửa sổ trễ T_r qua các chu kỳ replan: lưu thời gian giải
    đo được của chu kỳ liền trước.

    Cách dùng, mỗi chu kỳ:
        T_r = clock.effective_window()   # tính biên với cửa sổ này
        ... giải ...
        clock.update(thời_gian_giải_đo_được)

    Khi có trần (T_cap khác None):
      - window()          : T_r chưa chặn (giá trị đo, dùng để ghi log).
      - effective_window(): min(T_r, T_cap), cửa sổ dùng để tính δ_trôi.
      - capped()          : True khi và chỉ khi T_r > T_cap
                            (T_r == T_cap không tính là bị chặn).
      - window_info()     : dict {T_r, T_r_eff, capped, T_cap} để ghi log.
    Khi T_cap = None, effective_window() == window() và capped() luôn False.
    """

    def __init__(self, T_replan: float, T_cap: float | None = None):
        """T_replan: chu kỳ replan danh định (s); T_cap: trần cửa sổ trễ
        (s), None = không chặn."""
        self.T_replan = float(T_replan)
        if T_cap is not None and T_cap <= 0:
            raise ValueError("T_cap must be positive")
        self.T_cap = None if T_cap is None else float(T_cap)
        self._prev_solve_time: float | None = None  # None ở chu kỳ đầu

    def window(self) -> float:
        """T_r của chu kỳ hiện tại, chưa chặn."""
        return latency_window(self.T_replan, self._prev_solve_time)

    def effective_window(self) -> float:
        """min(T_r, T_cap); bằng window() khi T_cap = None."""
        return effective_latency_window(
            self.T_replan, self._prev_solve_time, self.T_cap
        )

    def capped(self) -> bool:
        """True khi có trần và T_r > T_cap."""
        return self.T_cap is not None and self.window() > self.T_cap

    def window_info(self) -> dict:
        """Thông tin cửa sổ của chu kỳ hiện tại để ghi log:
        T_r     : cửa sổ chưa chặn (s);
        T_r_eff : cửa sổ sau khi áp trần (s), dùng để tính δ_trôi;
        capped  : True khi T_r > T_cap;
        T_cap   : trần đang áp (s), hoặc None."""
        return dict(
            T_r=self.window(),
            T_r_eff=self.effective_window(),
            capped=self.capped(),
            T_cap=self.T_cap,
        )

    def update(self, solve_time: float) -> None:
        """Lưu thời gian giải đo được của chu kỳ vừa xong; giá trị này tạo
        nên T_r của chu kỳ kế tiếp."""
        if solve_time < 0:
            raise ValueError("solve_time must be non-negative")
        self._prev_solve_time = float(solve_time)


def delta_drift(
    v_hat: float,
    a_max: float,
    T_r: float,
    tracking_error: float = TRACKING_ERROR_DEFAULT,
) -> float:
    """Hạng tử trôi δ_trôi = v̂ · T_r + ½ · a_max · T_r² + e_bám  (m).

    Hàm không tự áp trần; khi dùng trần, phía gọi truyền
    ReplanClock.effective_window() làm T_r.

    Args:
        v_hat: tốc độ ước lượng của vật cản (m/s).
        a_max: cận trên gia tốc của vật cản (m/s²).
        T_r: cửa sổ trễ (s).
        tracking_error: sai số bám e_bám của bộ điều khiển (m).

    Returns:
        Độ dịch tối đa của vật cản trong cửa sổ trễ (m).

    Raises:
        ValueError: khi v_hat, a_max hoặc T_r âm.
    """
    if v_hat < 0 or a_max < 0 or T_r < 0:
        raise ValueError("v_hat, a_max, T_r must be non-negative")
    return float(v_hat * T_r + 0.5 * a_max * T_r**2 + tracking_error)


def delta_sample(u_max: float, a_max: float, dt: float) -> float:
    """δ_mẫu = (u_max + a_max) · Δt² / 8  (m).

    Ví dụ: u_max = 4, a_max = 0 (vật cản vận tốc không đổi), Δt = 0,25
    cho 0,03125 m.
    """
    if u_max < 0 or a_max < 0 or dt <= 0:
        raise ValueError("u_max, a_max must be non-negative; dt must be positive")
    return float((u_max + a_max) * dt**2 / 8.0)


def estimate_speed(centers_forecast: np.ndarray, dt: float) -> float:
    """Ước lượng tốc độ v̂_m từ quỹ đạo tâm dự báo (K+1, 3) bằng sai phân
    ở đầu chân trời: ‖ô_1 − ô_0‖/Δt. Trả 0 nếu chỉ có một điểm."""
    c = np.asarray(centers_forecast, dtype=float)
    if c.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(c[1] - c[0]) / dt)


# ------------------------------------------------------------------
# Chế độ legacy (ablation A0/A2)
# ------------------------------------------------------------------
def legacy_rho(
    sigma_pos: np.ndarray, sigma_rad: np.ndarray, kappa: float = LEGACY_KAPPA
) -> np.ndarray:
    """ρ_k = κ · sqrt(σ_vị_trí,k² + σ_bán_kính,k²).

    Đây là biên kiểu Gauss lấy κ lần độ lệch chuẩn dự báo, không gắn với
    một mức xác suất bao phủ nào. Chỉ dùng ở chế độ legacy.
    """
    sp = np.asarray(sigma_pos, dtype=float)
    sr = np.asarray(sigma_rad, dtype=float)
    return kappa * np.sqrt(sp**2 + sr**2)


def legacy_extra_margin(cfg: MarginConfig) -> float:
    """extra_margin của planner ở chế độ legacy: δ = 0,03 m."""
    return float(cfg.legacy_delta)


# ------------------------------------------------------------------
# Lắp ráp biên cho một vật cản
# ------------------------------------------------------------------
def margin_components(
    cfg: MarginConfig, v_hat: float, a_max: float, T_r: float, rho_eps, K: int = 28
) -> dict:
    """Các thành phần biên của MỘT vật cản ở chế độ "h1".

    rho_eps: một số ρ_traj(ε) (quantile mức quỹ đạo, được lặp lại cho mọi
    k), hoặc mảng (K+1,) ρ_k(ε) theo từng bước. K chỉ dùng khi rho_eps là
    một số.

    Trả dict:
      d, delta_drift, delta_sample : số (m);
      rho_eps      : mảng (K+1,);
      margin_per_k : δ_trôi + δ_mẫu + ρ(ε), mảng (K+1,);
      b_per_k      : d + margin_per_k. Đây là phần cộng thêm ngoài bán
                     kính; planner cộng tiếp bán kính dự báo.
    """
    if cfg.mode != "h1":
        raise ValueError(
            "margin_components requires mode='h1'; "
            "mode='legacy' uses legacy_extra_margin/legacy_rho"
        )
    rho = np.asarray(rho_eps, dtype=float)
    if rho.ndim == 0:
        rho = np.full(K + 1, float(rho))
    dd = delta_drift(v_hat, a_max, T_r, cfg.tracking_error)
    ds = delta_sample(cfg.u_max, a_max, cfg.dt)
    margin = dd + ds + rho
    return dict(
        d=cfg.d_safe,
        delta_drift=dd,
        delta_sample=ds,
        rho_eps=rho,
        margin_per_k=margin,
        b_per_k=cfg.d_safe + margin,
    )


def compose_rho_for_planner(
    rho_eps: np.ndarray, delta_drift_m: float, delta_sample_m: float
) -> np.ndarray:
    """[Chẩn đoán] Mảng 'rho' đưa cho planner khi dùng ρ theo từng bước:
        rho_planner[k] = ρ_{m,k}(ε) + δ_trôi_m + δ_mẫu_m,
    đi kèm PlannerParams.extra_margin = 0 và d_safe giữ nguyên."""
    rho = np.asarray(rho_eps, dtype=float)
    return rho + float(delta_drift_m) + float(delta_sample_m)


def compose_rho_traj_for_planner(
    rho_traj: float, delta_drift_m: float, delta_sample_m: float, K: int = 28
) -> np.ndarray:
    """Mảng 'rho' đưa cho planner khi dùng quantile mức quỹ đạo ρ_traj(ε):
        rho_planner[k] = ρ_traj(ε) + δ_trôi_m + δ_mẫu_m   với mọi k,
    đi kèm extra_margin = 0. Khi đó ngưỡng của planner là
    R + d + rho_planner = R + b_m. ρ_traj là quantile của sai số lớn nhất
    trên cả chân trời, nên dùng cùng một giá trị cho mọi k là bảo thủ."""
    return np.full(
        K + 1, float(rho_traj) + float(delta_drift_m) + float(delta_sample_m)
    )
