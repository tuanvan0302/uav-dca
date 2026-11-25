# -*- coding: utf-8 -*-
"""
Dự báo hình dạng vật cản ellipsoid — tầng (G) của ST-DCA.

Mở rộng core/obstacle_predict.py từ vật cản cầu (tâm + bán kính) sang
ellipsoid (tâm + ba bán trục + yaw). Đầu ra là dict đúng dạng mà
core.dca_planner.plan nhận cho vật cản ellipsoid (khoá 'axes', 'yaws').

Hàm sự thật nền của một vật cản: fn(t) -> (tâm (3,), bán trục (3,), yaw).

Giả thiết đầu vào: có một tầng ước lượng hình dạng cung cấp, tại mỗi thời
điểm đo trong quá khứ, bộ (o, log a, ψ) kèm nhiễu:
  - tâm: nhiễu Gauss, độ lệch chuẩn meas_std (như obstacle_predict);
  - log-bán-trục: nhiễu Gauss, độ lệch chuẩn LOG_MEAS_STD (không thứ
    nguyên; 0,05 ứng với sai số tương đối khoảng 5% của bán trục);
  - yaw: nhiễu Gauss, độ lệch chuẩn YAW_MEAS_STD (rad).
Trong mô phỏng, các số đo này được sinh từ sự thật nền cộng nhiễu, với rng
do phía gọi cung cấp.

Mô hình dự báo:
  - Tâm: Kalman vận tốc không đổi (kalman_cv_forecast và Q_POS của
    obstacle_predict).
  - Log-bán-trục, từng trục độc lập, gồm hai lớp:
      lớp 1: hồi quy điều hoà với cơ sở [1, t, cos/sin của các hài
             1..n_harm]; tần số cơ bản chọn bằng quét lưới chu kỳ, lấy
             chu kỳ cho tổng bình phương sai số nhỏ nhất;
      lớp 2: Kalman vận tốc không đổi trên phần dư của lớp 1
             (Q_LOG = Q_RAD).
    Dự báo = lớp 1 + lớp 2. σ_log theo từng bước lấy từ hiệp phương sai
    dự báo của lớp 2, nên không gồm bất định của chính lớp 1 (tần số và hệ
    số điều hoà được coi như đã biết). Phần bất định còn thiếu được bù ở
    bước hiệu chuẩn conformal (core/conformal.py).
  - Yaw: chỉ ước lượng khi vật đủ dẹt theo phương ngang, tức tỉ lệ trục
    ngang dài/ngắn (trung bình trên lịch sử đo) > YAW_GATE_RATIO = 1,15.
      * Cổng mở: yaw dự báo bằng Kalman vận tốc không đổi trên góc đã mở
        gói (unwrap). Bất định yaw σ_ψ được đổi thành phần phồng của hai
        trục ngang: mỗi trục ngang cộng thêm (trục ngang còn lại)·sin(σ_ψ).
        Đây là xấp xỉ bậc nhất theo σ_ψ, không bảo đảm phủ ellipsoid thật
        khi vật kéo dài và sai số yaw lớn (xem
        tests/test_shape_forecast.py::test_yaw_inflation_is_first_order).
      * Cổng đóng (vật gần tròn theo phương ngang, yaw khó quan sát): yaw
        giữ ở giá trị đo cuối và hai trục ngang được đặt bằng trục lớn hơn
        trong hai trục. Hình thu được chứa ellipsoid với mọi yaw (nếu tâm
        và bán trục dự báo đúng).
      * Cổng được tính trên tỉ lệ trục đo có nhiễu, nên khi tỉ lệ thật gần
        1,15, trạng thái cổng có thể đổi giữa hai lần replan liên tiếp.
  - ρ_k = κ·sqrt(σ_vị_trí,k² + σ_hình_dạng,k²), κ = 2, với
    σ_hình_dạng,k = max_j (a_jk · σ_log,jk) (đổi bất định log sang mét:
    δa ≈ a·δ(log a)).

Bảo vệ khi ngoại suy kênh log-bán-trục. Nếu cửa sổ lịch sử chứa quá ít chu
kỳ biến dạng, khớp điều hoà có thể chọn sai tần số và ngoại suy ra giá trị
rất lớn hoặc rất nhỏ. Vì vậy mỗi trục được dự báo theo một trong ba chế
độ, chọn tự động và ghi vào khoá chẩn đoán 'axis_mode' (mã trong
MODE_NAMES):
  - TĨNH (0): biên độ điều hoà khớp được ≤ AMP_GATE, tức trục không dao
    động đáng kể. Bỏ cả thành phần điều hoà lẫn xu thế tuyến tính (khi
    ngoại suy xa, cả hai biến nhiễu đo thành trôi giả). Dự báo = trung bình
    các lần đo + thành phần mức hiện tại của bộ lọc Kalman trên phần dư;
    σ lấy từ bộ lọc đó.
  - ĐIỀU HOÀ (1): biên độ đáng kể, cửa sổ chứa ≥ MIN_CYCLES chu kỳ hiệu
    dụng, và ma trận thiết kế không gần suy biến (số điều kiện sau khi
    chuẩn hoá cột ≤ COND_MAX). Dùng dự báo hai lớp.
  - DỰ PHÒNG (2): biên độ đáng kể nhưng cửa sổ quá ngắn hoặc thiết kế suy
    biến. Dùng Kalman vận tốc không đổi trực tiếp trên log-bán-trục đo
    (không bắt được dao động nhưng không bùng nổ); σ được cộng thêm
    (amp/2)² cho phần dao động chưa mô hình hoá.
Sau đó, ở mọi chế độ, log-bán-trục dự báo bị kẹp trong
[min đo − LOG_BAND, max đo + LOG_BAND] và trong
[log A_ABS_MIN, log A_ABS_MAX], nên planner không bao giờ nhận bán trục
vô lý. Để dùng được chế độ điều hoà với biến dạng chu kỳ T, lịch sử cần
dài ≥ MIN_CYCLES·T (ví dụ T = 6 s, hist_dt = 0,25 s ⇒ n_hist ≥ 36).
"""

from __future__ import annotations

import numpy as np

from core.obstacle_predict import KAPPA, MEAS_STD, Q_POS, Q_RAD, kalman_cv_forecast

LOG_MEAS_STD = 0.05  # độ lệch chuẩn nhiễu đo log-bán-trục (giả thiết)
YAW_MEAS_STD = 0.10  # độ lệch chuẩn nhiễu đo yaw, rad (giả thiết)
Q_LOG = Q_RAD  # nhiễu quá trình Kalman cho phần dư log-bán-trục
Q_YAW = 1e-4  # nhiễu quá trình Kalman cho yaw. Phương sai dự báo
# tăng theo t³ nên Q phải nhỏ: với Q = 1e-4, σ_ψ ở
# cuối chân trời 7 s khoảng 0,185 rad; với Q cỡ Q_POS
# (0,002), σ_ψ lên tới ~0,6 rad và phần phồng
# a·sin(σ_ψ) làm bán trục dự báo quá lớn.
YAW_GATE_RATIO = 1.15  # ngưỡng tỉ lệ trục ngang dài/ngắn để ước lượng yaw

# --- Ngưỡng chọn chế độ dự báo log-bán-trục (xem docstring module)
AMP_GATE = 0.10  # biên độ điều hoà (đơn vị log) ≤ ngưỡng này ⇒ TĨNH.
# ≈ 2× LOG_MEAS_STD: khớp trên dữ liệu chỉ có nhiễu
# hiếm khi cho biên độ vượt ngưỡng này.
MIN_CYCLES = 1.5  # số chu kỳ hiệu dụng tối thiểu trong cửa sổ lịch sử
# để dùng ngoại suy điều hoà
COND_MAX = 100.0  # số điều kiện tối đa của thiết kế LS đã chuẩn hoá cột
LOG_BAND = 0.5  # kẹp dự báo trong dải đo ± LOG_BAND (đơn vị log;
# e^0,5 ≈ 1,65 lần bán trục)
A_ABS_MIN, A_ABS_MAX = 0.02, 10.0  # dải bán trục tuyệt đối (m)
MIN_HIST_FIT = 12  # số mẫu lịch sử tối thiểu để khớp hồi quy điều hoà

# Mã chế độ dự báo của một trục. PERSIST chỉ dùng trong ablation
# (shape_layers = ()): giữ nguyên giá trị đo cuối.
MODE_STATIC, MODE_HARM, MODE_FALLBACK, MODE_PERSIST = 0, 1, 2, 3
MODE_NAMES = {
    MODE_STATIC: "tĩnh",
    MODE_HARM: "điều hòa",
    MODE_FALLBACK: "dự phòng",
    MODE_PERSIST: "persistence",
}


# ----------------------------------------------------------------------
# Sự thật nền
# ----------------------------------------------------------------------
def sample_ellipsoid_truth(truth_fns, t0: float, K: int, dt: float):
    """Lấy mẫu sự thật nền tại t0 + k·dt, k = 0..K. Trả danh sách, mỗi vật
    cản một bộ (tâm (K+1, 3), bán trục (K+1, 3), yaw (K+1,))."""
    out = []
    for fn in truth_fns:
        centers = np.empty((K + 1, 3))
        axes = np.empty((K + 1, 3))
        yaws = np.empty(K + 1)
        for k in range(K + 1):
            c, a, psi = fn(t0 + k * dt)
            centers[k] = c
            axes[k] = a
            yaws[k] = psi
        out.append((centers, axes, yaws))
    return out


def truth_ellipsoid_obstacles(truth_fns, t0: float, K: int, dt: float) -> list:
    """Chế độ sự thật nền: ellipsoid thật trên chân trời, ρ_k = 0."""
    return [
        dict(centers=c, axes=a, yaws=y, rho=np.zeros(K + 1))
        for (c, a, y) in sample_ellipsoid_truth(truth_fns, t0, K, dt)
    ]


def bounding_sphere_tuples(obstacle_dicts: list) -> list:
    """Đổi các vật cản (cầu hoặc ellipsoid, theo dict của planner) sang danh
    sách hình cầu (tâm, bán kính) cho core/astar_seed.py, vốn chỉ làm việc
    với cầu tĩnh.

    Mỗi vật cản sinh ra một cầu TẠI MỖI BƯỚC k của chân trời:
      - tâm centers[k];
      - bán kính = bán trục lớn nhất tại bước k (hoặc bán kính, nếu là cầu),
        cộng nửa độ dịch tâm lớn nhất giữa bước k và các bước kề.
    Phần cộng nửa độ dịch tâm làm hợp các cầu phủ kín cả đoạn tâm nối hai
    bước liên tiếp: mọi điểm trên đoạn cách một trong hai đầu không quá nửa
    độ dài đoạn. Như vậy đường A* tránh toàn bộ vùng vật cản quét qua trên
    chân trời.

    Cách này bảo thủ theo thời gian: một điểm chỉ bị vật chiếm ở cuối chân
    trời vẫn bị chặn, dù UAV có thể đi qua đó sớm hơn. Ở hành lang hẹp, A*
    có thể không tìm được đường; khi đó plan() dùng đường thẳng. Khi tâm
    đứng yên, độ dịch bằng 0 và hợp các cầu chính là một cầu
    (tâm, max_k r_k)."""
    out = []
    for ob in obstacle_dicts:
        centers = np.asarray(ob["centers"], dtype=float)
        if centers.ndim == 1:
            centers = centers.reshape(1, 3)
        n = len(centers)
        # Bán kính cầu bao tại mỗi bước.
        if "axes" in ob:
            axes = np.asarray(ob["axes"], dtype=float)
            if axes.ndim == 1:
                axes = np.repeat(axes.reshape(1, 3), n, axis=0)
            rad = np.max(axes, axis=1)
        else:
            radii = np.asarray(ob["radii"], dtype=float)
            rad = np.full(n, float(radii)) if radii.ndim == 0 else radii.astype(float)
        # half[k] = nửa độ dịch tâm lớn nhất giữa bước k và các bước kề.
        if n > 1:
            disp = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            half = np.empty(n)
            half[0] = 0.5 * disp[0]
            half[-1] = 0.5 * disp[-1]
            if n > 2:
                half[1:-1] = 0.5 * np.maximum(disp[:-1], disp[1:])
        else:
            half = np.zeros(1)
        for k in range(n):
            out.append((centers[k].copy(), float(rad[k] + half[k])))
    return out


# ----------------------------------------------------------------------
# Lớp 1: hồi quy điều hoà trên log-bán-trục (mỗi trục có pha riêng)
# ----------------------------------------------------------------------
def _harmonic_design(tc: np.ndarray, omega: float, n_harm: int) -> np.ndarray:
    """Ma trận thiết kế với các cột
    [1, t, cos(ωt), sin(ωt), ..., cos(n_harm·ωt), sin(n_harm·ωt)]."""
    cols = [np.ones_like(tc), tc]
    for h in range(1, n_harm + 1):
        cols.append(np.cos(h * omega * tc))
        cols.append(np.sin(h * omega * tc))
    return np.column_stack(cols)


def _fit_harmonic(times: np.ndarray, y: np.ndarray, n_harm: int = 2):
    """Hồi quy điều hoà cho một trục. Tần số cơ bản chọn bằng quét lưới
    chu kỳ hai tầng:
      - thô: 160 chu kỳ cách đều theo log trên [1, 80] s;
      - tinh: 81 chu kỳ trong ±4% quanh chu kỳ tốt nhất của tầng thô.
    Ở mỗi chu kỳ giải bình phương tối thiểu, giữ nghiệm có tổng bình phương
    phần dư nhỏ nhất. Cần tầng tinh vì sai số tương đối ε của tần số gây
    sai pha cỡ ε·ω·t khi ngoại suy, tăng dần theo thời gian.
    Trả (omega, coef, resid)."""
    tc = times - times[0]  # đo thời gian từ lần đo đầu tiên

    def _try(period):
        # Khớp LS với một chu kỳ cố định; trả (SSE, omega, hệ số, phần dư).
        omega = 2.0 * np.pi / period
        A = _harmonic_design(tc, omega, n_harm)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        return float(resid @ resid), omega, coef, resid

    best = None
    for period in np.geomspace(1.0, 80.0, 160):
        cand = _try(period)
        if best is None or cand[0] < best[0]:
            best = cand
    t_best = 2.0 * np.pi / best[1]
    for period in np.geomspace(t_best / 1.04, t_best * 1.04, 81):
        cand = _try(period)
        if cand[0] < best[0]:
            best = cand
    _, omega, coef, resid = best
    return omega, coef, resid


def _harmonic_eval(
    times0: float, omega: float, coef: np.ndarray, n_harm: int, t_eval: np.ndarray
) -> np.ndarray:
    """Giá trị của mô hình điều hoà đã khớp tại các thời điểm t_eval
    (times0 là gốc thời gian đã dùng khi khớp)."""
    A = _harmonic_design(np.asarray(t_eval, dtype=float) - times0, omega, n_harm)
    return A @ coef


def _harmonic_diagnostics(
    times: np.ndarray, omega: float, coef: np.ndarray, n_harm: int
):
    """Các đại lượng dùng để chọn chế độ dự báo:
    - amp: biên độ (nửa hiệu đỉnh–đáy) của riêng phần điều hoà (không gồm
      hằng số và xu thế), tính trên lưới 200 điểm của một chu kỳ cơ bản;
    - period_eff: chu kỳ hiệu dụng = chu kỳ cơ bản / bậc của hài có biên độ
      lớn nhất. Phép quét có thể chọn chu kỳ cơ bản gấp đôi chu kỳ thật và
      để hài bậc 2 mang tín hiệu; chu kỳ hiệu dụng mới là chu kỳ vật lý;
    - cycles: số chu kỳ hiệu dụng nằm trong cửa sổ lịch sử;
    - cond: số điều kiện của ma trận thiết kế (tại các thời điểm đo) sau
      khi chuẩn hoá mỗi cột về chuẩn 1."""
    period = 2.0 * np.pi / omega
    tg = times[0] + np.linspace(0.0, period, 200)
    A = _harmonic_design(tg - times[0], omega, n_harm)
    harm_only = A[:, 2:] @ coef[2:]  # bỏ hai cột đầu: hằng số và t
    amp = 0.5 * (float(np.max(harm_only)) - float(np.min(harm_only)))
    # Biên độ của hài h: sqrt(c_h² + s_h²), hệ số cos/sin ở cột 2h, 2h+1.
    amp_h = [
        float(np.hypot(coef[2 + 2 * (h - 1)], coef[3 + 2 * (h - 1)]))
        for h in range(1, n_harm + 1)
    ]
    h_star = int(np.argmax(amp_h)) + 1
    period_eff = period / h_star
    cycles = (times[-1] - times[0]) / period_eff if period_eff > 0 else 0.0
    Aw = _harmonic_design(times - times[0], omega, n_harm)
    norms = np.linalg.norm(Aw, axis=0)
    norms[norms == 0.0] = 1.0
    cond = float(np.linalg.cond(Aw / norms))
    return amp, period_eff, cycles, cond


# ----------------------------------------------------------------------
# Dự báo log-bán-trục cho MỘT trục
# ----------------------------------------------------------------------
def _forecast_axis_log(
    times: np.ndarray,
    y: np.ndarray,
    t_eval: np.ndarray,
    K: int,
    dt: float,
    n_harm: int,
    log_meas_std: float,
    layers: tuple,
):
    """Dự báo log-bán-trục của một trục trên chân trời.

    times, y: thời điểm và giá trị log-bán-trục đo được trong quá khứ;
    t_eval: các thời điểm dự báo t_now + k·dt, k = 0..K.

    layers = ("harm", "resid"): cấu hình đầy đủ, chọn chế độ
    tĩnh / điều hoà / dự phòng như mô tả ở docstring module.
    Các cấu hình ablation dùng thẳng đầu ra của từng lớp, không chọn chế độ:
      ("harm",)  : chỉ lớp 1;
      ("resid",) : chỉ Kalman trên log-bán-trục đo;
      ()         : giữ giá trị đo cuối (persistence).
    Bước kẹp dải ở cuối áp cho mọi cấu hình.
    Trả (loga_fc (K+1,), slog (K+1,), info dict)."""
    full = ("harm" in layers) and ("resid" in layers)
    info = dict(mode=MODE_FALLBACK, period=0.0, cycles=0.0, amp=0.0, cond=0.0)

    # Khớp lớp 1 nếu có đủ mẫu; nếu không, fit = None.
    fit = None
    if len(times) >= MIN_HIST_FIT:
        omega, coef, resid = _fit_harmonic(times, y, n_harm)
        amp, period_eff, cycles, cond = _harmonic_diagnostics(
            times, omega, coef, n_harm
        )
        fit = (omega, coef, resid)
        info.update(period=period_eff, cycles=cycles, amp=amp, cond=cond)

    def _cv(series, q, var):
        # Kalman vận tốc không đổi một chiều trên một dãy theo times.
        return kalman_cv_forecast(
            np.asarray(series, dtype=float).reshape(-1, 1), times, K, dt, q, var
        )

    if not layers:
        # Persistence: giữ giá trị đo cuối. σ lấy từ bộ lọc Kalman trên
        # chính dãy đo, chỉ để ρ có giá trị xác định.
        _, cs = _cv(y, Q_LOG, log_meas_std**2)
        loga, slog = np.full(K + 1, float(y[-1])), cs
        info["mode"] = MODE_PERSIST
    elif full and fit is not None and info["amp"] <= AMP_GATE:
        # TĨNH: mức trung bình + mức hiện tại của Kalman trên phần dư
        # (rf[0]), không ngoại suy xu thế.
        level = float(np.mean(y))
        resid = y - level
        rf, rs = _cv(resid, Q_LOG, max(float(np.var(resid)), 1e-8))
        loga = level + np.full(K + 1, float(rf[0, 0]))
        slog = rs
        info["mode"] = MODE_STATIC
    elif full and (
        fit is None or info["cycles"] < MIN_CYCLES or info["cond"] > COND_MAX
    ):
        # DỰ PHÒNG: Kalman trên chính log-bán-trục đo. Nếu đã khớp được
        # biên độ, cộng (amp/2)² vào phương sai cho phần dao động chưa mô
        # hình hoá.
        cv, cs = _cv(y, Q_LOG, log_meas_std**2)
        loga = cv[:, 0]
        slog = cs
        if fit is not None:
            slog = np.sqrt(cs**2 + (0.5 * info["amp"]) ** 2)
        info["mode"] = MODE_FALLBACK
    elif "harm" in layers and fit is not None:
        # ĐIỀU HOÀ: lớp 1 tại t_eval, cộng dự báo Kalman của phần dư (lớp 2)
        # nếu có "resid". Phương sai nhiễu đo của bộ lọc phần dư lấy bằng
        # phương sai mẫu của phần dư.
        omega, coef, resid = fit
        loga = _harmonic_eval(times[0], omega, coef, n_harm, t_eval)
        rf, rs = _cv(resid, Q_LOG, max(float(np.var(resid)), 1e-8))
        slog = rs
        if "resid" in layers:
            loga = loga + rf[:, 0]
        info["mode"] = MODE_HARM
    else:
        # Ablation ("resid",), hoặc ("harm",) khi chưa khớp được: Kalman
        # trên log-bán-trục đo.
        cv, cs = _cv(y, Q_LOG, log_meas_std**2)
        loga, slog = cv[:, 0], cs
        info["mode"] = MODE_FALLBACK

    # Kẹp dải: trong dải đo ± LOG_BAND và trong dải tuyệt đối.
    lo = max(float(np.min(y)) - LOG_BAND, float(np.log(A_ABS_MIN)))
    hi = min(float(np.max(y)) + LOG_BAND, float(np.log(A_ABS_MAX)))
    loga = np.clip(loga, lo, hi)
    return loga, slog, info


# ----------------------------------------------------------------------
# Chế độ ước lượng: số đo nhiễu trong quá khứ -> dự báo ellipsoid
# ----------------------------------------------------------------------
def kalman_ellipsoid_obstacles(
    truth_fns,
    t_now: float,
    K: int,
    dt: float,
    rng: np.random.Generator,
    n_hist: int = 24,
    hist_dt: float = 0.25,
    meas_std: float = MEAS_STD,
    kappa: float = KAPPA,
    log_meas_std: float = LOG_MEAS_STD,
    yaw_meas_std: float = YAW_MEAS_STD,
    n_harm: int = 2,
    shape_layers: tuple = ("harm", "resid"),
) -> list:
    """Dự báo ellipsoid trên chân trời từ n_hist lần đo nhiễu trong quá khứ
    (cách nhau hist_dt, lần cuối tại t_now).

    shape_layers chỉ dùng cho ablation (xem _forecast_axis_log); mặc định
    ("harm", "resid") là cấu hình đầy đủ.

    Trả danh sách dict, mỗi vật cản một dict:
      centers (K+1, 3), axes (K+1, 3), yaws (K+1,), rho (K+1,) : đầu vào
          của planner;
      sigma_log (K+1, 3) : độ lệch chuẩn dự báo log-bán-trục;
      sigma_yaw (K+1,)   : độ lệch chuẩn dự báo yaw (0 nếu cổng yaw đóng);
      axes_pure (K+1, 3) : bán trục dự báo trước khi phồng/tròn hoá theo yaw;
      axis_mode (3,)     : chế độ dự báo của từng trục (mã MODE_NAMES);
      axis_period (3,)   : chu kỳ hiệu dụng khớp được (s);
      axis_cycles (3,)   : số chu kỳ hiệu dụng trong cửa sổ lịch sử.
    Planner chỉ đọc bốn khoá đầu; các khoá còn lại dùng để chẩn đoán.
    """
    times = t_now - hist_dt * np.arange(n_hist - 1, -1, -1.0)  # các lần đo
    t_eval = t_now + dt * np.arange(K + 1)  # các bước dự báo
    out = []
    for fn in truth_fns:
        # Sinh số đo: sự thật nền + nhiễu (bán trục nhiễu theo log).
        true_c = np.array([fn(t)[0] for t in times])
        true_a = np.array([fn(t)[1] for t in times])
        true_yaw = np.array([fn(t)[2] for t in times])
        meas_c = true_c + rng.normal(0.0, meas_std, size=true_c.shape)
        meas_loga = np.log(true_a) + rng.normal(0.0, log_meas_std, size=true_a.shape)
        meas_yaw = true_yaw + rng.normal(0.0, yaw_meas_std, size=true_yaw.shape)

        # --- Tâm: Kalman vận tốc không đổi, như obstacle_predict.
        fc, sc = kalman_cv_forecast(meas_c, times, K, dt, Q_POS, meas_std**2)

        # --- Log-bán-trục: dự báo từng trục độc lập.
        loga_fc = np.empty((K + 1, 3))
        slog = np.empty((K + 1, 3))
        axis_mode = np.empty(3, dtype=int)
        axis_period = np.empty(3)
        axis_cycles = np.empty(3)
        for j in range(3):
            loga_j, slog_j, info_j = _forecast_axis_log(
                times,
                meas_loga[:, j],
                t_eval,
                K,
                dt,
                n_harm,
                log_meas_std,
                shape_layers,
            )
            loga_fc[:, j] = loga_j
            slog[:, j] = slog_j
            axis_mode[j] = info_j["mode"]
            axis_period[j] = info_j["period"]
            axis_cycles[j] = info_j["cycles"]
        axes_fc = np.exp(loga_fc)
        axes_pure = axes_fc.copy()  # bán trục trước khi xử lý yaw

        # --- Yaw: cổng theo tỉ lệ trục ngang dài/ngắn trung bình.
        meas_axes = np.exp(meas_loga)
        ratio_h = float(
            np.mean(np.max(meas_axes[:, :2], axis=1) / np.min(meas_axes[:, :2], axis=1))
        )
        if ratio_h > YAW_GATE_RATIO:
            # Cổng mở: Kalman trên yaw đã unwrap, rồi đưa kết quả về [−π, π).
            yaw_un = np.unwrap(meas_yaw)
            yaw_fc_raw, syaw = kalman_cv_forecast(
                yaw_un.reshape(-1, 1), times, K, dt, Q_YAW, yaw_meas_std**2
            )
            yaw_fc = (yaw_fc_raw[:, 0] + np.pi) % (2.0 * np.pi) - np.pi
            # Phồng hai trục ngang theo bất định yaw: mỗi trục cộng thêm
            # (trục ngang còn lại)·sin(σ_ψ). σ_ψ được chặn ở π/2. Đây là xấp
            # xỉ bậc nhất, không bảo đảm phủ ellipsoid thật (xem docstring
            # module).
            s = np.sin(np.minimum(syaw, np.pi / 2.0))
            a1 = axes_fc[:, 0] + axes_fc[:, 1] * s
            a2 = axes_fc[:, 1] + axes_fc[:, 0] * s
            axes_fc[:, 0], axes_fc[:, 1] = a1, a2
        else:
            # Cổng đóng: giữ yaw đo cuối và đặt hai trục ngang bằng trục lớn
            # hơn. Hình tròn theo phương ngang này chứa ellipsoid với mọi yaw.
            yaw_fc = np.full(K + 1, float(meas_yaw[-1]))
            syaw = np.zeros(K + 1)
            amax_h = np.max(axes_fc[:, :2], axis=1)
            axes_fc[:, 0] = amax_h
            axes_fc[:, 1] = amax_h

        # --- ρ_k: gộp bất định vị trí và hình dạng (đổi từ log sang mét).
        s_shape = np.max(axes_fc * slog, axis=1)
        rho = kappa * np.sqrt(sc**2 + s_shape**2)
        out.append(
            dict(
                centers=fc,
                axes=axes_fc,
                yaws=yaw_fc,
                rho=rho,
                sigma_log=slog,
                sigma_yaw=syaw,
                axes_pure=axes_pure,
                axis_mode=axis_mode,
                axis_period=axis_period,
                axis_cycles=axis_cycles,
            )
        )
    return out


def forecast_error_ellipsoid(
    truth_fns, predicted: list, t0: float, K: int, dt: float
) -> dict:
    """Sai số dự báo tại cuối chân trời (bước K), lấy lớn nhất trên các vật
    cản: tâm (m), log-bán-trục (không thứ nguyên, lớn nhất trên ba trục),
    yaw (rad, hiệu góc đưa về [−π, π))."""
    truth = sample_ellipsoid_truth(truth_fns, t0, K, dt)
    err_c = max(
        float(np.linalg.norm(predicted[m]["centers"][K] - truth[m][0][K]))
        for m in range(len(truth_fns))
    )
    err_a = max(
        float(np.max(np.abs(np.log(predicted[m]["axes"][K]) - np.log(truth[m][1][K]))))
        for m in range(len(truth_fns))
    )
    dyaw = np.array(
        [predicted[m]["yaws"][K] - truth[m][2][K] for m in range(len(truth_fns))]
    )
    err_yaw = float(np.max(np.abs((dyaw + np.pi) % (2.0 * np.pi) - np.pi)))
    return dict(center=err_c, log_axes=err_a, yaw=err_yaw)
