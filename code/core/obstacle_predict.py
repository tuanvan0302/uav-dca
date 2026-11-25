# -*- coding: utf-8 -*-
"""
Biểu diễn và dự báo vật cản cầu. Với mỗi vật cản và mỗi bước k của chân
trời, module trả về tâm dự báo ô_mk, bán kính dự báo r_mk và biên ρ_k, ở
đúng dạng dict mà core.dca_planner.plan nhận:
    dict(centers=(K+1, 3), radii=(K+1,), rho=(K+1,)).

Hai chế độ:

  - Sự thật nền (truth_obstacles): lấy mẫu trực tiếp quỹ đạo thật của vật
    cản, ρ_k = 0. Dùng làm đối chứng: sai số dự báo bằng 0.

  - Ước lượng (kalman_obstacles): chỉ dùng các quan sát có nhiễu trong quá
    khứ. Tâm được ước lượng bằng bộ lọc Kalman mô hình vận tốc không đổi
    (trạng thái vị trí–vận tốc). Bán kính dùng bộ lọc Kalman một chiều cùng
    mô hình (trạng thái bán kính–tốc độ đổi bán kính), để bám được vật cản
    co giãn. Sau đó ngoại suy trên cả chân trời và đặt
        ρ_k = κ · sqrt(σ_vị_trí,k² + σ_bán_kính,k²),   κ = 2,0.
"""

from __future__ import annotations

import numpy as np

from sim.dynamic_scenarios import sample_truth

KAPPA = 2.0  # bội số độ lệch chuẩn trong ρ_k
MEAS_STD = 0.05  # độ lệch chuẩn nhiễu đo vị trí (m)
# Cường độ nhiễu quá trình (phổ gia tốc) của mô hình vận tốc không đổi.
# Phương sai dự báo do nhiễu quá trình tăng theo t³, nên giá trị lớn làm σ
# ở cuối chân trời phình rất nhanh. Q_RAD nhỏ hơn Q_POS vì bán kính thường
# thay đổi chậm.
Q_POS = 0.002  # cho tâm
Q_RAD = 0.0005  # cho bán kính


def truth_obstacles(truth_fns, t0: float, K: int, dt: float) -> list:
    """Chế độ sự thật nền: tâm và bán kính thật tại t0 + k·dt, k = 0..K;
    ρ_k = 0."""
    return [
        dict(centers=c, radii=r, rho=np.zeros(K + 1))
        for (c, r) in sample_truth(truth_fns, t0, K, dt)
    ]


def kalman_cv_forecast(
    meas: np.ndarray, times: np.ndarray, K: int, dt: float, q: float, meas_var: float
) -> tuple[np.ndarray, np.ndarray]:
    """Lọc Kalman mô hình vận tốc không đổi rồi ngoại suy K bước.

    Dùng chung cho tâm vật cản (module này) và cho bán trục/yaw của
    ellipsoid (core/shape_forecast.py).

    Args:
        meas: (n, d) dãy đo vị trí (đơn vị của đại lượng được lọc).
        times: (n,) thời điểm đo (s), tăng dần.
        K: số bước ngoại suy.
        dt: bước ngoại suy (s).
        q: cường độ nhiễu quá trình (phổ gia tốc), xem Q_POS / Q_RAD.
        meas_var: phương sai nhiễu đo.

    Returns:
        (out, std): out (K+1, d) dự báo vị trí, out[0] là ước lượng tại
        lần đo cuối; std (K+1,) độ lệch chuẩn dự báo trung bình theo các
        trục, sqrt(trace(P_vị_trí)/d).
    """
    d = meas.shape[1]
    n = len(times)
    # Trạng thái x = [vị trí (d), vận tốc (d)], khởi tạo từ lần đo đầu với
    # vận tốc 0.
    x = np.concatenate([meas[0], np.zeros(d)])
    P = np.eye(2 * d) * (meas_var if d == 1 else 1.0)
    # Chỉ đo được vị trí: H = [I 0].
    H = np.zeros((d, 2 * d))
    H[:, :d] = np.eye(d)
    for i in range(1, n):
        h = times[i] - times[i - 1]
        # Ma trận chuyển trạng thái: vị trí += h · vận tốc.
        F = np.eye(2 * d)
        F[:d, d:] = h * np.eye(d)
        # Nhiễu quá trình của mô hình gia tốc trắng liên tục, rời rạc hoá
        # trên khoảng h.
        Q = q * np.block(
            [
                [h**3 / 3.0 * np.eye(d), h**2 / 2.0 * np.eye(d)],
                [h**2 / 2.0 * np.eye(d), h * np.eye(d)],
            ]
        )
        # Bước dự đoán.
        x = F @ x
        P = F @ P @ F.T + Q
        # Bước cập nhật với lần đo thứ i.
        S = H @ P @ H.T + meas_var * np.eye(d)
        Kg = P @ H.T @ np.linalg.inv(S)
        x = x + Kg @ (meas[i] - H @ x)
        P = (np.eye(2 * d) - Kg @ H) @ P
    # Ngoại suy trên chân trời: chỉ lặp bước dự đoán, không còn lần đo.
    out = np.empty((K + 1, d))
    std = np.empty(K + 1)
    xf, Pf = x.copy(), P.copy()
    out[0] = xf[:d]
    std[0] = np.sqrt(np.trace(Pf[:d, :d]) / d)
    for k in range(1, K + 1):
        F = np.eye(2 * d)
        F[:d, d:] = dt * np.eye(d)
        Q = q * np.block(
            [
                [dt**3 / 3.0 * np.eye(d), dt**2 / 2.0 * np.eye(d)],
                [dt**2 / 2.0 * np.eye(d), dt * np.eye(d)],
            ]
        )
        xf = F @ xf
        Pf = F @ Pf @ F.T + Q
        out[k] = xf[:d]
        std[k] = np.sqrt(np.trace(Pf[:d, :d]) / d)
    return out, std


def kalman_obstacles(
    truth_fns,
    t_now: float,
    K: int,
    dt: float,
    rng: np.random.Generator,
    n_hist: int = 8,
    hist_dt: float = 0.25,
    meas_std: float = MEAS_STD,
    kappa: float = KAPPA,
) -> list:
    """Chế độ ước lượng: sinh n_hist quan sát có nhiễu trong quá khứ (cách
    nhau hist_dt, lần cuối tại t_now), lọc Kalman, rồi dự báo trên chân trời
    kèm ρ_k. rng do phía gọi cung cấp để kết quả tái lập được."""
    # Các thời điểm đo: t_now − (n_hist−1)·hist_dt, ..., t_now.
    times = t_now - hist_dt * np.arange(n_hist - 1, -1, -1.0)
    out = []
    for fn in truth_fns:
        true_c = np.array([fn(t)[0] for t in times])
        true_r = np.array([fn(t)[1] for t in times])
        # Nhiễu đo bán kính lấy bằng một nửa nhiễu đo vị trí.
        meas_c = true_c + rng.normal(0.0, meas_std, size=true_c.shape)
        meas_r = true_r + rng.normal(0.0, meas_std / 2.0, size=true_r.shape)
        fc, sc = kalman_cv_forecast(meas_c, times, K, dt, Q_POS, meas_std**2)
        fr, sr = kalman_cv_forecast(
            meas_r.reshape(-1, 1), times, K, dt, Q_RAD, (meas_std / 2.0) ** 2
        )
        rho = kappa * np.sqrt(sc**2 + sr**2)
        out.append(dict(centers=fc, radii=fr[:, 0], rho=rho))
    return out


def forecast_error(truth_fns, predicted: list, t0: float, K: int, dt: float) -> float:
    """Sai số dự báo tâm lớn nhất tại cuối chân trời:
    max_m ‖ô_mK(dự báo) − o_mK(thật)‖."""
    truth = sample_truth(truth_fns, t0, K, dt)
    return max(
        float(np.linalg.norm(predicted[m]["centers"][K] - truth[m][0][K]))
        for m in range(len(truth_fns))
    )
