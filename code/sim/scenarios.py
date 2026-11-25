# -*- coding: utf-8 -*-
"""Các kịch bản vật cản TĨNH dùng cho kiểm thử (tương ứng T1–T3 của Giai đoạn A/B)."""
import numpy as np


def scenario(name: str) -> dict:
    """Trả về dict: start, goal, obstacles = [(tâm (3,), bán kính), ...]."""
    if name == "S0_trong":
        # Không vật cản: nghiệm đúng phải là đường gần thẳng tới đích.
        return dict(start=np.array([0.0, 0.0, 1.5]), goal=np.array([10.0, 0.0, 1.5]),
                    obstacles=[])
    if name == "S1_mot_vat":
        # Một hình cầu chắn giữa đường thẳng đầu–đích.
        return dict(start=np.array([0.0, 0.0, 1.5]), goal=np.array([10.0, 0.0, 1.5]),
                    obstacles=[(np.array([5.0, 0.0, 1.5]), 1.2)])
    if name == "S2_hanh_lang":
        # Hành lang: hai hàng cầu so le, phải luồn qua.
        obs = []
        for x in (3.0, 6.0, 9.0):
            obs.append((np.array([x, 1.8, 1.5]), 1.1))
            obs.append((np.array([x + 1.5, -1.8, 1.5]), 1.1))
        return dict(start=np.array([0.0, 0.0, 1.5]), goal=np.array([12.0, 0.0, 1.5]),
                    obstacles=obs)
    if name == "S3_loi_hep":
        # Lối hẹp giữa hai cầu gần chạm nhau (khe hở thật ~0.9 m, biên an toàn 0.5 m).
        return dict(start=np.array([0.0, 0.0, 1.5]), goal=np.array([10.0, 0.0, 1.5]),
                    obstacles=[(np.array([5.0, 1.45, 1.5]), 1.0),
                               (np.array([5.0, -1.45, 1.5]), 1.0)])
    if name == "S4_ngau_nhien":
        # Trường vật cản ngẫu nhiên có hạt giống cố định (tái lập được).
        rng = np.random.default_rng(7)
        start = np.array([0.0, 0.0, 1.5]); goal = np.array([14.0, 2.0, 1.5])
        obs = []
        tries = 0
        while len(obs) < 8 and tries < 500:
            tries += 1
            c = np.array([rng.uniform(2.5, 11.5), rng.uniform(-3.5, 5.5), 1.5])
            r = float(rng.uniform(0.7, 1.1))
            if np.linalg.norm(c - start) < r + 2.0 or np.linalg.norm(c - goal) < r + 2.0:
                continue
            if any(np.linalg.norm(c - o) < r + rr + 0.8 for (o, rr) in obs):
                continue
            obs.append((c, r))
        return dict(start=start, goal=goal, obstacles=obs)
    raise ValueError(name)


ALL = ["S0_trong", "S1_mot_vat", "S2_hanh_lang", "S3_loi_hep", "S4_ngau_nhien"]
