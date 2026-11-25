# -*- coding: utf-8 -*-
"""Baselines Tier-1 để so sánh với ST-DCA (SPEC v1.1).

Ba module dùng chung một giao diện plan() khớp dca_planner.plan:

    plan(p0, v0, goal, obstacles, prm, seed_P=None) -> BaselineResult

  - ngo2023 : penalty-DCA của Ngo et al. (CDC 2023) — SPEC-BL-NGO-2023
  - cbf2021 : Convexified-CBF của Varun et al. (ICC 2021) — SPEC-BL-CBF-2021
  - mpcqp   : MPC-QP tuyến tính hóa một lần (mốc tham chiếu) — SPEC-BL-MPCQP

Mọi phương pháp tuân thủ SPEC-FAIR-EXP4 (SPEC_cong_bang): cùng mô
hình, cùng mảng dự báo đóng băng, cùng J, cùng dung sai solver tường
minh. Xem README_BASELINES.md trong thư mục này.

METHODS ánh xạ tên phương pháp (giá trị của --method) sang
(module, config_id).
"""

from __future__ import annotations

from baselines import cbf2021, mpcqp, ngo2023

METHODS = {
    "Ngo2023": (ngo2023, "SPEC-BL-NGO-2023 v1.1 main"),
    "CBF2021": (cbf2021, "SPEC-BL-CBF-2021 v1.1 main"),
    "MPCQP": (mpcqp, "SPEC-BL-MPCQP v1.1 main"),
}

__all__ = ["ngo2023", "cbf2021", "mpcqp", "METHODS"]
