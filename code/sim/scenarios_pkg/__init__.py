# -*- coding: utf-8 -*-
"""
Gói kịch bản benchmark ST-DCA (giao thức thực nghiệm 1.2, §2).

  - registry : sổ đăng ký thống nhất — get_scenario(name),
               all_names(), resolve_for_runner(name)
  - family_f : họ F (ellipsoid biến dạng F1–F5 + ô minh họa F1-MS)
  - validate : kiểm tra tính hợp lệ độc lập cấu hình (§2.5(i)(ii))

Các họ S / D / loạt A / loạt B tái dùng nguyên văn module của Bài 1
(sim/scenarios.py, sim/dynamic_scenarios.py,
sim/benchmark_scenarios.py) — sổ đăng ký chỉ bọc chúng vào một
giao diện chọn-theo-tên duy nhất.

(Cây package 2026-11-24: gói con sim.scenarios_pkg trong cây
refactor của người dùng — import tuyệt đối sim.*/core.*, không
còn bootstrap sys.path của cây phẳng cũ.)
"""
from .registry import all_names, get_scenario, resolve_for_runner  # noqa: E402

__all__ = ["all_names", "get_scenario", "resolve_for_runner"]
