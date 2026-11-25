# -*- coding: utf-8 -*-
"""
Chứng minh tích hợp runner: chạy thử 1 episode A0 ở chế độ ĐÓNG BĂNG
THỜI GIAN (time_model='frozen' — đường chạy gốc của Bài 1) trên một
kịch bản loạt A và một kịch bản họ F, chọn kịch bản THEO TÊN qua sổ
đăng ký thống nhất của gói này.

Cơ chế: sim_realtime.run_episode hiện nạp kịch bản qua
dynamic_scenarios.dynamic_scenario (chỉ có họ D). Script này thay
hàm đó bằng scenarios_pkg.registry.resolve_for_runner — ĐÚNG một điểm
nối, không sửa file nào của hạ tầng. Khi hạ tầng sẵn sàng, việc tích
hợp chính thức chỉ là cho run_episode dùng resolve_for_runner thay
dynamic_scenario (xem README.md của gói).

Seed chạy thử thuộc dải riêng 9_000_000 + i (§3.3 giao thức) — kết
quả KHÔNG bao giờ vào bảng kết quả chính.

Chạy (từ thư mục exp/code):
    python3 -m scenarios_pkg.runner_smoke
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim import sim_realtime
from core.dca_planner import PlannerParams

from .registry import resolve_for_runner

# điểm nối duy nhất: run_episode tra kịch bản qua hàm này
sim_realtime.dynamic_scenario = resolve_for_runner

CASES = [
    # (kịch bản, chế độ thông tin, seed chạy thử)
    ("A_L1_r0", "truth", 9000000),        # loạt A: trường tĩnh, truth (§2.3)
    ("F1_thanh_dai", "truth", 9000001),   # họ F: A0 qua cầu-tương-đương (§2.7)
]


def ellipsoid_clearance_of_cycles(entry, cycles) -> None:
    """Chỗ giữ: khe hở ellipsoid thật của episode họ F cần quỹ đạo bay
    đầy đủ; log chu kỳ của sim hiện không ghi vị trí từng bước nên
    không tính lại được ở đây. Khe hở trong metrics của lần chạy này
    là khe hở so với cầu thể-tích-tương-đương (§2.7) — đúng giao diện
    A0/A1 định nghĩa trên họ F. Đánh giá khe hở ellipsoid thật thuộc
    đường đánh giá có tầng hình dạng (A2/A3); hàm khoảng cách dùng
    chung nằm ở scenarios_pkg.validate.ellipsoid_signed_distance."""
    return None


def main() -> int:
    outdir = Path(__file__).parent / "smoke_results"
    outdir.mkdir(exist_ok=True)
    prm = PlannerParams()  # A0: tham số gốc của Bài 1, không chỉnh
    summary = []
    for name, mode, seed in CASES:
        out = sim_realtime.run_episode(name, mode=mode, seed=seed,
                                       prm=prm, time_model="frozen")
        m = out["metrics"]
        rec = dict(scenario=name, mode=mode, seed=seed,
                   time_model="frozen", method="A0",
                   status=m["status"], success=m["success"],
                   safe_success=m["safe_success"], collision=m["collision"],
                   min_clearance_continuous=m["min_clearance_continuous"],
                   path_length_m=m["path_length_m"],
                   n_cycles=m["n_cycles"],
                   n_planner_failures=m["n_planner_failures"])
        if name.startswith("F"):
            rec["clearance_reference"] = ("cầu thể-tích-tương-đương "
                                           "r_eq (§2.7) — giao diện A0")
        summary.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    (outdir / "runner_smoke.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {outdir / 'runner_smoke.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
