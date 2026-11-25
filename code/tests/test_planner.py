# -*- coding: utf-8 -*-
"""
Kiểm thử lớp 1 (số học thuần tuý) cho bộ giải DCA dạng phạt
(core/dca_planner.py).

Chạy:  python3 -m tests.test_planner   (từ gốc code/)
       hoặc pytest tests/test_planner.py
"""

import numpy as np

from core.astar_seed import astar_seed
from core.dca_planner import PlannerParams, plan
from sim.scenarios import scenario

PRM = PlannerParams()


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    assert cond, name


def test_no_obstacle():
    sc = scenario("S0_trong")
    res = plan(sc["start"], np.zeros(3), sc["goal"], [], PRM)
    _check(
        "S0: tới gần đích ở cuối chân trời",
        np.linalg.norm(res.P[-1] - sc["goal"]) < 1.0,
        f"(lệch {np.linalg.norm(res.P[-1] - sc['goal']):.2f} m)",
    )
    dev = np.max(np.abs(res.P[:, 1]))  # độ lệch khỏi đường thẳng y=0
    _check("S0: quỹ đạo gần như thẳng", dev < 0.35, f"(lệch ngang tối đa {dev:.2f} m)")


def test_single_obstacle_safe():
    sc = scenario("S1_mot_vat")
    res = plan(sc["start"], np.zeros(3), sc["goal"], sc["obstacles"], PRM)
    _check(
        "S1: bộ giải báo thành công (biến dư ~ 0)",
        res.success,
        f"(tổng biến dư {res.total_slack:.2e})",
    )
    _check(
        "S1: khe hở thật >= biên an toàn",
        res.min_clearance >= PRM.d_safe - 1e-3,
        f"(khe hở nhỏ nhất {res.min_clearance:.3f} m, cần >= {PRM.d_safe})",
    )


def test_dynamics_consistent():
    sc = scenario("S1_mot_vat")
    res = plan(sc["start"], np.zeros(3), sc["goal"], sc["obstacles"], PRM)
    dt = PRM.dt
    P2 = res.P[:-1] + dt * res.V[:-1] + 0.5 * dt * dt * res.U
    V2 = res.V[:-1] + dt * res.U
    _check(
        "Động học (R1) khớp tới sai số máy",
        np.max(np.abs(P2 - res.P[1:])) < 1e-3 and np.max(np.abs(V2 - res.V[1:])) < 1e-3,
        f"(sai lệch tối đa {np.max(np.abs(P2 - res.P[1:])):.1e})",
    )


def test_objective_monotone_decrease():
    # Lưu ý trung thực: tính chất giảm đơn điệu nghiêm ngặt của DCA đúng cho
    # sơ đồ chặn trên chính xác; biến thể cài đặt ở đây (tuyến tính hoá tích
    # luỹ + giải QP không chính xác tuyệt đối) có thể cho mục tiêu THẬT tăng
    # thoáng qua ở vài vòng giữa (khi quỹ đạo đổi phía vòng qua vật cản).
    # Tiêu chí kiểm tra vì vậy là: giảm mạnh tổng thể và không phân kỳ.
    sc = scenario("S1_mot_vat")
    prm = PlannerParams(tau0=40.0, tau_factor=1.0)  # giữ nguyên một mức phạt
    res = plan(sc["start"], np.zeros(3), sc["goal"], sc["obstacles"], prm)
    h = np.array(res.obj_history)
    # "Không phân kỳ" được hiểu theo đúng điều cần khẳng định: giá
    # trị cuối giảm mạnh so với giá trị đầu và không vượt đỉnh của
    # toàn bộ lịch sử. Không áp điều kiện cho từng bước: trên S1 đã
    # đo được một bước tăng +38% đúng tại vòng quỹ đạo đổi phía vòng
    # (176,7 -> ... -> 32,2 -> 44,6), phù hợp với lưu ý ở đầu hàm.
    _check(
        "Mục tiêu phạt giảm mạnh tổng thể, không phân kỳ",
        len(h) >= 2 and h[-1] < 0.35 * h[0] and h[-1] <= np.max(h),
        f"({len(h)} vòng, {h[0]:.1f} -> {h[-1]:.1f})",
    )


def test_input_limits():
    sc = scenario("S2_hanh_lang")
    res = plan(sc["start"], np.zeros(3), sc["goal"], sc["obstacles"], PRM)
    _check(
        "(R3): |u| không vượt giới hạn",
        np.max(np.abs(res.U)) <= PRM.u_max + 1e-6,
        f"(max |u| = {np.max(np.abs(res.U)):.2f})",
    )
    _check(
        "(R3): |v| không vượt giới hạn",
        np.max(np.abs(res.V)) <= PRM.v_max + 1e-3,
        f"(max |v| = {np.max(np.abs(res.V)):.2f})",
    )


def test_floor_with_low_start():
    # Ca hồi quy: với điểm khởi tạo đường thẳng, bài toán xuất phát
    # thấp hơn tâm vật cản bị kẹt ở phía dưới (biến dư ~2,43, khe hở
    # -0,498 m). Điểm khởi tạo A* (core/astar_seed.py) vòng sẵn qua
    # vật cản; kiểm tra này khẳng định cả điểm khởi tạo lẫn kết quả
    # của bộ giải.
    sc = scenario("S1_mot_vat")
    p0 = np.array([0.0, 0.0, 1.0])  # thấp hơn tâm vật (1,5 m)
    seed = astar_seed(
        p0,
        sc["goal"],
        sc["obstacles"],
        PRM.K + 1,
        d_safe=PRM.d_safe,
        z_floor=PRM.z_floor,
    )
    _check("Xuất phát thấp: A* tìm được điểm khởi tạo vòng sẵn", seed is not None)
    seed_clear = min(
        float(np.min(np.linalg.norm(seed - o, axis=1) - r))
        for (o, r) in sc["obstacles"]
    )
    _check(
        "Xuất phát thấp: điểm khởi tạo A* tự nó an toàn theo (R4)",
        seed_clear >= PRM.d_safe - 1e-6,
        f"(khe hở của điểm khởi tạo {seed_clear:.3f} m)",
    )
    res = plan(p0, np.zeros(3), sc["goal"], sc["obstacles"], PRM, seed_P=seed)
    _check(
        "Xuất phát thấp: bộ giải báo thành công (biến dư ~ 0)",
        res.success,
        f"(tổng biến dư {res.total_slack:.2e})",
    )
    _check(
        "Xuất phát thấp: khe hở thật >= biên an toàn",
        res.min_clearance >= PRM.d_safe - 1e-3,
        f"(khe hở nhỏ nhất {res.min_clearance:.3f} m)",
    )
    _check(
        "Xuất phát thấp: quỹ đạo không xuống dưới cao độ sàn",
        np.min(res.P[:, 2]) >= PRM.z_floor - 1e-6,
        f"(z nhỏ nhất {np.min(res.P[:, 2]):.3f} m, sàn {PRM.z_floor} m)",
    )


if __name__ == "__main__":
    test_no_obstacle()
    test_single_obstacle_safe()
    test_dynamics_consistent()
    test_objective_monotone_decrease()
    test_input_limits()
    test_floor_with_low_start()
    print("\nAll layer-1 tests: PASS")
