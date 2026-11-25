# -*- coding: utf-8 -*-
"""
Kiểm thử điểm khởi tạo A* (DV-H8, Quyết định 4 §4.2/§4.5) cho
core/astar_seed.py và core/method_adapter.py:

  - astar_seed: extra theo từng vật (danh sách toàn trị bằng vô
    hướng -> trùng bit); chặn nút tôn trọng need của vật có extra
    lớn; is_segment_clean dùng need + dải bảo vệ 0,05 khi được yêu
    cầu, giữ kiểm tra không need và không dải bảo vệ khi không;
    ngân sách kép (nút / wall-clock)
    vượt -> None + budget_exceeded trong info_out.
  - adapter: chu kỳ 0 kích hoạt astar_L1 với seed_max_viol <=
    1e-9 trên kịch bản S1; chu
    kỳ sau thành công giữ warm/straight với t_seed = 0; chu kỳ
    sau sập cứng (biến dư = inf) kích hoạt post_crash_or_slack;
    không bản ghi nào mang seed_trigger = new_obstacle (kích
    hoạt chết, ĐK4.6).

Chạy:  python3 -m tests.test_astar_seed   (từ gốc code/)
       hoặc pytest tests/test_astar_seed.py
"""

import numpy as np

from core.astar_seed import astar_seed, is_segment_clean
from core.dca_planner import PlannerParams, plan
from core.method_adapter import MethodAdapter

P0 = np.array([0.0, 0.0, 1.5])
GOAL = np.array([10.0, 0.0, 1.5])
CENTER = np.array([5.0, 0.0, 1.5])


# ---- astar_seed: tương thích ngược + extra theo từng vật ----
def test_extras_list_equals_scalar():
    obs = [(CENTER.copy(), 1.2), (np.array([5.0, 3.0, 1.5]), 0.8)]
    a = astar_seed(P0, GOAL, obs, 29, extra_margin=0.25)
    b = astar_seed(P0, GOAL, obs, 29, extra_margin=[0.25, 0.25])
    assert a is not None and np.array_equal(a, b)


def test_per_obstacle_need_respected():
    # Vật 1 extra lớn (need = 1,2 + 0,5 + 2,0 = 3,7); vật 2 extra 0.
    # Chặn nút phồng thêm guard + dip như adapter DV-H8 xây L1;
    # kiểm tra đoạn ở need thuần + guard qua segment_extras.
    from core import method_adapter as ma

    obs = [(CENTER.copy(), 1.2), (np.array([5.0, 3.0, 1.5]), 0.8)]
    node_extras = [
        2.0 + ma.SEED_SEGMENT_GUARD_M + ma.SEED_NODE_DIP_M,
        0.0 + ma.SEED_SEGMENT_GUARD_M + ma.SEED_NODE_DIP_M,
    ]
    seed = astar_seed(
        P0,
        GOAL,
        obs,
        29,
        extra_margin=node_extras,
        segment_use_need=True,
        segment_guard=0.05,
        segment_extras=[2.0, 0.0],
    )
    assert seed is not None
    d1 = np.linalg.norm(seed - CENTER, axis=1)
    assert float(np.min(d1)) >= 3.7 - 1e-9, float(np.min(d1))


def test_segment_clean_need_and_guard():
    o = np.array([0.0, 0.0, 1.5])
    # Đoạn thẳng song song trục x, khoảng cách gần nhất tới tâm là
    # 1,78: qua legacy (r + d = 1,7) nhưng hụt need + guard
    # = 1,2 + 0,5 + 0,1 + 0,05 = 1,85.
    a = np.array([-3.0, 1.78, 1.5])
    b = np.array([3.0, 1.78, 1.5])
    assert is_segment_clean(a, b, [(o, 1.2)], 0.5, 0.8)
    assert not is_segment_clean(a, b, [(o, 1.2)], 0.5, 0.8, extras=[0.1], guard=0.05)
    # Khoảng cách gần nhất 1,90 > 1,85: qua cả need + guard.
    a2 = np.array([-3.0, 1.90, 1.5])
    b2 = np.array([3.0, 1.90, 1.5])
    assert is_segment_clean(a2, b2, [(o, 1.2)], 0.5, 0.8, extras=[0.1], guard=0.05)


def test_budget_nodes_and_time():
    info = {}
    s = astar_seed(
        P0,
        GOAL,
        [(CENTER.copy(), 1.2)],
        29,
        extra_margin=2.66,
        max_nodes=50,
        info_out=info,
    )
    assert s is None and info["budget_exceeded"] is True
    assert info["nodes_explored"] <= 50
    info2 = {}
    s2 = astar_seed(
        P0,
        GOAL,
        [(CENTER.copy(), 1.2)],
        29,
        extra_margin=2.66,
        time_budget_s=0.0,
        info_out=info2,
    )
    assert s2 is None and info2["budget_exceeded"] is True
    info3 = {}
    # Trần thời gian để rộng (10 s) cho ca này: mục tiêu là "đủ
    # ngân sách -> thành công, budget_exceeded = False"; trên VM
    # tranh chấp lần gọi này dao động 0,65-1,7 s nên ngân sách
    # 1,0 s sẽ làm test chập chờn — hành vi CHẶN ngân sách đã
    # được kiểm bởi hai ca trên (nút + thời gian = 0).
    s3 = astar_seed(
        P0,
        GOAL,
        [(CENTER.copy(), 1.2)],
        29,
        extra_margin=2.66,
        max_nodes=100_000,
        time_budget_s=10.0,
        info_out=info3,
    )
    assert s3 is not None and info3["budget_exceeded"] is False


# ---- adapter: kích hoạt + thang + bản ghi ----
class _FakeRes:
    def __init__(self, slack):
        self.total_slack = slack
        self.iterations = 1
        self.solve_time = 0.01


def _stub_adapter(method="A1"):
    ad = MethodAdapter(method, eps=0.05)
    return ad


def test_trigger_cycle0_and_warm_passthrough():
    ad = _stub_adapter("A2")
    calls = []

    def stub(p0, v0, goal, obstacles, prm, seed_P=None):
        calls.append(seed_P)
        return _FakeRes(0.0)

    wrapped = ad.wrap_plan(stub)
    prm = PlannerParams()
    obs = [(CENTER.copy(), 1.2)]
    wrapped(P0, np.zeros(3), GOAL, obs, prm, seed_P=None)
    rec0 = ad.last_seed_info
    assert rec0["seed_trigger"] == "cycle0"
    assert rec0["seed_source"] == "astar_L1"
    assert rec0["seed_max_viol"] <= 1e-9
    assert calls[0] is not None  # hạt giống A* đã thay seed thẳng
    # Chu kỳ sau thành công: giữ warm truyền vào, t_seed = 0.
    warm = np.linspace(0.0, 1.0, prm.K + 1)[:, None] * GOAL
    wrapped(P0, np.zeros(3), GOAL, obs, prm, seed_P=warm)
    rec1 = ad.last_seed_info
    assert rec1["seed_trigger"] == "none"
    assert rec1["seed_source"] == "warm" and rec1["t_seed"] == 0.0
    assert calls[1] is warm


def test_trigger_post_crash():
    ad = _stub_adapter("A2")
    seq = iter([_FakeRes(np.inf), _FakeRes(0.0), _FakeRes(0.5), _FakeRes(0.0)])

    def stub(p0, v0, goal, obstacles, prm, seed_P=None):
        return next(seq)

    wrapped = ad.wrap_plan(stub)
    prm = PlannerParams()
    obs = [(CENTER.copy(), 1.2)]
    wrapped(P0, np.zeros(3), GOAL, obs, prm, seed_P=None)  # cycle0
    wrapped(P0, np.zeros(3), GOAL, obs, prm, seed_P=None)  # sau sập inf
    assert ad.last_seed_info["seed_trigger"] == "post_crash_or_slack"
    wrapped(P0, np.zeros(3), GOAL, obs, prm, seed_P=None)  # prev slack 0
    assert ad.last_seed_info["seed_trigger"] == "none"
    wrapped(P0, np.zeros(3), GOAL, obs, prm, seed_P=None)  # prev slack 0,5
    assert ad.last_seed_info["seed_trigger"] == "post_crash_or_slack"
    assert all(
        r["seed_trigger"] != "new_obstacle" for r in ad.summary()["seed_records"]
    )


def test_a1_s1_real_plan_l1():
    """Ô 1 chân S1 ở mức adapter: lần gọi đầu trên vật tĩnh S1
    phải dùng astar_L1, seed_max_viol <= 1e-9 và plan() thành
    công (biến dư <= 1e-2). Ngân sách wall-clock của A* được
    nới riêng cho test này (SEED_TIME_BUDGET_S = None) vì nó đã
    được kiểm ở tầng astar (test_budget_nodes_and_time) và trên
    VM tranh chấp t_seed dao động quanh ngân sách 1,0 s — test
    này nhắm vào CƠ CHẾ L1 (xây hạt giống + viol + plan), còn
    tương tác ngân sách đo ở ô kiểm chứng Ô 1 bằng adapter
    nguyên bản."""
    from core import method_adapter as ma

    old = ma.SEED_TIME_BUDGET_S
    ma.SEED_TIME_BUDGET_S = None
    try:
        ad = MethodAdapter("A1", eps=0.05)
        wrapped = ad.wrap_plan(plan)
        prm = PlannerParams(extra_margin=0.0)
        res = wrapped(P0, np.zeros(3), GOAL, [(CENTER.copy(), 1.2)], prm, seed_P=None)
    finally:
        ma.SEED_TIME_BUDGET_S = old
    rec = ad.last_seed_info
    assert rec["seed_source"] == "astar_L1", rec
    assert rec["seed_max_viol"] <= 1e-9, rec
    assert res.success and res.total_slack <= 1e-2, (res.iterations, res.total_slack)


if __name__ == "__main__":
    tests = [
        test_extras_list_equals_scalar,
        test_per_obstacle_need_respected,
        test_segment_clean_need_and_guard,
        test_budget_nodes_and_time,
        test_trigger_cycle0_and_warm_passthrough,
        test_trigger_post_crash,
        test_a1_s1_real_plan_l1,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
