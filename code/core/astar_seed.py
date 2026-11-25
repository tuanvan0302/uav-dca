# -*- coding: utf-8 -*-
"""
Điểm khởi tạo cho bộ giải DCA bằng tìm đường A* trên lưới ba chiều.

Khi không được cấp điểm khởi tạo, core.dca_planner.plan dùng đường thẳng
từ vị trí hiện tại tới đích. Nếu vật cản chắn giữa, đường thẳng này đi
xuyên vật cản; các mặt cắt tuyến tính hoá tại những điểm đó có thể giữ
quá trình lặp ở phía vòng sai của vật cản (ví dụ đi bên dưới), làm biến dư
không về 0. Module này sinh một đường đã vòng qua vật cản ngay từ đầu.

Cách làm:
  - Lưới bước h (mặc định 0,25 m), neo tại p0. Một nút bị chặn nếu tâm nút
    nằm trong bán kính r + d_safe + extra_m của một vật cản, hoặc thấp hơn
    cao độ sàn. Nút đầu (vị trí thật của UAV) và nút đích không bị chặn.
    Mặc định extra_m = 0,25 m ≥ (√3/2)·h ≈ 0,217 m: mọi điểm trên đoạn nối
    hai nút kề cách một đầu mút không quá (√3/2)·h, nên đoạn nối hai nút tự
    do không vi phạm biên r + d_safe.
  - A* với 26 nút kề, chi phí là độ dài Euclid, heuristic là khoảng cách
    thẳng tới đích.
  - Rút gọn đường gấp khúc theo kiểu tham lam: từ mỗi điểm, nhảy tới điểm
    xa nhất mà đoạn thẳng nối tới nó sạch. Đoạn được kiểm tra bằng lấy mẫu
    dọc đoạn (bước ≤ 0,1 m).
  - Lấy mẫu lại đường đã rút gọn đều theo độ dài cung thành đúng n_points
    điểm (K+1 nút của chân trời).
  - Không tìm được đường thì trả None; khi đó plan() dùng đường thẳng.

Tuỳ chọn:
  - extra_margin là một số (dùng cho mọi vật cản) hoặc danh sách theo từng
    vật cản, song song với obstacles.
  - segment_use_need=True: bước rút gọn kiểm tra đoạn với bán kính
    r + d_safe + extra_m + segment_guard thay vì r + d_safe. Hàm khe hở
    ‖q − o‖ là 1-Lipschitz và hai mẫu liền nhau cách nhau ≤ 0,1 m, nên mọi
    điểm trên đoạn cách mẫu gần nhất ≤ 0,05 m; vì vậy segment_guard = 0,05
    bảo đảm cả đoạn (không chỉ các mẫu) thoả r + d_safe + extra_m.
    segment_extras (tuỳ chọn) cho phép dùng biên kiểm tra đoạn khác với
    biên chặn nút.
  - Giới hạn tính toán: max_nodes nút được mở rộng hoặc time_budget_s giây,
    điều kiện nào tới trước; vượt giới hạn thì trả None. info_out (dict tuỳ
    chọn) nhận nodes_explored, budget_exceeded, elapsed_s.
"""

from __future__ import annotations

import heapq
import time

import numpy as np


def is_segment_clean(
    a: np.ndarray,
    b: np.ndarray,
    obstacles,
    d_safe: float,
    z_floor: float,
    check_floor: bool = True,
    extras=None,
    guard: float = 0.0,
) -> bool:
    """Đoạn thẳng a->b có sạch không, kiểm tra bằng lấy mẫu đều với bước
    ≤ 0,1 m: mọi mẫu phải nằm ngoài mọi hình cầu bán kính
    r + d_safe + extra_m + guard và (nếu check_floor) không thấp hơn sàn.
    extras=None nghĩa là extra_m = 0 với mọi vật cản."""
    # Số mẫu: đủ để hai mẫu liền nhau cách nhau ≤ 0,1 m, tối thiểu 2 (hai đầu).
    length = float(np.linalg.norm(b - a))
    n = max(2, int(np.ceil(length / 0.1)) + 1)
    for i in range(n):
        q = a + (b - a) * (i / (n - 1))
        if check_floor and q[2] < z_floor - 1e-9:
            return False
        for m, (o, r) in enumerate(obstacles):
            extra = 0.0 if extras is None else float(extras[m])
            if np.linalg.norm(q - o) < r + d_safe + extra + guard - 1e-9:
                return False
    return True


def astar_seed(
    p0,
    goal,
    obstacles,
    n_points: int,
    d_safe: float = 0.5,
    z_floor: float = 0.8,
    grid_step: float = 0.25,
    extra_margin: float = 0.25,
    pad: float = 2.5,
    segment_use_need: bool = False,
    segment_guard: float = 0.0,
    segment_extras=None,
    max_nodes: int | None = None,
    time_budget_s: float | None = None,
    info_out: dict | None = None,
) -> np.ndarray | None:
    """Sinh điểm khởi tạo (n_points, 3) đi từ p0 tới goal, vòng qua vật cản.

    obstacles: danh sách (tâm (3,), bán kính) tại thời điểm hiện tại.
    extra_margin: một số, hoặc danh sách theo từng vật cản.
    pad: phần nới rộng (m) của hộp lưới quanh p0 và goal.
    segment_use_need / segment_guard / segment_extras: xem docstring module.
    max_nodes / time_budget_s: giới hạn tính toán; vượt thì trả None và
    info_out["budget_exceeded"] = True.

    Trả None nếu không tìm được đường.
    """
    t_start = time.perf_counter()

    def _fill_info(nodes: int, exceeded: bool):
        # Ghi thống kê vào info_out (nếu có) trước mỗi lần return.
        if info_out is not None:
            info_out.update(
                nodes_explored=int(nodes),
                budget_exceeded=bool(exceeded),
                elapsed_s=float(time.perf_counter() - t_start),
            )

    p0 = np.asarray(p0, dtype=float)
    goal = np.asarray(goal, dtype=float)
    obstacles = [(np.asarray(o, dtype=float), float(r)) for (o, r) in obstacles]
    # Đưa extra_margin về danh sách theo từng vật cản.
    if np.isscalar(extra_margin):
        extras = [float(extra_margin)] * len(obstacles)
    else:
        extras = [float(e) for e in extra_margin]
        if len(extras) != len(obstacles):
            raise ValueError(
                f"per-obstacle extra_margin must match obstacles: "
                f"{len(extras)} != {len(obstacles)}"
            )
    if not obstacles or n_points < 2:
        _fill_info(0, False)
        return None
    h = float(grid_step)

    # ---- Lưới: hộp bao p0 và goal nới thêm pad, neo tại p0 để p0 trùng
    # một nút lưới.
    lo = np.minimum(p0, goal) - pad
    hi = np.maximum(p0, goal) + pad
    for (o, r), extra_m in zip(obstacles, extras):
        # Nâng trần hộp đủ cao để có thể vòng qua phía trên vật cản.
        hi[2] = max(hi[2], o[2] + r + d_safe + extra_m + pad)
    lo[2] = min(lo[2], z_floor)
    axes = []
    for ax in range(3):
        i_min = int(np.floor((lo[ax] - p0[ax]) / h))
        i_max = int(np.ceil((hi[ax] - p0[ax]) / h))
        axes.append(p0[ax] + h * np.arange(i_min, i_max + 1))
    # Toạ độ các nút lưới theo từng trục; nút được đánh chỉ số (i, j, k).
    xs, ys, zs = axes

    def node_pos(idx) -> np.ndarray:
        return np.array([xs[idx[0]], ys[idx[1]], zs[idx[2]]])

    # Nút đầu và nút đích: nút lưới gần p0 và goal nhất.
    start = (
        int(np.argmin(np.abs(xs - p0[0]))),
        int(np.argmin(np.abs(ys - p0[1]))),
        int(np.argmin(np.abs(zs - p0[2]))),
    )
    goal_idx = (
        int(np.argmin(np.abs(xs - goal[0]))),
        int(np.argmin(np.abs(ys - goal[1]))),
        int(np.argmin(np.abs(zs - goal[2]))),
    )
    goal_pos = node_pos(goal_idx)

    blocked_cache: dict = {}

    def blocked(idx) -> bool:
        # Nút bị chặn nếu thấp hơn sàn hoặc nằm trong vùng cấm của một vật
        # cản. Kết quả được lưu đệm theo chỉ số nút.
        if idx == start or idx == goal_idx:
            return False
        if idx in blocked_cache:
            return blocked_cache[idx]
        q = node_pos(idx)
        b = q[2] < z_floor - 1e-9
        if not b:
            for (o, r), extra_m in zip(obstacles, extras):
                if np.linalg.norm(q - o) < r + d_safe + extra_m - 1e-9:
                    b = True
                    break
        blocked_cache[idx] = b
        return b

    # ---- A* trên lưới, 26 nút kề. g_score: chi phí tốt nhất đã biết từ
    # start; came: nút cha để dựng lại đường; open_heap: (f = g + h, nút).
    dims = (len(xs), len(ys), len(zs))
    g_score = {start: 0.0}
    came: dict = {}
    open_heap = [(float(np.linalg.norm(node_pos(start) - goal_pos)), start)]
    closed = set()
    found = False
    n_expanded = 0
    budget_exceeded = False
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        # Một nút có thể nằm trong heap nhiều lần; chỉ xử lý lần đầu lấy ra.
        if cur in closed:
            continue
        closed.add(cur)
        n_expanded += 1
        if cur == goal_idx:
            found = True
            break
        # Kiểm tra giới hạn tính toán. Đồng hồ chỉ được đọc mỗi 128 nút để
        # giảm chi phí.
        if max_nodes is not None and n_expanded >= max_nodes:
            budget_exceeded = True
            break
        if (
            time_budget_s is not None
            and n_expanded % 128 == 0
            and time.perf_counter() - t_start > time_budget_s
        ):
            budget_exceeded = True
            break
        cur_pos = node_pos(cur)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == dj == dk == 0:
                        continue
                    nb = (cur[0] + di, cur[1] + dj, cur[2] + dk)
                    if not (
                        0 <= nb[0] < dims[0]
                        and 0 <= nb[1] < dims[1]
                        and 0 <= nb[2] < dims[2]
                    ):
                        continue
                    if nb in closed or blocked(nb):
                        continue
                    ng = g_score[cur] + float(np.linalg.norm(node_pos(nb) - cur_pos))
                    if ng < g_score.get(nb, np.inf):
                        g_score[nb] = ng
                        came[nb] = cur
                        f = ng + float(np.linalg.norm(node_pos(nb) - goal_pos))
                        heapq.heappush(open_heap, (f, nb))
    if not found:
        _fill_info(n_expanded, budget_exceeded)
        return None

    # ---- Dựng lại đường gấp khúc từ came. Hai đầu thay bằng p0 và goal thật
    # (thay vì nút lưới gần nhất).
    idx_path = [goal_idx]
    while idx_path[-1] != start:
        idx_path.append(came[idx_path[-1]])
    idx_path.reverse()
    path = [p0] + [node_pos(i) for i in idx_path[1:-1]] + [goal]

    # ---- Rút gọn tham lam. Chọn biên dùng để kiểm tra đoạn: mặc định
    # r + d_safe; với segment_use_need thì r + d_safe + extra_m + guard.
    if segment_use_need:
        seg_extras = (
            extras if segment_extras is None else [float(e) for e in segment_extras]
        )
        if len(seg_extras) != len(obstacles):
            raise ValueError(
                f"segment_extras must match obstacles: "
                f"{len(seg_extras)} != {len(obstacles)}"
            )
        seg_guard = float(segment_guard)
    else:
        seg_extras, seg_guard = None, 0.0

    def _seg_ok(a, b, check_floor):
        return is_segment_clean(
            a,
            b,
            obstacles,
            d_safe,
            z_floor,
            check_floor=check_floor,
            extras=seg_extras,
            guard=seg_guard,
        )

    reduced = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        # Tìm j lớn nhất (từ cuối đường lùi dần) để đoạn path[i] -> path[j]
        # sạch. Nếu UAV đang ở dưới sàn (p0[2] < z_floor), các đoạn xuất
        # phát từ p0 không bị kiểm tra sàn, chỉ kiểm tra vật cản.
        while j > i + 1 and not _seg_ok(
            path[i], path[j], not (i == 0 and p0[2] < z_floor)
        ):
            j -= 1
        # Ngay cả đoạn tới điểm kề cũng không sạch: bỏ cuộc.
        if not _seg_ok(path[i], path[j], not (i == 0 and p0[2] < z_floor)):
            _fill_info(n_expanded, budget_exceeded)
            return None
        reduced.append(path[j])
        i = j

    # ---- Lấy mẫu lại n_points điểm cách đều theo độ dài cung: cum là độ
    # dài cung tích luỹ tại các đỉnh, nội suy tuyến tính từng toạ độ.
    pts = np.array(reduced)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(seg))
    if total <= 0:
        _fill_info(n_expanded, budget_exceeded)
        return None
    targets = np.linspace(0.0, total, n_points)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = np.empty((n_points, 3))
    for c in range(3):
        out[:, c] = np.interp(targets, cum, pts[:, c])
    # Gán lại hai đầu để loại sai số làm tròn của phép nội suy.
    out[0] = p0
    out[-1] = goal
    _fill_info(n_expanded, budget_exceeded)
    return out
