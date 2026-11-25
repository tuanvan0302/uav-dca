# -*- coding: utf-8 -*-
"""
method_adapter.py — chạy các cấu hình ablation A1/A2/A3 của ST-DCA bằng
cách bọc hàm plan() mà vòng lặp mô phỏng gọi.

Cách nối: sim_realtime.run_episode tự dựng danh sách vật cản mỗi chu kỳ
(dict cầu có centers/radii/rho, hoặc tuple (tâm, bán kính) ở mode frozen)
rồi gọi hàm `plan` trong namespace của nó. experiments/experiment_runner
thay hàm đó bằng
    sim_realtime.plan = adapter.wrap_plan(sim_realtime.plan).
Ở mỗi lời gọi, hàm đã bọc tạo danh sách vật cản MỚI theo cấu hình, chọn
điểm khởi tạo, rồi gọi planner thật. Danh sách vật cản gốc của vòng lặp
không bị sửa. A0 và các baseline không đi qua adapter.

Các cấu hình:
  A1 = A0 + biên (M): mảng rho của mỗi vật cản được thay bằng
           rho_planner[k] = ρ_traj(ε) + δ_trôi + δ_mẫu   với mọi k
       (core/safety_margin.compose_rho_traj_for_planner), trong đó ρ_traj(ε)
       tra từ bảng conformal theo nhóm tuổi,
       δ_trôi = v̂·T_r + ½·a_max·T_r² + e_bám (T_r lấy từ ReplanClock),
       δ_mẫu = (u_max + a_max)·Δt²/8.
       PlannerParams.extra_margin phải bằng 0 (runner đặt) vì toàn bộ biên
       đã nằm trong rho; d_safe giữ nguyên.
  A2 = A0 + hình dạng (G): mỗi vật cầu được đổi thành dict ellipsoid với
       bán trục (r_k, r_k, r_k) và yaw = 0. Cầu là trường hợp riêng của
       ellipsoid nên nhánh gauge của planner cho cùng ràng buộc với nhánh
       cầu. Biên giữ như A0 (extra_margin = 0,03, rho = κ·σ của bộ dự báo).
       Bán trục bị kẹp vào [A_ABS_MIN, A_ABS_MAX] = [0,02; 10] m vì bộ dự
       báo bán kính ngoại suy tuyến tính có thể cho bán kính âm ở cuối chân
       trời (ví dụ khi bán kính dao động sin), mà Ellipsoid không nhận bán
       trục ≤ 0. Số lần kẹp được đếm trong summary(). Ở các bước bị kẹp,
       kết quả A2 khác A0.
       Vòng lặp hiện chỉ có quan sát vật cản cầu, nên A2 đổi từ dự báo cầu;
       bộ dự báo hình dạng đầy đủ (shape_forecast.kalman_ellipsoid_obstacles)
       chưa được nối vào vòng lặp.
  A3 = ST-DCA đầy đủ: đổi sang ellipsoid như A2 và thay rho như A1. Planner
       cộng rho vào ngưỡng a_min + d của ràng buộc gauge (cùng đơn vị mét như
       nhánh cầu).

Bảng conformal: nạp bằng ConformalTable.load từ core/conformal_tables/
stdca_h1.{npz,json}.

Nhóm tuổi: bảng có ρ riêng cho mỗi nhóm tuổi, ứng với n_hist của bộ dự báo
(2 → age_new; 4, 8 → age_mid; 12 → age_old). Vòng lặp luôn gọi
kalman_obstacles với n_hist = 8 nên mặc định dùng age_mid cho mọi vật cản;
runner có thể ghi đè bằng --age-group.

a_max là cận trên gia tốc TÂM vật cản, dùng trong δ_trôi và δ_mẫu. Các
kịch bản D và các họ hiệu chuẩn đều có tâm chuyển động thẳng đều, nên mặc
định a_max = 0.

Điểm khởi tạo (seed) cho plan(), tính sau transform() từ đúng mảng rho mà
planner sẽ nhận:
  - need của điểm khởi tạo theo từng vật: need_m[k] = R_m[k] + d + rho_m[k],
    với R là bán kính (cầu) hoặc bán trục lớn nhất tại bước k (ellipsoid).
    Vật cản được đổi sang các cầu bao vùng quét (bounding_sphere_tuples),
    mỗi cầu được phồng thêm d + rho_m[k] qua extra_margin của astar_seed.
  - A* chỉ được chạy ở các chu kỳ kích hoạt:
      (a) chu kỳ đầu tiên;
      (b) chu kỳ ngay sau một lần plan() có tổng biến dư không hữu hạn hoặc
          > SEED_SLACK_TRIGGER.
    Ở các chu kỳ khác dùng seed_P do vòng lặp truyền vào (warm-start từ
    nghiệm trước), hoặc để plan() dùng đường thẳng nếu không có.
    seed_trigger có thể nhận "cycle0", "post_crash_or_slack" hoặc "none".
  - Thứ tự thử: L1 = A* theo need (nút chặn ở need + guard + dip, đoạn
    kiểm tra ở need + guard) → L2 = A* với phồng cố định r + d + 0,25 m →
    L3 = đường thẳng (seed_P = None).
  - Mỗi lần gọi A* bị giới hạn ở SEED_MAX_NODES nút hoặc SEED_TIME_BUDGET_S
    giây; vượt thì coi như không tìm được và chuyển xuống mức sau.
  - Mỗi chu kỳ sinh một bản ghi: seed_source, seed_trigger, t_seed,
    seed_max_viol (vi phạm need lớn nhất của điểm khởi tạo thực sự đưa vào
    plan()), seed_budget_exceeded, seed_nodes, cùng call, iterations,
    total_slack. Bản ghi nằm ở last_seed_info (của adapter và của hàm đã
    bọc) và trong summary()["seed_records"]. sim_realtime đọc
    last_seed_info sau mỗi lần gọi để cộng t_seed vào T_latency.
  - Đồng hồ ReplanClock được cập nhật bằng solve_time + t_seed, nên T_r của
    chu kỳ sau gồm cả thời gian tìm điểm khởi tạo.

Trần cửa sổ trễ T_cap (tham số t_cap, runner: --t-cap; mặc định None = không
chặn): với A1/A3, δ_trôi được tính trên min(T_r, T_cap) qua
ReplanClock.window_info() của safety_margin. Sau mỗi lần gọi, hàm đã bọc có
thuộc tính last_window_info = {T_r, T_r_eff, capped, T_cap}; sim_realtime
ghi bốn trường này vào bản ghi chu kỳ (kể cả khi không có trần, lúc đó
capped = False và T_r_eff = T_r). summary() có thêm window_records,
n_drift_capped và drift_capped_rate. A2 không có hạng tử trôi nên không có
window_info.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from core import conformal as _cf
from core import safety_margin as _mh
from core.astar_seed import astar_seed
from core.shape_forecast import A_ABS_MAX, A_ABS_MIN, bounding_sphere_tuples

_HERE = Path(__file__).resolve().parent
# Bảng conformal đã hiệu chuẩn.
TABLE_PREFIX = _HERE / "conformal_tables" / "stdca_h1"

# Các chuỗi mô tả dưới đây được ghi nguyên văn vào provenance của log
# (tiếng Anh, như mọi nội dung log).
SCORE_FORM_NOTE = (
    "sum of metric channels (protocol §4.2): s_k = e_pos,k + e_rad,k; "
    "e_yaw = 0 for spheres"
)
TABLE_SCOPE_NOTE = (
    "offline calibration, single source, sphere families (i)-(iii) of protocol §4.3"
)
AGE_RULE_NOTE = (
    "age group = group of the forecaster configuration in use "
    "(kalman_obstacles n_hist=8 -> age_mid via conformal.age_group_of); "
    "detection cycles are not counted; no policy is defined for "
    "age_new obstacles"
)

# ---- Hằng số cho điểm khởi tạo A* ----
SEED_MAX_NODES = 100_000  # số nút tối đa mỗi lần gọi A*
SEED_TIME_BUDGET_S = 1.0  # thời gian tối đa mỗi lần gọi A* (s)
SEED_SEGMENT_GUARD_M = 0.05  # dải bảo vệ khi kiểm tra đoạn (m), xem astar_seed
# Phần phồng thêm của nút so với mức kiểm tra đoạn: (√3/2)·h với h = 0,25 m
# là bước lưới mặc định của astar_seed. Mọi điểm trên đoạn nối hai nút kề
# cách một đầu mút không quá (√3/2)·h, nên khi nút tự do ở mức
# need + guard + dip thì đoạn nối hai nút tự do không vi phạm mức
# need + guard; nhờ vậy bước rút gọn đường luôn có ít nhất một đoạn sạch.
SEED_NODE_DIP_M = float(np.sqrt(3.0) / 2.0 * 0.25)
SEED_L2_EXTRA_M = 0.25  # phồng cố định ở mức L2 (= mặc định của astar_seed)
SEED_SLACK_TRIGGER = 1e-2  # tổng biến dư lớn hơn mức này thì chạy lại A*
SEED_RULE_NOTE = (
    "DV-H8: A* seed from the cycle's need (r + d + assembled rho per "
    "obstacle; ellipsoids via swept bounding spheres); triggered on "
    "cycle 0 and after a crash or total slack > 1e-2; ladder L1 -> L2 "
    "(fixed 0.25 m inflation) -> L3 (straight line); budget 100000 "
    "nodes or 1.0 s per call; T_latency includes t_seed"
)


def load_conformal_table(table_prefix: Path = TABLE_PREFIX):
    """Nạp bảng conformal.

    Trả (ConformalTable, provenance); provenance gồm đường dẫn các file
    bảng và mô tả bảng, dùng để ghi log.
    """
    table = _cf.ConformalTable.load(str(table_prefix))
    npz_path = Path(str(table_prefix) + ".npz")
    json_path = Path(str(table_prefix) + ".json")
    prov = dict(
        table_npz=dict(path=str(npz_path)),
        table_json=dict(path=str(json_path)),
        table_protocol_version=table.meta.get("protocol_version"),
        table_scope=TABLE_SCOPE_NOTE,
        score_form=SCORE_FORM_NOTE,
    )
    return table, prov


def _centers_radii(ob) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Lấy (centers, radii) từ một vật cản cầu ở đầu vào của plan():
    dict có centers/radii, hoặc tuple (tâm, bán kính)."""
    if isinstance(ob, dict):
        return (
            np.asarray(ob["centers"], dtype=float),
            np.asarray(ob["radii"], dtype=float),
        )
    o, r = ob
    return np.asarray(o, dtype=float), np.asarray(r, dtype=float)


class MethodAdapter:
    """Bọc plan() cho các cấu hình A1/A2/A3: biến đổi vật cản, chọn điểm
    khởi tạo, cập nhật đồng hồ trễ và ghi lại thông tin từng chu kỳ.

    Tham số:
      method    : "A1", "A2" hoặc "A3".
      eps       : mức ε để tra ρ_traj(ε) trong bảng conformal.
      age_group : nhóm tuổi dùng để tra bảng; None = nhóm của n_hist = 8.
      a_max     : cận trên gia tốc tâm vật cản (m/s²).
      t_cap     : trần cửa sổ trễ (s); None = không chặn.
      table, table_provenance : bảng conformal đã nạp sẵn (tuỳ chọn); nếu
                  không truyền, A1/A3 tự nạp.
    """

    def __init__(
        self,
        method: str,
        eps: float = 0.05,
        age_group: str | None = None,
        a_max: float = 0.0,
        t_cap: float | None = None,
        table=None,
        table_provenance: dict | None = None,
    ):
        """Khởi tạo adapter cho một phương pháp; tham số xem docstring lớp."""
        if method not in ("A1", "A2", "A3"):
            raise ValueError(f"adapter only serves A1/A2/A3, got {method!r}")
        self.method = method
        self.eps = float(eps)
        self.a_max = float(a_max)
        # Trần cửa sổ trễ; chỉ có tác dụng với A1/A3 (qua ReplanClock trong
        # _margin_rho). A2 không có hạng tử trôi.
        if t_cap is not None and t_cap <= 0:
            raise ValueError("t_cap must be positive")
        self.t_cap = None if t_cap is None else float(t_cap)
        self.uses_h1 = method in ("A1", "A3")
        self.uses_h2 = method in ("A2", "A3")
        # Nhóm tuổi mặc định: nhóm của n_hist = 8, giá trị vòng lặp dùng.
        self.age_group = age_group or _cf.age_group_of(8)
        self.table = table
        self.table_provenance = table_provenance
        if self.uses_h1:
            if table is None:
                self.table, self.table_provenance = load_conformal_table()
            # ρ_traj(ε) cố định trong suốt episode.
            self.rho_traj = self.table.rho_traj(self.age_group, self.eps)
        else:
            self.rho_traj = None
        self._clock: _mh.ReplanClock | None = None  # tạo ở lần gọi đầu
        self._history: list = []  # mỗi chu kỳ: solve_time, T_r, δ_trôi, ...
        # Thông tin cửa sổ trễ (chỉ A1/A3). _margin_rho đặt
        # _pending_window_info trong transform(); wrap_plan chuyển nó thành
        # last_window_info (cả trên adapter lẫn trên hàm đã bọc) để
        # sim_realtime ghi vào bản ghi chu kỳ, và thêm vào _window_records.
        self.last_window_info: dict | None = None
        self._pending_window_info: dict | None = None
        self._window_records: list = []
        self._n_calls = 0  # số lần plan() đã được gọi
        self._n_axis_clamps = 0  # số phần tử bán trục bị kẹp (A2/A3)
        # Trạng thái cho điểm khởi tạo: tổng biến dư của lần plan() trước
        # (None trước lần đầu) và bản ghi của từng chu kỳ.
        self._prev_slack: float | None = None
        self._seed_records: list = []
        self.last_seed_info: dict | None = None

    # ---- đồng hồ cửa sổ trễ ----
    def _ensure_clock(self, prm) -> None:
        """Tạo ReplanClock nếu chưa có. Đồng hồ vốn chỉ được tạo
        trong _margin_rho — đường chỉ chạy khi có vật cản để tính
        biên; chu kỳ KHÔNG vật cản không qua _margin_rho nhưng vẫn
        phải có đồng hồ để wrap_plan cập nhật được (sửa lỗi
        2026-11-24: A1/A3 crash "'NoneType' object has no
        attribute 'update'" trên kịch bản không vật cản). Cập
        nhật đồng hồ ở MỌI chu kỳ cũng là điều đúng nghĩa: T_r
        của chu kỳ sau phải là thời gian giải của chu kỳ LIỀN
        TRƯỚC (quy ước Chương 6), kể cả khi chu kỳ liền trước
        không có vật cản nào — nếu bỏ qua, T_r sẽ lấy nhầm thời
        gian giải của chu kỳ có-vật gần nhất."""
        if self._clock is None:
            self._clock = _mh.ReplanClock(T_replan=prm.dt, T_cap=self.t_cap)

    # ---- lắp ráp rho (A1/A3) cho một vật cản ----
    def _margin_rho(self, centers, prm) -> tuple[np.ndarray, dict]:
        """Mảng rho (K+1,) cho planner:
        ρ_traj(ε) + δ_trôi + δ_mẫu, cùng một giá trị cho mọi k.
        Trả (rho, info) với info gồm T_r, v_hat, delta_drift, delta_sample."""
        self._ensure_clock(prm)
        # δ_trôi dùng cửa sổ đã áp trần T_r_eff = min(T_r, T_cap) (bằng T_r
        # khi không có trần); info ghi T_r chưa chặn.
        wi = self._clock.window_info()
        self._pending_window_info = wi
        T_r = wi["T_r"]
        # Tốc độ vật cản ước lượng từ hai tâm dự báo đầu tiên; vật tĩnh
        # (centers chỉ một điểm) có v̂ = 0.
        v_hat = (
            _mh.estimate_speed(centers, prm.dt)
            if centers is not None and centers.ndim == 2
            else 0.0
        )
        dd = _mh.delta_drift(
            v_hat, self.a_max, wi["T_r_eff"], _mh.TRACKING_ERROR_DEFAULT
        )
        ds = _mh.delta_sample(prm.u_max, self.a_max, prm.dt)
        rho = _mh.compose_rho_traj_for_planner(self.rho_traj, dd, ds, K=prm.K)
        return rho, dict(T_r=T_r, v_hat=v_hat, delta_drift=dd, delta_sample=ds)

    # ---- biến đổi danh sách vật cản ----
    def transform(self, obstacles, prm) -> tuple[list, dict]:
        """Tạo danh sách vật cản mới cho planner theo cấu hình:
          - A1/A3: thay rho bằng rho của _margin_rho;
          - A2/A3: đổi cầu sang dict ellipsoid (axes = r·(1, 1, 1), yaw = 0).
        Trả (danh sách mới, dict(per_obstacle=[info của từng vật])).
        Đầu vào phải là vật cản cầu."""
        out, parts = [], []
        for ob in obstacles:
            if isinstance(ob, dict) and "axes" in ob:
                raise ValueError(
                    "adapter expects spherical obstacles (get_obs output "
                    "of sim_realtime); ellipsoid input is not supported"
                )
            centers, radii = _centers_radii(ob)
            info = {}
            rho_new = None
            if self.uses_h1:
                rho_new, info = self._margin_rho(centers, prm)
            if self.uses_h2:
                if radii.ndim == 1:
                    # Bán kính theo thời gian (K+1,) -> axes (K+1, 3),
                    # yaws (K+1,).
                    axes = radii.reshape(-1, 1) * np.ones((1, 3))
                    yaws = np.zeros(len(axes))  # (K+1,)
                else:
                    # Bán kính là một số (vật tĩnh, mode frozen): axes (3,)
                    # và yaws là một số. dca_planner.normalize_obstacles nhận
                    # 'yaws' dạng (K+1,) hoặc một số, không nhận mảng (3,).
                    axes = radii * np.ones(3)
                    yaws = 0.0
                # Kẹp bán trục vào dải hợp lệ và đếm số phần tử bị kẹp.
                clamped = np.clip(axes, A_ABS_MIN, A_ABS_MAX)
                self._n_axis_clamps += int(np.count_nonzero(clamped != axes))
                axes = clamped
                new = dict(centers=centers, axes=axes, yaws=yaws)
                # rho: của A1/A3 nếu có, nếu không thì giữ rho gốc. Tuple tĩnh
                # không có rho; thiếu khoá rho thì planner coi rho = 0.
                if rho_new is not None:
                    new["rho"] = rho_new
                elif isinstance(ob, dict) and "rho" in ob:
                    new["rho"] = np.asarray(ob["rho"], dtype=float)
            else:
                # A1: giữ dạng cầu, chỉ thay rho.
                new = dict(centers=centers, radii=radii)
                if rho_new is not None:
                    new["rho"] = rho_new
                elif isinstance(ob, dict) and "rho" in ob:
                    new["rho"] = np.asarray(ob["rho"], dtype=float)
            out.append(new)
            parts.append(info)
        return out, dict(per_obstacle=parts)

    # ---- điểm khởi tạo: dãy theo thời gian của một vật đã biến đổi ----
    @staticmethod
    def _series(ob, K: int):
        """Trả (centers (K+1, 3), bán kính cơ sở (K+1,), rho (K+1,)) của một
        vật cản đã biến đổi, mở rộng các giá trị tĩnh ra K+1 bước như
        dca_planner.normalize_obstacles.
        Bán kính cơ sở là bán kính (cầu) hoặc bán trục lớn nhất tại mỗi bước
        (ellipsoid, tức bán kính cầu bao). Thiếu rho thì rho = 0, giống cách
        planner xử lý."""
        if isinstance(ob, dict) and "axes" in ob:
            c = np.asarray(ob["centers"], dtype=float)
            centers = (
                c
                if c.shape == (K + 1, 3)
                else np.repeat(c.reshape(1, 3), K + 1, axis=0)
            )
            ax = np.asarray(ob["axes"], dtype=float)
            AX = (
                ax
                if ax.shape == (K + 1, 3)
                else np.repeat(ax.reshape(1, 3), K + 1, axis=0)
            )
            base = np.max(AX, axis=1)
        elif isinstance(ob, dict):
            c = np.asarray(ob["centers"], dtype=float)
            centers = (
                c
                if c.shape == (K + 1, 3)
                else np.repeat(c.reshape(1, 3), K + 1, axis=0)
            )
            r = np.asarray(ob["radii"], dtype=float)
            base = r if r.shape == (K + 1,) else float(r) * np.ones(K + 1)
        else:
            o, r = ob
            centers = np.repeat(np.asarray(o, dtype=float).reshape(1, 3), K + 1, axis=0)
            base = float(r) * np.ones(K + 1)
        rho = np.zeros(K + 1)
        if isinstance(ob, dict) and ob.get("rho") is not None:
            rr = np.asarray(ob["rho"], dtype=float)
            rho = rr if rr.shape == (K + 1,) else float(rr) * np.ones(K + 1)
        return centers, base, rho

    @staticmethod
    def _straight_seed(p0, goal, K: int) -> np.ndarray:
        """Đường thẳng p0 -> goal gồm K+1 điểm, giống điểm khởi tạo plan()
        tự dựng khi seed_P = None. Chỉ dùng để đo seed_max_viol."""
        ts = np.linspace(0.0, 1.0, K + 1)[:, None]
        return (
            np.asarray(p0, dtype=float)[None, :] * (1 - ts)
            + np.asarray(goal, dtype=float)[None, :] * ts
        )

    def _seed_max_viol(self, seed_P, new_obs, prm) -> float | None:
        """Vi phạm lớn nhất của điểm khởi tạo:
            max_{m,k} (need_m[k] − ‖seed[k] − o_m[k]‖),
            need_m[k] = R_m[k] + d + rho_m[k]
        với R là bán kính cơ sở của _series. Giá trị ≤ 0 nghĩa là điểm khởi
        tạo thoả need tại mọi nút. None nếu không có vật cản."""
        viol = None
        for ob in new_obs:
            centers, base, rho = self._series(ob, prm.K)
            need = base + float(prm.d_safe) + rho
            dist = np.linalg.norm(seed_P - centers, axis=1)
            v = float(np.max(need - dist))
            viol = v if viol is None else max(viol, v)
        return viol

    def _seed_tuples(self, new_obs, prm):
        """Đầu vào cho A* mức L1: (tuples, extras).
        tuples: các cầu bao vùng quét (một cầu cho mỗi vật và mỗi bước k);
        extras: rho_m[k] tương ứng với từng cầu.
        bounding_sphere_tuples được gọi riêng cho từng vật để biết cầu nào
        thuộc vật nào. astar_seed cộng thêm d_safe, nên bán kính chặn là
        R_bao + d + rho, tức need tính trên cầu bao."""
        tuples, extras = [], []
        for ob in new_obs:
            sub = bounding_sphere_tuples([ob])
            _, _, rho = self._series(ob, prm.K)
            for k in range(len(sub)):
                tuples.append(sub[k])
                extras.append(float(rho[min(k, len(rho) - 1)]))
        return tuples, extras

    def _decide_seed(self, p0, goal, new_obs, prm, seed_in):
        """Chọn điểm khởi tạo cho chu kỳ hiện tại.
        Trả (seed_info, seed_P đưa vào plan() hoặc None). seed_info chưa có
        call/iterations/total_slack; wrap_plan bổ sung sau khi plan() chạy."""
        # Xác định chu kỳ này có phải chu kỳ kích hoạt A* không.
        trigger = None
        if self._n_calls == 1:
            trigger = "cycle0"
        elif self._prev_slack is not None and (
            not np.isfinite(self._prev_slack) or self._prev_slack > SEED_SLACK_TRIGGER
        ):
            trigger = "post_crash_or_slack"
        # Không kích hoạt: dùng seed_in (warm-start) hoặc đường thẳng.
        if trigger is None:
            seed_use = seed_in
            probe = (
                seed_use
                if seed_use is not None
                else self._straight_seed(p0, goal, prm.K)
            )
            return dict(
                seed_source="warm" if seed_use is not None else "straight",
                seed_trigger="none",
                t_seed=0.0,
                seed_max_viol=self._seed_max_viol(probe, new_obs, prm),
                seed_budget_exceeded=False,
                seed_nodes=0,
            ), seed_use
        tuples, extras = self._seed_tuples(new_obs, prm)
        # Mức L1: chặn nút ở need + guard + dip, kiểm tra đoạn ở need + guard
        # (segment_extras = rho, guard được astar_seed cộng riêng); xem
        # SEED_NODE_DIP_M. Vì vậy điểm khởi tạo L1 có seed_max_viol ≤ −guard.
        node_extras = [e + SEED_SEGMENT_GUARD_M + SEED_NODE_DIP_M for e in extras]
        t0 = time.perf_counter()
        info1: dict = {}
        seed = astar_seed(
            p0,
            goal,
            tuples,
            prm.K + 1,
            d_safe=prm.d_safe,
            z_floor=prm.z_floor,
            extra_margin=node_extras,
            segment_use_need=True,
            segment_guard=SEED_SEGMENT_GUARD_M,
            segment_extras=extras,
            max_nodes=SEED_MAX_NODES,
            time_budget_s=SEED_TIME_BUDGET_S,
            info_out=info1,
        )
        nodes = int(info1.get("nodes_explored", 0))
        exceeded = bool(info1.get("budget_exceeded", False))
        if seed is not None:
            source = "astar_L1"
        else:
            # Mức L2: A* với phần phồng cố định SEED_L2_EXTRA_M quanh các
            # cầu bao; nếu vẫn thất bại thì L3 (seed = None, đường thẳng).
            info2: dict = {}
            seed = astar_seed(
                p0,
                goal,
                tuples,
                prm.K + 1,
                d_safe=prm.d_safe,
                z_floor=prm.z_floor,
                extra_margin=SEED_L2_EXTRA_M,
                max_nodes=SEED_MAX_NODES,
                time_budget_s=SEED_TIME_BUDGET_S,
                info_out=info2,
            )
            nodes += int(info2.get("nodes_explored", 0))
            exceeded = exceeded or bool(info2.get("budget_exceeded", False))
            source = "astar_L2" if seed is not None else "straight"
        t_seed = float(time.perf_counter() - t0)
        # Đo vi phạm của điểm khởi tạo thực sự được dùng.
        probe = seed if seed is not None else self._straight_seed(p0, goal, prm.K)
        return dict(
            seed_source=source,
            seed_trigger=trigger,
            t_seed=t_seed,
            seed_max_viol=self._seed_max_viol(probe, new_obs, prm),
            seed_budget_exceeded=exceeded,
            seed_nodes=nodes,
        ), seed

    # ---- bọc plan() ----
    def wrap_plan(self, base_plan):
        """Trả hàm cùng chữ ký với plan(): biến đổi vật cản, chọn điểm khởi
        tạo, gọi base_plan, rồi cập nhật đồng hồ trễ và các bản ghi. Hàm trả
        về có thêm thuộc tính last_seed_info và last_window_info của lần gọi
        gần nhất."""

        def wrapped(p0, v0, goal, obstacles, prm, seed_P=None):
            self._n_calls += 1
            self._pending_window_info = None
            new_obs, info = self.transform(obstacles, prm)
            seed_info, seed_use = self._decide_seed(p0, goal, new_obs, prm, seed_P)
            res = base_plan(p0, v0, goal, new_obs, prm, seed_P=seed_use)
            seed_info = dict(
                seed_info,
                call=self._n_calls - 1,
                iterations=int(res.iterations),
                total_slack=float(res.total_slack),
            )
            self._seed_records.append(seed_info)
            self.last_seed_info = seed_info
            wrapped.last_seed_info = seed_info
            if self.uses_h1:
                # Công bố window_info của lần gọi này cho sim_realtime. Giá trị
                # là None nếu không có vật cản nào (_margin_rho không được gọi).
                self.last_window_info = self._pending_window_info
                wrapped.last_window_info = self.last_window_info
                if self.last_window_info is not None:
                    self._window_records.append(
                        dict(call=self._n_calls - 1, **self.last_window_info)
                    )
            self._prev_slack = float(res.total_slack)
            if self.uses_h1:
                # Cập nhật đồng hồ: T_r của chu kỳ sau = T_replan + thời gian
                # giải + thời gian tìm điểm khởi tạo của chu kỳ này. Thời gian
                # dự báo do vòng lặp đo nên không được cộng ở đây. Đồng hồ
                # phải tồn tại cả khi chu kỳ này không có vật cản nào (khi đó
                # _margin_rho không chạy nên đồng hồ chưa được tạo).
                self._ensure_clock(prm)
                self._clock.update(float(res.solve_time) + seed_info["t_seed"])
                # Lưu thông tin biên của vật cản đầu tiên cho summary().
                rec = dict(call=len(self._history), solve_time=float(res.solve_time))
                if info["per_obstacle"]:
                    rec.update(
                        {k: float(v) for k, v in info["per_obstacle"][0].items()}
                    )
                self._history.append(rec)
            return res

        wrapped.last_seed_info = None
        wrapped.last_window_info = None
        return wrapped

    # ---- provenance + tóm tắt cho log ----
    def provenance(self) -> dict:
        """Thông tin để ghi log: cấu hình, đường dẫn các module và bảng đã
        dùng, cùng các chuỗi mô tả quy tắc."""
        prov = dict(
            adapter_module=dict(path=str(Path(__file__).resolve())),
            astar_module=dict(path=str(_HERE / "astar_seed.py")),
            method=self.method,
            uses_h1_margin=self.uses_h1,
            uses_h2_ellipsoid_branch=self.uses_h2,
            seed_rule=SEED_RULE_NOTE,
            seed_budget=dict(
                max_nodes=SEED_MAX_NODES,
                time_budget_s=SEED_TIME_BUDGET_S,
                segment_guard_m=SEED_SEGMENT_GUARD_M,
            ),
            # Trần như được truyền vào (None = không chặn); chỉ A1/A3 dùng.
            t_cap_s=self.t_cap,
        )
        if self.uses_h1:
            prov.update(
                margin_module=dict(path=str(_HERE / "safety_margin.py")),
                conformal_module=dict(path=str(_HERE / "conformal.py")),
                conformal_table=self.table_provenance,
                eps=self.eps,
                age_group=self.age_group,
                age_rule=AGE_RULE_NOTE,
                rho_traj_m=float(self.rho_traj),
                a_max_m_s2=self.a_max,
                tracking_error_m=_mh.TRACKING_ERROR_DEFAULT,
                tracking_error_measured=bool(_mh.TRACKING_ERROR_MEASURED),
                margin_path=(
                    "Q4: rho_planner[k] = rho_traj(eps) + "
                    "delta_drift + delta_sample; "
                    "extra_margin = 0; d_safe unchanged"
                ),
                t_cap_rule=(
                    "DV-H7: delta_drift uses the effective "
                    "window min(T_r, T_cap) from "
                    "safety_margin.ReplanClock.window_info(); "
                    "t_cap_s = None -> no cap; logged T_r is "
                    "the uncapped value; each cycle logs "
                    "T_r / T_r_eff / drift_capped / T_cap"
                ),
            )
        if self.uses_h2:
            prov.update(
                ellipsoid_module=dict(path=str(_HERE / "ellipsoid.py")),
                shape_path=(
                    "sphere dict -> ellipsoid dict with "
                    "axes=(r_k, r_k, r_k), yaw=0 from the loop's "
                    "sphere forecast; the full shape forecaster "
                    "shape_forecast.kalman_ellipsoid_obstacles "
                    "is not wired into the loop"
                ),
            )
        return prov

    def summary(self) -> dict:
        """Tóm tắt sau episode: số lần gọi, số lần kẹp bán trục, T_r và
        δ_trôi lớn nhất (A1/A3), các bản ghi điểm khởi tạo và cửa sổ trễ."""
        out = dict(n_plan_calls=self._n_calls, n_axis_clamps=self._n_axis_clamps)
        if self._history:
            out["T_r_max_s"] = max(h.get("T_r", 0.0) for h in self._history)
            out["delta_drift_max_m"] = max(
                h.get("delta_drift", 0.0) for h in self._history
            )
        if self._seed_records:
            # Bản ghi điểm khởi tạo của từng chu kỳ, và thời gian A* của
            # các chu kỳ có kích hoạt.
            out["seed_records"] = self._seed_records
            out["t_seed_astar_s"] = [
                r["t_seed"] for r in self._seed_records if r["seed_trigger"] != "none"
            ]
        if self._window_records:
            # Bản ghi cửa sổ trễ của từng chu kỳ (chỉ A1/A3) và tỉ lệ chu kỳ
            # có T_r > T_cap.
            n_capped = sum(1 for w in self._window_records if w["capped"])
            out["window_records"] = self._window_records
            out["n_drift_capped"] = n_capped
            out["drift_capped_rate"] = n_capped / len(self._window_records)
        return out
