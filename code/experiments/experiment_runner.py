# -*- coding: utf-8 -*-
"""
experiment_runner.py — chạy MỘT cấu hình (phương pháp × kịch bản × seed)
và ghi log JSON cấu trúc theo giao thức thực nghiệm ST-DCA (bản 1.2).

Mọi lần chạy đều có log, kể cả khi lỗi: lỗi được ghi với mã lỗi và
traceback trong chính file log, không im lặng bỏ qua. Log không ghi
đè (§7.2): chạy lại cùng run_id tạo file mới có hậu tố lần chạy _r2,
_r3, ...

Mỗi log chứa (§7.2/§9.1): mã + phiên bản giao thức, code_commit (hash
nội dung của các file code quyết định hành vi chạy — thay cho git
hash vì thư mục này không phải git repo), thông tin máy (kèm
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS theo §0), cấu hình, kịch bản,
chế độ, seed, mã trạng thái episode §6.4 (ok / solver_fail / timeout
/ crash / infra — trường `status` ở tầng log chính là mã này; ngoại
lệ thoát khỏi run_episode được ghi là `crash`), toàn bộ chỉ số §1.

Ví dụ:
  python3 experiment_runner.py --method A0 --scenario D1_cat_ngang \
      --seed 9000000 --mode kalman --time-model realtime \
      --outdir ../results/logs

Phương pháp đã đăng ký:
  A0 — Bài 1 gốc: cầu bao, δ = 0,03 m, ρ_k với κ = 2 (PlannerParams
       mặc định, không sửa một tham số nào).
  A1 — A0 + biên H1 (bảng conformal hiện hành, ε qua --eps);
  A2 — A0 + nhánh ellipsoid H2 của planner (vật cầu suy biến);
  A3 — ST-DCA đầy đủ (H1 + H2). A1–A3 nối qua method_adapter.py
       (bọc hàm plan của sim_realtime, provenance ghi ở
       config.method_adapter trong log) — xem README_HA_TANG §3.2.
       Phương pháp chưa đăng ký thì log lỗi METHOD_NOT_IMPLEMENTED
       thay vì chạy sai.
  Ngo2023 / CBF2021 / MPCQP — ba baseline Tier-1 (GĐ4, SPEC v1.1)
       trong gói baselines/: đăng ký vào cùng runner này (tích hợp
       2026-11-22) — dispatch như baselines/run_baseline.py: thay
       sim_realtime.plan bằng plan() của baseline kèm lớp bọc ghi
       metadata từng lần gọi vào trường `plan_calls` của log;
       trường `planner_file` của log trỏ tới file baseline (path +
       sha256 đầy đủ, origin "baseline_module"), code_commit tính
       thêm file baseline + baselines/common.py. Tham số harness
       PlannerParams mặc định như A0 (SPEC_cong_bang §2).

Kịch bản chọn theo tên qua sim_realtime.run_episode: họ D như cũ,
họ S / loạt A / loạt B / họ F qua scenarios_pkg.registry (49 tên);
kịch bản loạt B mang platform (v_max, u_max) — run_episode tự ghi
đè vào PlannerParams của episode và ghi scenario_platform +
planner_effective vào config của log.

Chế độ quan sát (--obs-mode, mặc định "sphere"): "sphere" là
đường quan sát cầu cũ; "ellipsoid" bật hook quan sát ellipsoid
của sim_realtime (get_obs trả dict ellipsoid của tầng hình dạng
— README_HA_TANG §3.4). Ở chế độ ellipsoid, A2/A3 KHÔNG qua
method_adapter (adapter chỉ nhận đầu vào cầu): quan sát ellipsoid
gốc thay thế bước chuyển đổi cầu -> ellipsoid của adapter, và với
A3 phần lắp ráp biên H1 của adapter KHÔNG được áp — rho là rho
plug-in κ = 2 của tầng hình dạng (ghi tường minh ở
config.obs_mode_note trong log). A1 và ba baseline chỉ tiêu thụ
dự báo cầu nên kết hợp với --obs-mode ellipsoid bị từ chối tường
minh (mã obs_mode_incompatible) thay vì chạy sai.

FILE PLANNER ĐƯỢC NẠP TƯỜNG MINH (sửa 2026-11-22): cơ chế cũ chèn
code_review/v2026-11-20 lên đầu sys.path làm `import dca_planner`
nạp BẢN GỐC thay vì bản làm việc exp/code mà không ai hay biết
(phát hiện ở ô kiểm chứng tính trơ — xem inertness_bugB.md, phụ
lục truy vết). Nay: --planner chỉ định rõ file dca_planner.py của
lần chạy (mặc định = bản làm việc exp/code/dca_planner.py), file
được nạp vào sys.modules TRƯỚC khi import sim_realtime, và mọi log
ghi trường `planner_file` = {path tuyệt đối, sha256 đầy đủ, origin}
cùng `code_files` — provenance kiểm chứng được, không suy đoán.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent
# Cây package của người dùng (2026-11-24): thư mục code/ (chứa
# core/, baselines/, sim/, experiments/) phải nằm trên sys.path để
# các import tuyệt đối core.*/baselines.*/sim.* nạp được.
# KHÔNG chèn code_review nữa — xem docstring đầu file.
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

DEFAULT_PLANNER_PATH = _CODE_DIR / "core" / "dca_planner.py"


def resolve_planner_path(raw: str | None) -> tuple[Path, str]:
    """Đường dẫn file planner từ --planner: nhận file .py hoặc thư
    mục chứa dca_planner.py. Mặc định: core/dca_planner.py của cây
    này (bản refactor của người dùng)."""
    if raw is None:
        return DEFAULT_PLANNER_PATH, "default_working_copy"
    p = Path(raw).expanduser()
    if p.is_dir():
        p = p / "dca_planner.py"
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"planner_file_not_found:{p}")
    return p, "cli_arg"


def load_planner(path: Path):
    """Nạp tường minh ĐÚNG file planner chỉ định vào sys.modules
    dưới tên 'core.dca_planner' TRƯỚC khi module nào khác import
    nó (đặc biệt sim.sim_realtime), để mọi
    `from core.dca_planner import ...` về sau đều trỏ đúng file
    này. Trả module đã nạp."""
    import importlib.util

    import core  # bảo đảm gói cha có trong sys.modules
    spec = importlib.util.spec_from_file_location(
        "core.dca_planner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"planner_load_failed:{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["core.dca_planner"] = mod
    spec.loader.exec_module(mod)
    return mod


# Nạp planner TRƯỚC khi import sim_realtime: quét --planner từ dòng
# lệnh ngay tại đây (main() khai lại cờ cho --help; giá trị hiệu lực
# là giá trị đã nạp ở đây).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--planner", default=None)
PLANNER_PATH, PLANNER_ORIGIN = resolve_planner_path(
    _pre.parse_known_args()[0].planner)
PLANNER_MODULE = load_planner(PLANNER_PATH)
PLANNER_SHA256 = hashlib.sha256(PLANNER_PATH.read_bytes()).hexdigest()
PlannerParams = PLANNER_MODULE.PlannerParams

from sim.sim_realtime import run_episode   # noqa: E402

PROTOCOL = "ST-DCA-EXP"
PROTOCOL_VERSION = "1.2"

METHODS = {
    "A0": "Bài 1 gốc (cầu, δ=0,03 m, κ=2)",
    "A1": "A0 + biên H1 ba hạng tử (conformal, đường Q4) — method_adapter",
    "A2": "A0 + nhánh ellipsoid H2 (cầu suy biến) — method_adapter",
    "A3": "ST-DCA đầy đủ (H1 + H2) — method_adapter",
}

# Baseline Tier-1 (gói baselines/, SPEC v1.1) — module + lớp cấu
# hình của từng phương pháp tra trong baselines.METHODS lúc chạy;
# bảng này chỉ giữ mô tả cho log.
BASELINE_METHODS = {
    "Ngo2023": "Baseline Tier-1: penalty-DCA của Ngo et al. (CDC 2023) "
               "— baselines/ngo2023.py",
    "CBF2021": "Baseline Tier-1: Convexified-CBF của Varun et al. "
               "(ICC 2021) — baselines/cbf2021.py",
    "MPCQP": "Baseline Tier-1: MPC-QP tuyến tính hóa một lần "
             "— baselines/mpcqp.py",
}
METHODS.update(BASELINE_METHODS)

# Tên lớp cấu hình riêng của từng baseline (trong module của nó).
_BASELINE_CONFIG_CLS = {"Ngo2023": "NgoConfig", "CBF2021": "CbfConfig",
                        "MPCQP": "MpcqpConfig"}


def environment_info() -> dict:
    import importlib.metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return None

    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    ram_gb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                ram_gb = round(int(line.split()[1]) / 1e6, 2)
                break
    except Exception:
        pass
    return dict(
        hostname=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        cpu=cpu,
        vcpus=os.cpu_count(),
        cpu_count=os.cpu_count(),
        ram_gb=ram_gb,
        python=platform.python_version(),
        numpy=ver("numpy"), scipy=ver("scipy"),
        cvxpy=ver("cvxpy"), osqp=ver("osqp"),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
    )


def code_commit(extra_files: dict | None = None,
                extra_full: str | None = None) -> tuple[str, dict]:
    """Định danh phiên bản code bằng hash nội dung các file quyết định
    hành vi chạy (đường dẫn thật của module đã nạp + file này +
    sim_realtime.py). Trả (hash gộp rút gọn, {module: {path, sha256}}).

    extra_files: {tên: đường dẫn} cộng thêm vào tập hash (chạy
    baseline: file baseline + baselines/common.py); extra_full là
    tên file trong đó cần ghi thêm sha256 ĐẦY ĐỦ như dca_planner.py.
    """
    from core import dca_planner
    from core import obstacle_predict
    from sim import dynamic_scenarios
    from sim import sim_realtime

    files = {}
    for mod in (dca_planner, dynamic_scenarios, obstacle_predict,
                sim_realtime):
        p = Path(mod.__file__).resolve()
        files[p.name] = p
    files[Path(__file__).name] = Path(__file__).resolve()
    if extra_files:
        files.update(extra_files)
    detail, acc = {}, hashlib.sha256()
    for name in sorted(files):
        h = hashlib.sha256(files[name].read_bytes()).hexdigest()
        detail[name] = dict(path=str(files[name]), sha256=h[:16])
        if name == "dca_planner.py" or name == extra_full:
            # Provenance planner: hash ĐẦY ĐỦ (trường rút gọn 16 ký
            # tự ở trên giữ cho tương thích log cũ).
            detail[name]["sha256_full"] = h
        acc.update(f"{name}:{h}\n".encode())
    return f"sha256:{acc.hexdigest()[:16]}", detail


def build_params(method: str) -> PlannerParams:
    if method == "A0":
        return PlannerParams()  # mặc định gốc, không chỉnh
    if method in BASELINE_METHODS:
        # Baseline Tier-1: tham số harness y hệt A0 (SPEC_cong_bang §2)
        return PlannerParams()
    if method in ("A1", "A3"):
        # Biên H1 gộp toàn bộ vào mảng rho (method_adapter), nên
        # đệm hằng số của planner phải bằng 0; d_safe giữ nguyên.
        return PlannerParams(extra_margin=0.0)
    if method == "A2":
        return PlannerParams()  # biên legacy như A0, chỉ đổi nhánh hình học
    raise KeyError(f"method_not_implemented:{method}")


def _error_code(exc: BaseException) -> str:
    """Mã lỗi sạch (bản cũ tách str(KeyError) nên dính ký tự ' thừa)."""
    raw = exc.args[0] if exc.args and isinstance(exc.args[0], str) \
        else str(exc)
    return raw.split(":", 1)[0].strip().strip("'\"") or type(exc).__name__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="kalman",
                    choices=["truth", "kalman", "frozen"])
    ap.add_argument("--obs-mode", default="sphere",
                    choices=["sphere", "ellipsoid"],
                    help="chế độ quan sát của vòng lặp: 'sphere' "
                         "(mặc định — quan sát cầu cũ) hoặc "
                         "'ellipsoid' (hook quan sát ellipsoid của "
                         "sim_realtime, dict của tầng hình dạng — "
                         "README_HA_TANG §3.4; A2/A3 không qua "
                         "method_adapter ở chế độ này, A1 và "
                         "baseline bị từ chối)")
    ap.add_argument("--time-model", default="realtime",
                    choices=["realtime", "frozen"])
    ap.add_argument("--t-max", type=float, default=None,
                    help="mặc định: theo họ kịch bản (§2) trong run_episode")
    ap.add_argument("--planner", default=None,
                    help="file dca_planner.py (hoặc thư mục chứa nó) "
                         "nạp TƯỜNG MINH cho lần chạy; mặc định: bản "
                         "làm việc exp/code/dca_planner.py. Giá trị "
                         "hiệu lực được nạp lúc khởi động module và "
                         "ghi vào log ở trường planner_file.")
    ap.add_argument("--eps", type=float, default=0.05,
                    choices=[0.05, 0.1],
                    help="mức ε của quantile conformal ρ_traj cho "
                         "A1/A3 (mặc định 0,05 — mức chính của giao thức)")
    ap.add_argument("--age-group", default=None,
                    choices=["age_new", "age_mid", "age_old", "__all__"],
                    help="ghi đè nhóm tuổi tra bảng conformal cho "
                         "A1/A3; mặc định: nhóm của bộ dự báo vòng "
                         "lặp (n_hist=8 -> age_mid) — xem "
                         "method_adapter.AGE_RULE_NOTE")
    ap.add_argument("--t-cap", type=float, default=None,
                    help="trần cửa sổ trễ T_cap (s) của DV-H7 cho "
                         "A1/A3: δ_trôi tính trên cửa sổ hiệu lực "
                         "min(T_r, T_cap) qua ReplanClock của "
                         "margin_h1, và mọi chu kỳ A1/A3 ghi T_r / "
                         "T_r_eff / drift_capped / T_cap vào log; "
                         "mặc định None = không chặn (hành vi hiện "
                         "hành giữ nguyên). Giá trị trần hiệu lực "
                         "cuối cùng theo changelog giao thức bản "
                         "2.0 (dự thảo 2,0 s)")
    ap.add_argument("--anchor-log", action="store_true",
                    help="ghi log phụ *_anchors.json: trạng thái neo "
                         "(v0 đầy đủ + phân loại hộp theo dung sai "
                         "3e-3 của Quyết định 1 §1.5) ở MỌI lần gọi "
                         "plan() — điều kiện tiên quyết của Ô 1 "
                         "DV-H4; mặc định tắt, log chính không đổi")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    if args.planner is not None:
        _p, _ = resolve_planner_path(args.planner)
        if _p != PLANNER_PATH:  # không thể xảy ra cùng tiến trình
            raise SystemExit(f"planner_mismatch:{_p}!={PLANNER_PATH}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = (f"{args.method}_{args.scenario}_{args.mode}_{args.time_model}"
              f"_seed{args.seed}")
    if args.obs_mode == "ellipsoid":
        # Hậu tố chỉ khi bật ellipsoid — run_id của đường cầu giữ
        # nguyên định dạng cũ để tên log đã kiểm chứng không đổi.
        run_id += "_ellipsoidobs"
    # Baseline Tier-1: tra module + file của phương pháp TRƯỚC khi
    # tính code_commit và dựng log (planner_file trỏ file baseline).
    baseline = None
    if args.method in BASELINE_METHODS:
        from baselines import METHODS as _BASELINE_MODULES
        _mod, _config_id = _BASELINE_MODULES[args.method]
        baseline = dict(module=_mod, config_id=_config_id,
                        file=Path(_mod.__file__).resolve())
    if baseline is not None:
        commit, commit_files = code_commit(
            extra_files={baseline["file"].name: baseline["file"],
                         "common.py": _CODE_DIR / "baselines" / "common.py"},
            extra_full=baseline["file"].name)
        planner_file = dict(
            path=str(baseline["file"]),
            sha256=hashlib.sha256(
                baseline["file"].read_bytes()).hexdigest(),
            origin="baseline_module")
    else:
        commit, commit_files = code_commit()
        planner_file = dict(path=str(PLANNER_PATH), sha256=PLANNER_SHA256,
                            origin=PLANNER_ORIGIN)
    env = environment_info()
    log = dict(
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol=PROTOCOL,
        protocol_version=PROTOCOL_VERSION,
        code_commit=commit,
        code_files=commit_files,
        planner_file=planner_file,
        machine=env,
        env="sim",
        status="crash",
        error=None,
        method=args.method,
        method_description=METHODS.get(args.method),
        scenario=args.scenario,
        mode=args.mode,
        obs_mode=args.obs_mode,
        time_model=args.time_model,
        seed=args.seed,
        environment=env,
        config=None,
        metrics=None,
        cycles=None,
        plan_calls=None,
    )
    diag_sink: list = []
    anchor_sink: list | None = [] if args.anchor_log else None
    adapter = None
    plan_calls: list | None = None
    try:
        if args.method not in METHODS:
            raise KeyError(f"method_not_implemented:{args.method}")
        prm = build_params(args.method)
        log["config"] = dict(planner=asdict(prm))
        if args.obs_mode == "ellipsoid":
            # Điểm chọn chế độ quan sát ellipsoid (tối thiểu, xem
            # docstring đầu file và README_HA_TANG §3.4): ghi
            # provenance của tầng hình dạng vào log; các phương
            # pháp chỉ tiêu thụ dự báo cầu bị từ chối tường minh.
            from core import shape_forecast as _sf
            _sf_path = Path(_sf.__file__).resolve()
            log["config"]["shape_forecast_file"] = dict(
                path=str(_sf_path),
                sha256=hashlib.sha256(_sf_path.read_bytes()).hexdigest())
            if args.method == "A1" or baseline is not None:
                raise KeyError(
                    "obs_mode_incompatible:phương pháp này chỉ tiêu "
                    "thụ dự báo cầu; --obs-mode ellipsoid chỉ hợp lệ "
                    "cho A0/A2/A3 (xem README_HA_TANG §3.4)")
        if baseline is not None:
            # Dispatch baseline (như baselines/run_baseline.py):
            # thay sim_realtime.plan bằng plan() của baseline kèm
            # lớp bọc ghi metadata từng lần gọi; giao diện
            # run_episode giữ nguyên, không cơ chế chọn phương
            # pháp thứ hai — cùng điểm nối _sr.plan với adapter.
            from sim import sim_realtime as _sr
            _module = baseline["module"]
            _cfg_obj = getattr(_module,
                               _BASELINE_CONFIG_CLS[args.method])()
            log["config"]["baseline"] = asdict(_cfg_obj)
            log["config"]["baseline_config_id"] = baseline["config_id"]
            plan_calls = []

            def recording_plan(p0, v0, goal, obstacles, _prm,
                               seed_P=None):
                res = _module.plan(p0, v0, goal, obstacles, _prm,
                                   seed_P=seed_P)
                plan_calls.append(dict(
                    call_index=len(plan_calls),
                    iterations=int(res.iterations),
                    total_slack=float(res.total_slack),
                    success_internal=bool(res.success),
                    solve_time=float(res.solve_time),
                    min_clearance_plan=float(res.min_clearance),
                    meta=res.meta))
                return res

            _sr.plan = recording_plan
        elif args.method in ("A1", "A2", "A3"):
            if args.obs_mode == "ellipsoid":
                # Chế độ quan sát ellipsoid: get_obs của
                # sim_realtime đã trả dict ellipsoid gốc của tầng
                # hình dạng — đúng định dạng adapter VỐN sản xuất
                # cho A2/A3 — nên adapter (đầu vào chỉ là cầu, tự
                # bác bỏ dict ellipsoid) KHÔNG được bọc ở chế độ
                # này. Hệ quả phải ghi tường minh: với A3, phần
                # lắp ráp biên H1 của adapter không được áp; rho
                # trong planner là rho plug-in κ = 2 của tầng hình
                # dạng, nên đầu vào planner của A3 trùng A2 ở chế
                # độ này. Muốn A3 đầy đủ (ellipsoid gốc + biên H1)
                # phải mở rộng method_adapter nhận đầu vào
                # ellipsoid — đầu việc riêng, ngoài hook này.
                log["config"]["obs_mode_note"] = (
                    "obs_mode=ellipsoid: quan sát ellipsoid gốc của "
                    "tầng hình dạng (sim_realtime.get_obs), "
                    "method_adapter không bọc; A3 không áp biên H1 "
                    "ở chế độ này — rho là rho plug-in của tầng "
                    "hình dạng, đầu vào planner của A3 trùng A2")
            else:
                # Hook tối thiểu cho A1–A3 (Mộc): bọc hàm plan trong
                # namespace của sim_realtime bằng adapter của cấu hình;
                # sim_realtime.py không bị sửa, giao diện run_episode
                # giữ nguyên. Adapter A1/A3 nạp + kiểm hash bảng
                # conformal tại đây — lỗi bảng sinh log crash có mã.
                from sim import sim_realtime as _sr
                from core.method_adapter import MethodAdapter
                adapter = MethodAdapter(args.method, eps=args.eps,
                                        age_group=args.age_group,
                                        t_cap=args.t_cap)
                _sr.plan = adapter.wrap_plan(_sr.plan)
                log["config"]["method_adapter"] = adapter.provenance()
        # DV-H10 (phán quyết vòng kiểm chứng SOTA 2026-11-23 mục
        # 6): dự phòng hạt giống A* khi một lần gọi plan() dùng
        # > 20 vòng lặp DCA — CHỈ bật cho A0, là đối tượng của
        # deviation. A1–A3 đi đường hạt giống DV-H8 của adapter;
        # baseline Tier-1 không bật: giao diện hạt giống của
        # chúng đóng băng theo SPEC_cong_bang §5 (Ngo2023/MPCQP
        # có tiêu thụ seed_P qua initial_trajectory nhưng thay
        # hạt giống giữa episode sẽ là thay đổi chưa đăng ký vào
        # giao diện đã kiểm chứng; CBF2021 không tiêu thụ seed_P).
        out = run_episode(args.scenario, mode=args.mode, seed=args.seed,
                          prm=prm, time_model=args.time_model,
                          t_max=args.t_max, diag_sink=diag_sink,
                          anchor_sink=anchor_sink,
                          obs_mode=args.obs_mode,
                          seed_fallback=(args.method == "A0"))
        log["config"].update(out["config"])
        if adapter is not None:
            log["config"]["method_adapter"]["run_summary"] = \
                adapter.summary()
        if plan_calls is not None:
            log["plan_calls"] = plan_calls
        log["metrics"] = out["metrics"]
        log["cycles"] = out["cycles"]
        # Mã trạng thái episode §6.4 do vòng lặp quyết định
        log["status"] = out["metrics"]["status"]
    except Exception as exc:  # mọi lỗi đều phải có log; ngoại lệ = crash
        log["status"] = "crash"
        log["error"] = dict(code=_error_code(exc), type=type(exc).__name__,
                            message=str(exc), traceback=traceback.format_exc())
    # §7.2: không ghi đè — chạy lại tạo file có hậu tố lần chạy
    path = outdir / f"{run_id}.json"
    suffix = 2
    while path.exists():
        path = outdir / f"{run_id}_r{suffix}.json"
        suffix += 1
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    if diag_sink:
        # Log phụ chẩn đoán chu kỳ sập (schema riêng, không đụng
        # schema log chính): trạng thái neo + đầu vào plan() đầy đủ.
        diag_path = path.with_name(path.stem + "_diag.json")
        diag_path.write_text(json.dumps(
            dict(run_id=run_id, planner_file=log["planner_file"],
                 n_failed_cycles=len(diag_sink), records=diag_sink),
            indent=2, ensure_ascii=False))
        print(f"  [diag] {len(diag_sink)} chu kỳ sập -> {diag_path}")
    if anchor_sink:
        # Log phụ trạng thái neo MỌI lần gọi plan() (schema riêng,
        # không đụng schema log chính): tiền đề trong-hộp của Ô 1
        # DV-H4 trích xuất từ đây cho cả hai lô trước/sau.
        n_gray = sum(1 for r in anchor_sink
                     if r["box_class"] == "gray_zone")
        n_out = sum(1 for r in anchor_sink
                    if r["box_class"] == "out_of_box")
        anchor_path = path.with_name(path.stem + "_anchors.json")
        anchor_path.write_text(json.dumps(
            dict(run_id=run_id, planner_file=log["planner_file"],
                 box_definition=dict(
                     in_box="max_i |v0_i| <= v_max",
                     gray_zone="(v_max, v_max + 3e-3]",
                     out_of_box="> v_max + 3e-3",
                     source="QUYET_DINH_THIET_KE.md Quyết định 1 "
                            "§1.5 Ô 1 (DV-H4)"),
                 n_plan_calls=len(anchor_sink),
                 n_gray_zone=n_gray, n_out_of_box=n_out,
                 max_v0_abs=max(r["v_anchor_max_abs_comp"]
                                for r in anchor_sink),
                 records=anchor_sink),
            indent=2, ensure_ascii=False))
        print(f"  [anchors] {len(anchor_sink)} lần gọi plan() "
              f"(gray={n_gray}, out={n_out}) -> {anchor_path}")
    m = log["metrics"] or {}
    print(f"[{log['status']}] {run_id} -> {path}"
          + (f" | success={m.get('success')} safe={m.get('safe_success')} "
             f"collision={m.get('collision')} "
             f"min_clear={m.get('min_clearance_continuous')} "
             f"solve_p95={m.get('solve_time_p95')}" if m else
             f" | error={log['error']['code'] if log['error'] else ''}"))
    return 0 if log["status"] != "crash" else 1


if __name__ == "__main__":
    raise SystemExit(main())
