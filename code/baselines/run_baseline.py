# -*- coding: utf-8 -*-
"""
run_baseline.py — chạy MỘT episode cho một baseline Tier-1 và ghi log
JSON (giao thức §7.2).

Cách nối: driver nạp sim.sim_realtime rồi thay `sim_realtime.plan` bằng
plan() của baseline, bọc thêm một lớp ghi metadata từng lần gọi. Việc thay
thế chỉ có hiệu lực trong tiến trình của driver. Log ghi đường dẫn file
baseline đã nạp ở trường `planner_file`.

Kịch bản tĩnh: run_episode chỉ biết họ động của sim.dynamic_scenarios;
driver đăng ký thêm các kịch bản tĩnh của sim.scenarios vào registry động
TRONG TIẾN TRÌNH (hàm sự thật nền hằng số theo thời gian).

Phụ thuộc: gói sim/ (nằm ngoài code/).

Ví dụ (từ gốc code/):
  python3 -m baselines.run_baseline --method CBF2021 \
      --scenario D1_cat_ngang --seed 9000000 --mode kalman \
      --time-model frozen --outdir <thư_mục_log>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from baselines import METHODS
from core.dca_planner import PlannerParams
from sim import dynamic_scenarios, sim_realtime
from sim import scenarios as static_scenarios

_HERE = Path(__file__).resolve().parent  # code/baselines

PROTOCOL = "ST-DCA-EXP"
PROTOCOL_VERSION = "1.2"


def register_static_scenarios() -> None:
    """Đăng ký kịch bản tĩnh của scenarios.py vào registry động
    (in-process). Chỉ đăng ký kịch bản có start/goal trùng họ động
    (S0, S1, S3): run_episode lấy start/goal từ dynamic_scenarios
    nên S2/S4 (đích khác) bị bỏ qua tường minh thay vì chạy sai."""
    for name in static_scenarios.ALL:
        if name in dynamic_scenarios._SCENARIOS:
            continue
        sc = static_scenarios.scenario(name)
        if not (
            np.allclose(sc["start"], dynamic_scenarios.START)
            and np.allclose(sc["goal"], dynamic_scenarios.GOAL)
        ):
            continue
        fns = []
        for center, radius in sc["obstacles"]:
            c = np.asarray(center, dtype=float)
            r = float(radius)
            fns.append(lambda t, c=c, r=r: (c.copy(), r))
        dynamic_scenarios._SCENARIOS[name] = fns


def environment_info() -> dict:
    """Thông tin môi trường chạy (Python, nền tảng, phiên bản thư viện
    số, số luồng BLAS) để ghi vào log."""
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
        numpy=ver("numpy"),
        scipy=ver("scipy"),
        cvxpy=ver("cvxpy"),
        osqp=ver("osqp"),
        ecos=ver("ecos"),
        clarabel=ver("clarabel"),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
    )


def collect_code_files(baseline_file: Path) -> dict:
    """Đường dẫn các file quyết định hành vi của episode: các module
    sim đã nạp, obstacle_predict, dca_planner, file baseline, common.py và
    chính driver này. Trả {tên file: {path}}."""
    from core import obstacle_predict

    files = {}
    for mod in (sim_realtime, dynamic_scenarios, obstacle_predict):
        p = Path(mod.__file__).resolve()
        files[p.name] = p
    from core import dca_planner

    files["dca_planner.py"] = Path(dca_planner.__file__).resolve()
    files[baseline_file.name] = baseline_file
    files["common.py"] = _HERE / "common.py"
    files[Path(__file__).name] = Path(__file__).resolve()
    return {name: dict(path=str(files[name])) for name in sorted(files)}


def main() -> int:
    """Điểm vào CLI: chạy một episode, ghi log JSON.

    Returns:
        Mã thoát của tiến trình: 0 nếu episode không crash, 1 nếu crash.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="kalman", choices=["truth", "kalman", "frozen"])
    ap.add_argument("--time-model", default="realtime", choices=["realtime", "frozen"])
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    register_static_scenarios()
    module, config_id = METHODS[args.method]
    baseline_file = Path(module.__file__).resolve()

    # Lớp bọc ghi metadata từng lần gọi plan() + checksum dự báo.
    plan_calls: list = []

    def recording_plan(p0, v0, goal, obstacles, prm, seed_P=None):
        res = module.plan(p0, v0, goal, obstacles, prm, seed_P=seed_P)
        plan_calls.append(
            dict(
                call_index=len(plan_calls),
                iterations=int(res.iterations),
                total_slack=float(res.total_slack),
                success_internal=bool(res.success),
                solve_time=float(res.solve_time),
                min_clearance_plan=float(res.min_clearance),
                meta=res.meta,
            )
        )
        return res

    sim_realtime.plan = recording_plan

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"{args.method}_{args.scenario}_{args.mode}_{args.time_model}_seed{args.seed}"
    )
    commit_files = collect_code_files(baseline_file)
    env = environment_info()
    prm = PlannerParams()  # tham số harness y hệt A0 (SPEC_cong_bang §2)
    log = dict(
        run_id=run_id,
        timestamp_utc=datetime.now(UTC).isoformat(),
        protocol=PROTOCOL,
        protocol_version=PROTOCOL_VERSION,
        code_files=commit_files,
        planner_file=dict(path=str(baseline_file), origin="baseline_module"),
        machine=env,
        env="sim",
        status="crash",
        error=None,
        method=args.method,
        method_description=config_id,
        scenario=args.scenario,
        mode=args.mode,
        time_model=args.time_model,
        seed=args.seed,
        environment=env,
        config=None,
        metrics=None,
        cycles=None,
        plan_calls=None,
    )
    diag_sink: list = []
    try:
        cfg_obj = module.__dict__.get(
            {"Ngo2023": "NgoConfig", "CBF2021": "CbfConfig", "MPCQP": "MpcqpConfig"}[
                args.method
            ]
        )()
        log["config"] = dict(
            planner=asdict(prm), baseline=asdict(cfg_obj), baseline_config_id=config_id
        )
        out = sim_realtime.run_episode(
            args.scenario,
            mode=args.mode,
            seed=args.seed,
            prm=prm,
            time_model=args.time_model,
            t_max=args.t_max,
            diag_sink=diag_sink,
        )
        log["config"].update(out["config"])
        log["metrics"] = out["metrics"]
        log["cycles"] = out["cycles"]
        log["plan_calls"] = plan_calls
        log["status"] = out["metrics"]["status"]
    except Exception as exc:
        log["status"] = "crash"
        raw = exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)
        log["error"] = dict(
            code=raw.split(":", 1)[0].strip().strip("'\"") or type(exc).__name__,
            type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
        )
    path = outdir / f"{run_id}.json"
    suffix = 2
    while path.exists():
        path = outdir / f"{run_id}_r{suffix}.json"
        suffix += 1
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    if diag_sink:
        diag_path = path.with_name(path.stem + "_diag.json")
        diag_path.write_text(
            json.dumps(
                dict(
                    run_id=run_id,
                    planner_file=log["planner_file"],
                    n_failed_cycles=len(diag_sink),
                    records=diag_sink,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"  [diag] {len(diag_sink)} failed cycles -> {diag_path}")
    m = log["metrics"] or {}
    print(
        f"[{log['status']}] {run_id} -> {path}"
        + (
            f" | success={m.get('success')} safe={m.get('safe_success')} "
            f"collision={m.get('collision')} "
            f"min_clear={m.get('min_clearance_continuous')} "
            f"solve_p95={m.get('solve_time_p95')}"
            if m
            else f" | error={log['error']['code'] if log['error'] else ''}"
        )
    )
    return 0 if log["status"] != "crash" else 1


if __name__ == "__main__":
    raise SystemExit(main())
