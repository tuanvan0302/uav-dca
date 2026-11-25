# -*- coding: utf-8 -*-
"""
THỰC NGHIỆM 4 — MÔ PHỎNG THEO FLIGHTBENCH (Series A).

Kiểm tra cái gì: khả năng vượt trường vật cản tĩnh ngẫu nhiên ở
8 cấp độ khó tăng dần (số vật cản từ 2 tới 20), mỗi cấp 3 lần
lặp với trường vật khác nhau — tổng 24 kịch bản A_L1_r0 ..
A_L8_r2 của sổ đăng ký scenarios_pkg, dùng nguyên vẹn.

MỨC TRUNG THỰC (đã có hồ sơ trong giao thức): thực nghiệm này
MƯỢN thang độ khó và bộ chỉ số của FlightBench — KHÔNG tái tạo
môi trường, bản đồ hay dữ liệu gốc của FlightBench. Trường vật
cản do bộ sinh random_field của dự án sinh theo seed cố định
(1000 + 100*cấp + lần lặp), tái lập được tuyệt đối.

Cách chạy:
  python3 tn4_flightbench.py                    # 24 kịch bản, 7 phương pháp, 3 seed
  python3 tn4_flightbench.py --quick            # 1 kịch bản (A_L1_r0), 1 seed
  python3 tn4_flightbench.py --methods A0 A3 --seeds 5
  python3 tn4_flightbench.py --list

Đầu ra: results_tn4/ (hoặc --out) — JSON từng episode +
tong_hop_tn4.csv + bảng in ra màn hình.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Bootstrap: chèn thư mục code/ (cha của experiments/) vào
# sys.path — cây package của người dùng dùng import tuyệt đối
# core.*/baselines.*/sim.*. Script nằm ở code/experiments/,
# thư viện ở code/core, code/baselines, code/sim.
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_CODE = _HERE.parent.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sim import dynamic_scenarios     # noqa: E402
from sim import sim_realtime          # noqa: E402
from experiments import experiment_runner as _runner  # noqa: E402

# plan() nguyên bản của planner: runner bọc hàm này theo từng
# phương pháp (adapter/baseline) và không tự khôi phục — script
# chạy nhiều episode trong một tiến trình nên phải khôi phục
# trước mỗi episode để các phương pháp không nhiễm nhau.
_PLAN_GOC = sim_realtime.plan

MAP_PHUONG_PHAP = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "A3",
                   "NGO": "Ngo2023", "CBF": "CBF2021", "MPCQP": "MPCQP"}
PHUONG_PHAP_MAC_DINH = ["A0", "A1", "A2", "A3", "NGO", "CBF", "MPCQP"]
SEED_GOC = 9000000   # quy ước seed lô chính thức: 9000000, 9000001, ...
T_CAP_A1_A3 = 2.0    # trần T_cap của DV-H7 cho A1/A3 (lô chính thức)


def dang_ky_kich_ban(ten: str, truth_fns: list) -> None:
    """Đăng ký kịch bản riêng của script vào dynamic_scenarios —
    đúng cơ chế các driver khác vẫn dùng: _resolve_scenario của
    sim_realtime tra dynamic_scenarios trước tiên, nên kịch bản
    đăng ký tại đây đi qua đúng đường runner hiện hành. Khung
    điểm đầu/đích cố định (0,0,1,5) -> (10,0,1,5) như họ S/D/F."""
    dynamic_scenarios._SCENARIOS[ten] = list(truth_fns)


def chay_episode(phuong_phap: str, kich_ban: str, seed: int,
                 outdir: Path, obs_mode: str = "sphere",
                 t_max: float | None = None) -> dict | None:
    """Chạy MỘT episode qua main() của experiment_runner — cùng
    dispatch (adapter cho A1–A3, junction baseline cho NGO/CBF/
    MPCQP), cùng định dạng log JSON, A1/A3 kèm --t-cap 2,0.
    Trả log đọc lại từ file JSON; None nếu không thấy file."""
    ma = MAP_PHUONG_PHAP[phuong_phap]
    argv = ["experiment_runner", "--method", ma, "--scenario", kich_ban,
            "--seed", str(seed), "--mode", "kalman",
            "--time-model", "realtime", "--outdir", str(outdir)]
    if obs_mode == "ellipsoid":
        argv += ["--obs-mode", "ellipsoid"]
    if t_max is not None:
        argv += ["--t-max", str(t_max)]
    if ma in ("A1", "A3"):
        argv += ["--t-cap", str(T_CAP_A1_A3)]
    truoc = set(outdir.glob("*.json"))
    sim_realtime.plan = _PLAN_GOC
    argv_cu, sys.argv = sys.argv, argv
    try:
        _runner.main()
    finally:
        sys.argv = argv_cu
        sim_realtime.plan = _PLAN_GOC
    moi = [p for p in outdir.glob("*.json") if p not in truoc
           and "_diag" not in p.name and "_anchors" not in p.name]
    if not moi:
        return None
    return json.loads(moi[-1].read_text(encoding="utf-8"))


def _thong_ke(logs: list) -> dict:
    """Gộp các log episode của một ô (kịch bản × phương pháp).

    Quy ước tỉ lệ theo §6.4 (như stats_report.summarise_group):
    ti_le_thanh_cong = (số episode success trong các log có metrics) /
    (TOÀN BỘ log của ô — gồm cả log crash và log không metrics); chỉ để
    trống khi ô không có log nào. so_va_cham và các cột trung bình liên
    tục chỉ tính trên episode có metrics (đại lượng của episode crash
    không đo được)."""
    that = [lg for lg in logs if lg is not None]
    ok = [lg for lg in that if lg.get("metrics")]
    ma_crash = [lg["error"]["code"] for lg in that
                if not lg.get("metrics") and lg.get("error")]

    def tb(key):
        vals = [lg["metrics"][key] for lg in ok
                if lg["metrics"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    so_thanh_cong = sum(1 for lg in ok if lg["metrics"]["success"])
    return dict(
        so_episode=len(that),
        ti_le_thanh_cong=(so_thanh_cong / len(that)) if that else None,
        so_va_cham=(int(sum(lg["metrics"]["collision"] for lg in ok))
                    if ok else 0),
        khe_ho_tb=tb("min_clearance_continuous"),
        duong_tb=tb("path_length_m"),
        jerk_tb=tb("jerk_rms"),
        giai_p95_tb=tb("solve_time_p95"),
        ma_crash=ma_crash)


def ghi_tong_hop(ban_ghi: list, outdir: Path, ten_csv: str) -> None:
    """ban_ghi: [{kich_ban, phuong_phap, log, ghi_chu}] ->
    ghi CSV tong hop + in bang ra man hinh."""
    nhom: dict = {}
    for bg in ban_ghi:
        khoa = (bg["kich_ban"], bg["phuong_phap"])
        muc = nhom.setdefault(khoa, dict(logs=[], ghi_chu=set()))
        muc["logs"].append(bg["log"])
        if bg.get("ghi_chu"):
            muc["ghi_chu"].add(bg["ghi_chu"])
    rows = []
    for (kb, pp), muc in nhom.items():
        tk = _thong_ke(muc["logs"])
        ghi_chu = "; ".join(sorted(muc["ghi_chu"]))
        if tk["ma_crash"]:
            ghi_chu = (ghi_chu + "; " if ghi_chu else "") + \
                "crash: " + ",".join(str(c) for c in tk["ma_crash"])
        rows.append(dict(kich_ban=kb, phuong_phap=pp, ghi_chu=ghi_chu, **tk))

    def fmt(v, nd=3):
        return "" if v is None else f"{v:.{nd}f}"

    duong_csv = outdir / ten_csv
    with duong_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kich_ban", "phuong_phap", "so_episode",
                    "ti_le_thanh_cong", "so_va_cham",
                    "khe_ho_nho_nhat_tb_m", "do_dai_duong_tb_m",
                    "jerk_rms_tb", "thoi_gian_giai_p95_tb_s", "ghi_chu"])
        for r in rows:
            w.writerow([r["kich_ban"], r["phuong_phap"], r["so_episode"],
                        fmt(r["ti_le_thanh_cong"]), r["so_va_cham"],
                        fmt(r["khe_ho_tb"]), fmt(r["duong_tb"]),
                        fmt(r["jerk_tb"]), fmt(r["giai_p95_tb"]),
                        r["ghi_chu"]])
    print("\n================ TONG HOP ================")
    print(f"{'kich_ban':<28}{'phuong_phap':<16}{'ep':>3} "
          f"{'thanh_cong':>10} {'va_cham':>7} {'khe_ho_tb':>9} "
          f"{'duong_tb':>8} {'giai_p95':>8}")
    for r in rows:
        print(f"{r['kich_ban']:<28}{r['phuong_phap']:<16}"
              f"{r['so_episode']:>3} "
              f"{fmt(r['ti_le_thanh_cong']):>10} {r['so_va_cham']:>7} "
              f"{fmt(r['khe_ho_tb']):>9} {fmt(r['duong_tb']):>8} "
              f"{fmt(r['giai_p95_tb']):>8}"
              + (f"  [{r['ghi_chu']}]" if r["ghi_chu"] else ""))
    print(f"\nCSV: {duong_csv}")


# ------------------------------------------------------------------
# Series A nguyên vẹn từ sổ đăng ký: 8 cấp x 3 lần lặp.
# ------------------------------------------------------------------
MA = "tn4"
KICH_BAN = [(f"A_L{lv}_r{rep}", f"A_L{lv}_r{rep}",
             f"Series A cấp {lv}, lần lặp {rep} (registry)")
            for lv in range(1, 9) for rep in range(3)]
T_MAX = None


def chuan_bi() -> None:
    """TN4 không có kịch bản riêng — Series A lấy thẳng từ sổ
    đăng ký scenarios_pkg qua tên."""
    pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Thực nghiệm 4 — mô phỏng theo FlightBench (Series A)")
    ap.add_argument("--methods", nargs="+", default=PHUONG_PHAP_MAC_DINH,
                    choices=list(MAP_PHUONG_PHAP))
    ap.add_argument("--seeds", type=int, default=3,
                    help="số seed (bắt đầu từ 9000000); --quick ép 1")
    ap.add_argument("--quick", action="store_true",
                    help="kiểm khói: 1 seed + rút gọn số kịch bản")
    ap.add_argument("--out", default=None,
                    help="thư mục kết quả (mặc định results_tn4/ cạnh script)")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="chỉ liệt kê kịch bản rồi thoát")
    args = ap.parse_args()

    chuan_bi()
    if args.list_only:
        for nhan, ten, ghi_chu in KICH_BAN:
            print(f"{nhan:<28} runner={ten:<28} {ghi_chu}")
        print(f"tong: {len(KICH_BAN)} kich ban")
        return 0

    kich_ban = KICH_BAN[:1] if args.quick else KICH_BAN
    n_seed = 1 if args.quick else args.seeds
    seeds = [SEED_GOC + i for i in range(n_seed)]
    outdir = Path(args.out) if args.out else _HERE.parent / f"results_{MA}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{MA}: {len(kich_ban)} kich ban x {len(args.methods)} "
          f"phuong phap x {n_seed} seed -> {outdir}")

    ban_ghi = []
    for nhan, ten, ghi_chu in kich_ban:
        for pp in args.methods:
            for sd in seeds:
                log = chay_episode(pp, ten, sd, outdir, t_max=T_MAX)
                ban_ghi.append(dict(kich_ban=nhan, phuong_phap=pp,
                                    log=log, ghi_chu=ghi_chu))
    ghi_tong_hop(ban_ghi, outdir, f"tong_hop_{MA}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
