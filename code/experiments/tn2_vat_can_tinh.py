# -*- coding: utf-8 -*-
"""
THỰC NGHIỆM 2 — VẬT CẢN TĨNH, NHIỀU HÌNH DẠNG, 3 CẤP ĐỘ.

Kiểm tra cái gì: khả năng tránh vật cản TĨNH có hình dạng khác
nhau — cầu, trụ đứng, hộp, thanh ngang, tường có cổng — ở ba cấp
độ dễ / trung bình / khó. Khung bay như họ S của Bài 1: điểm đầu
(0, 0, 1,5), đích (10, 0, 1,5), biên an toàn d = 0,5 m.

QUAN TRỌNG — cách biểu diễn hình dạng: planner và khâu đo khe hở
của hạ tầng hiện tại chỉ hiểu HÌNH CẦU. Mọi hình dạng ở đây được
xấp xỉ bằng một TẬP CẦU CON (đúng tiền lệ ô F1-MS của giao thức):
  - cầu        : 1 cầu con (khớp hệt S1 của Bài 1 ở cấp dễ);
  - trụ đứng   : chồng cầu con dọc trục, bán kính cầu con lớn hơn
                 bán kính trụ một chút để hợp cầu BAO mặt trụ —
                 phình tối đa ~0,05 m (hướng bảo thủ);
  - hộp, tường : lưới cầu con bao kín hộp (tự kiểm bằng lấy mẫu
                 6 mặt hộp ngay khi dựng kịch bản) — hợp cầu phình
                 khỏi mặt hình thật tối đa ~0,4 m (hướng bảo thủ);
  - thanh ngang: dãy cầu con dọc trục thanh, bề mặt hợp cầu lẹm
                 vào tối đa ~0,05 m giữa hai tâm kề nhau.
Sai lệch xấp xỉ do khoảng cách cầu con quyết định; các khoảng
cách đã chọn ở trên cho sai lệch <= ~0,05 m trừ lưới hộp/tường
(~0,4 m phình ra, an toàn hơn hình thật). Khe hở ở cấp khó được
tính TRÊN BIỂU DIỄN CẦU CON: khe mặt-đối-mặt giữa hai hình là
1,2 m theo cầu con (hình thật rộng hơn vì cầu con phình ra).

Ba cấp độ:
  de         : từng hình riêng lẻ, kích thước chuẩn, lối đi rộng.
  trung_binh : từng hình riêng lẻ, to hơn ~1,2–1,4 lần tùy hình.
  kho        : cặp hình chừa khe 1,2 m ngay trên đường bay thẳng
               (chỉ vừa đủ qua với biên an toàn 0,5 m mỗi phía),
               cộng 2 episode ghép nhiều hình.

Cách chạy:
  python3 tn2_vat_can_tinh.py                   # 17 kịch bản, 7 phương pháp, 3 seed
  python3 tn2_vat_can_tinh.py --quick           # 1 kịch bản, 1 seed
  python3 tn2_vat_can_tinh.py --methods A0 A3 --seeds 5
  python3 tn2_vat_can_tinh.py --list            # liệt kê + số cầu con từng kịch bản

Đầu ra: results_tn2/ (hoặc --out) — JSON từng episode +
tong_hop_tn2.csv + bảng in ra màn hình.
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
# Dựng hình dạng bằng tập cầu con. Mọi hàm trả [(tâm (3,), r), ...]
# trong khung x = 5 (mặt phẳng chắn giữa đường bay), bay ở z = 1,5.
# ------------------------------------------------------------------
def _luoi_cau_hop(mins, maxs, r_s=0.9, a=0.5, h_max=1.0):
    """Lưới cầu con bao kín hộp [mins, maxs]: tâm lùi vào trong
    a = 0,5 m mỗi mặt, bước <= 1,0 m, bán kính cầu con 0,9 m.
    Với bộ số này, điểm xa cầu con nhất trên bề mặt hộp (góc hộp:
    a*sqrt(3) = 0,87; tâm mặt: sqrt(a^2 + (h/2)^2 + (h/2)^2) =
    0,87) đều nằm trong cầu con gần nhất -> hợp cầu BAO hộp thật,
    phình khỏi mặt hộp tối đa r_s - a = 0,4 m (hướng bảo thủ).
    Tự kiểm bằng lấy mẫu 6 mặt hộp; sai thì raise ngay."""
    truc = []
    for lo, hi in zip(mins, maxs):
        if hi - lo <= 2 * a:
            truc.append([0.5 * (lo + hi)])
        else:
            n = int(np.ceil((hi - lo - 2 * a) / h_max)) + 1
            truc.append(list(np.linspace(lo + a, hi - a, n)))
    tam = [np.array([x, y, z])
           for x in truc[0] for y in truc[1] for z in truc[2]]
    # Tự kiểm bao phủ: mẫu 13x13 trên từng mặt hộp.
    mat = np.array(tam)
    xau_nhat = -np.inf
    for ax in range(3):
        hai = [i for i in range(3) if i != ax]
        for gia_tri in (mins[ax], maxs[ax]):
            for u in np.linspace(mins[hai[0]], maxs[hai[0]], 13):
                for v in np.linspace(mins[hai[1]], maxs[hai[1]], 13):
                    p = np.empty(3)
                    p[ax], p[hai[0]], p[hai[1]] = gia_tri, u, v
                    d = np.min(np.linalg.norm(mat - p, axis=1)) - r_s
                    xau_nhat = max(xau_nhat, d)
    if xau_nhat > 1e-6:
        raise ValueError(f"luoi cau khong bao kin hop: lech {xau_nhat:.3f} m")
    return [(c, float(r_s)) for c in tam]


def hinh_cau(R, cy=0.0):
    """Một hình cầu bán kính R tại (5, cy, 1,5)."""
    return [(np.array([5.0, cy, 1.5]), float(R))]


def hinh_tru(R, H, cy=0.0):
    """Trụ đứng bán kính R, cao H (z từ 0 tới H), trục tại (5, cy).
    Cầu con trên trục, bước 0,6 m, bán kính sqrt(R^2 + 0,3^2):
    điểm giữa hai tâm kề trên mặt trụ và vành nắp trụ đều nằm
    đúng trên cầu con -> hợp cầu bao mặt trụ, phình hướng kính
    tối đa sqrt(R^2+0,09) - R (~0,05 m ở R = 0,8)."""
    r_s = float(np.hypot(R, 0.3))
    zs = list(np.arange(0.3, H - 0.3 + 1e-9, 0.6))
    if zs[-1] < H - 0.3 - 1e-9:
        zs.append(H - 0.3)
    return [(np.array([5.0, cy, z]), r_s) for z in zs]


def hinh_hop(sx, sy, sz, cy=0.0, cz=1.5):
    """Hộp kích thước (sx, sy, sz) tâm (5, cy, cz) — lưới cầu con."""
    mins = (5.0 - sx / 2, cy - sy / 2, cz - sz / 2)
    maxs = (5.0 + sx / 2, cy + sy / 2, cz + sz / 2)
    return _luoi_cau_hop(mins, maxs)


def hinh_thanh(L, cz=1.5):
    """Thanh ngang dọc trục y, dài L, tại (5, ·, cz): dãy cầu con
    r = 0,35 m, bước <= 0,35 m. Thanh thật hiểu là ống tròn bán
    kính 0,35 dọc trục; bề mặt hợp cầu lẹm vào tối đa ~0,05 m
    giữa hai tâm kề."""
    n = int(np.ceil(L / 0.35)) + 1
    ys = np.linspace(-L / 2, L / 2, n)
    return [(np.array([5.0, y, cz]), 0.35) for y in ys]


def hinh_tuong(W, dich=0.0):
    """Tường tại mặt x = 5, dày 0,4 m, y từ -4 tới 4, z từ 0 tới 3,
    chừa CỔNG rộng W (theo hình thật) tâm tại y = dich. Hai khối
    hai bên cổng là hộp — lưới cầu con như hinh_hop. Vì cầu con
    phình 0,4 m mỗi phía, cổng theo BIỂU DIỄN cầu con hẹp hơn
    cổng hình thật 0,8 m."""
    khoi = []
    for y_lo, y_hi in ((-4.0, dich - W / 2), (dich + W / 2, 4.0)):
        khoi += _luoi_cau_hop((4.8, y_lo, 0.0), (5.2, y_hi, 3.0))
    return khoi


def _fn_tinh(c, r):
    c = np.asarray(c, dtype=float)

    def g(t):
        return c.copy(), float(r)
    return g


def _dinh_nghia() -> dict:
    """Bảng kịch bản TN2: tên -> tập cầu con. Khe cấp khó = 1,2 m
    tính trên biểu diễn cầu con (mặt-đối-mặt)."""
    D = {}
    # --- cấp dễ: từng hình riêng lẻ, kích thước chuẩn
    D["TN2_de_cau"] = hinh_cau(1.2)
    D["TN2_de_tru"] = hinh_tru(0.8, 3.0)
    D["TN2_de_hop"] = hinh_hop(1.6, 2.4, 2.0)
    D["TN2_de_thanh"] = hinh_thanh(4.0)
    D["TN2_de_tuong"] = hinh_tuong(3.4)          # cổng biểu diễn 2,6 m
    # --- cấp trung bình: từng hình riêng lẻ, to hơn
    D["TN2_tb_cau"] = hinh_cau(1.7)
    D["TN2_tb_tru"] = hinh_tru(1.1, 3.4)
    D["TN2_tb_hop"] = hinh_hop(2.0, 2.8, 2.4)
    D["TN2_tb_thanh"] = hinh_thanh(5.5)
    D["TN2_tb_tuong"] = hinh_tuong(2.6)          # cổng biểu diễn 1,8 m
    # --- cấp khó: cặp hình chừa khe biểu diễn 1,2 m trên đường thẳng
    D["TN2_kho_cau"] = hinh_cau(1.2, -1.8) + hinh_cau(1.2, 1.8)
    D["TN2_kho_tru"] = hinh_tru(0.8, 3.0, -1.45) + hinh_tru(0.8, 3.0, 1.45)
    D["TN2_kho_hop"] = hinh_hop(1.6, 1.6, 2.0, -1.8) + hinh_hop(1.6, 1.6, 2.0, 1.8)
    D["TN2_kho_thanh"] = hinh_thanh(4.0, 0.55) + hinh_thanh(4.0, 2.45)
    D["TN2_kho_tuong"] = hinh_tuong(2.0, dich=0.5)  # cổng biểu diễn 1,2 m, lệch tâm
    # --- cấp khó: ghép nhiều hình
    D["TN2_kho_ghep1"] = hinh_tru(0.9, 3.2, -1.6) + hinh_thanh(4.5, 2.3)
    D["TN2_kho_ghep2"] = hinh_tuong(2.0) + [(np.array([7.5, 0.0, 1.5]), 1.0)]
    return D


_DINH_NGHIA = _dinh_nghia()
GHI_CHU = {
    "de": "cấp dễ — hình đơn lẻ, lối đi rộng",
    "tb": "cấp trung bình — hình đơn lẻ, to hơn",
    "kho": "cấp khó — khe biểu diễn 1,2 m / ghép nhiều hình",
}
MA = "tn2"
KICH_BAN = [(ten, ten, GHI_CHU[ten.split("_")[1]])
            for ten in _DINH_NGHIA]
T_MAX = None


def chuan_bi() -> None:
    for ten, cau_con in _DINH_NGHIA.items():
        dang_ky_kich_ban(ten, [_fn_tinh(c, r) for c, r in cau_con])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Thực nghiệm 2 — vật cản tĩnh, nhiều hình dạng, 3 cấp độ")
    ap.add_argument("--methods", nargs="+", default=PHUONG_PHAP_MAC_DINH,
                    choices=list(MAP_PHUONG_PHAP))
    ap.add_argument("--seeds", type=int, default=3,
                    help="số seed (bắt đầu từ 9000000); --quick ép 1")
    ap.add_argument("--quick", action="store_true",
                    help="kiểm khói: 1 seed + rút gọn số kịch bản")
    ap.add_argument("--out", default=None,
                    help="thư mục kết quả (mặc định results_tn2/ cạnh script)")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="chỉ liệt kê kịch bản rồi thoát")
    args = ap.parse_args()

    chuan_bi()
    if args.list_only:
        for nhan, ten, ghi_chu in KICH_BAN:
            print(f"{nhan:<28} runner={ten:<28} "
                  f"so_cau_con={len(_DINH_NGHIA[ten]):>3}  {ghi_chu}")
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
