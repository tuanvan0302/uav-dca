# -*- coding: utf-8 -*-
"""
THỰC NGHIỆM 6 — BENCHMARK 10 TÌNH HUỐNG (động + biến dạng).

Thực nghiệm chủ lực: 10 tình huống BN1..BN10 ghép từ hình học
của họ D (động) và họ F (ellipsoid biến dạng) trong thư viện —
KHÔNG viết lại hình học, chỉ tái dùng truth_fns có sẵn:

  BN1  một vật cầu cắt ngang                (họ D: D1)
  BN2  một vật cầu đối đầu                  (họ D: D2)
  BN3  hai vật cắt ngang lệch pha           (định nghĩa tại script)
  BN4  vật biến dạng tại chỗ                (họ D: D3)
  BN5  vật vừa biến dạng vừa di chuyển      (họ F: F5)
  BN6  thanh dài quay yaw                   (họ F: trục của F1 + luật yaw của F3)
  BN7  vật phình dị hướng giữ thể tích      (họ F: F2)
  BN8  hành lang tĩnh, vật động bay dọc     (họ S: S2 + vật động tại script)
  BN9  hỗn hợp tĩnh + động + biến dạng      (ghép tại script, 4 vật)
  BN10 tổng hợp khó: khe hẹp + vật biến dạng + vật cắt ngang
                                            (ghép tại script, 4 vật)

KIỂM XUYÊN NHAU GIỮA CÁC VẬT: mọi cặp vật trong cả 10 tình
huống đã được quét theo thời gian (min của khoảng cách tâm −
tổng bán kính; vật biến dạng lấy bán kính tương đương theo
thời gian như truth_fns cho ra) — không cặp nào âm trong cửa
sổ episode. Bốn tình huống ghép tại script (BN3/BN8/BN9/BN10)
đã chỉnh hình học tối thiểu đúng để đạt tính chất này: bản
đầu có các cặp xuyên nhau (BN3 tâm trùng khít, BN8 vật động
xuyên hai cầu tường hành lang, BN9/BN10 sượt xuyên), đã sửa
sau rà soát thiết kế của Ba ngày 2026-11-24 — chi tiết từng
tình huống ở docstring của hàm dựng tương ứng bên dưới.

CHẾ ĐỘ CHẠY MẶC ĐỊNH: mọi phương pháp chạy trên biểu diễn CẦU
TƯƠNG ĐƯƠNG (r_eq = (a*b*c)^(1/3) tại tâm thật) — đúng quy ước
§2.7/Q6 của giao thức, để mọi phương pháp so sánh ngang hàng
trên cùng một thế giới cầu. Hệ quả phải biết: BN6 ở chế độ này
chỉ còn là một cầu nhỏ đứng yên (r_eq của thanh rất nhỏ) — độ
khó thật của BN6 chỉ xuất hiện ở chế độ ellipsoid dưới đây.

Cờ --ellipsoid: A2/A3 chạy thêm một lượt ở chế độ quan sát
ellipsoid qua hook obs_mode của runner. GIỚI HẠN ĐÃ BIẾT, ĐỌC
KỸ: ở chế độ này runner KHÔNG bọc method_adapter, nên A3 chưa
phải ST-DCA đầy đủ — rho trong planner là rho plug-in kappa = 2
của tầng hình dạng và đầu vào planner của A3 trùng A2. Kết quả
ellipsoid chỉ để xem tầng hình dạng hoạt động thế nào, KHÔNG
được trích như kết quả ST-DCA đầy đủ. Ánh xạ kịch bản ở chế độ
ellipsoid: BN5 -> F5_ket_hop và BN7 -> F2_phong_dihuong (ellipsoid
thật từ sổ đăng ký); BN6 -> chạy THAY bằng F3_xoay_yaw (ellipsoid
quay yaw thật của họ F — thanh dài + yaw chưa có trong sổ đăng
ký), hàng tổng hợp của lượt này ghi nhãn "BN6 (ellipsoid: F3)";
lưu ý BN6 cầu và BN6 (ellipsoid: F3) là HAI VẬT KHÁC NHAU (thanh
F1 dài 4,0 m dày 0,3 m so với ellipsoid F3 dài 3,2 m dày 1,0 m),
không so trực tiếp như một cặp biểu diễn; các BN còn lại vốn
toàn vật cầu nên quan sát ellipsoid suy
biến từ cầu khớp đúng vật thật. A0/A1/NGO/CBF/MPCQP không chạy
ở chế độ ellipsoid (A1 và baseline bị runner từ chối tường minh;
A0 không thuộc phạm vi của cờ này).

Cách chạy:
  python3 tn6_benchmark.py                      # 10 tình huống, 7 phương pháp, 3 seed
  python3 tn6_benchmark.py --quick              # 1 tình huống, 1 seed
  python3 tn6_benchmark.py --ellipsoid          # thêm lượt ellipsoid cho A2/A3
  python3 tn6_benchmark.py --methods A0 A3 --seeds 5
  python3 tn6_benchmark.py --list

Đầu ra: results_tn6/ (hoặc --out) — JSON từng episode +
tong_hop_tn6.csv + bảng in ra màn hình (hàng ellipsoid ghi
phương pháp dạng "A2 (ellipsoid)" / "A3 (ellipsoid)").
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
from sim import scenarios as _static_sc  # noqa: E402  (họ S của Bài 1)
from sim import sim_realtime          # noqa: E402
from experiments import experiment_runner as _runner  # noqa: E402
from sim.scenarios_pkg import family_f  # noqa: E402

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
# 10 tình huống BN. Mỗi mục: (truth_fns cầu, tên registry cho chế
# độ ellipsoid hoặc None, ghi chú nguồn hình học).
# ------------------------------------------------------------------
_D = dynamic_scenarios._SCENARIOS


def _fn_tinh(c, r):
    c = np.asarray(c, dtype=float)

    def g(t):
        return c.copy(), float(r)
    return g


def _bn3():
    """Hai vật r = 1,0 cắt ngang ngược chiều ở HAI mặt x khác
    nhau: vật 1 tại x = 5 qua trục bay ở t = 4 s, vật 2 tại
    x = 7,5 qua trục bay ở t = 5,5 s (lệch pha cả thời gian).
    Hai mặt cắt cách nhau 2,5 m > tổng bán kính 2,0 m nên hai
    vật không bao giờ xuyên nhau (khe tâm trừ tổng bán kính
    nhỏ nhất = +0,5 m). Bản đầu đặt cả hai vật cùng mặt x = 5
    với tốc độ bằng nhau khiến tâm trùng khít tại t = 4,75 s
    (xuyên −2,0 m) — đã sửa sau rà soát MB-1 của Ba."""
    def f1(t):
        return np.array([5.0, -4.0 + 1.0 * t, 1.5]), 1.0

    def f2(t):
        return np.array([7.5, 5.5 - 1.0 * t, 1.5]), 1.0
    return [f1, f2]


def _bn6_cau_tuong_duong():
    """BN6 ở giao diện cầu: thanh của F1 (bán trục 0,15; 2,0; 0,15)
    quay yaw theo luật F3 — hình chiếu cầu tương đương r_eq =
    (a*b*c)^(1/3) tại tâm thật, đúng hàm sphere_eq_fns của họ F
    áp cho hàm ellipsoid định nghĩa tại đây."""
    def ell(t):
        return (np.array([5.0, 0.0, 1.5]),
                np.array([0.15, 2.0, 0.15]),
                35.0 * (np.pi / 180.0) * np.sin(2.0 * np.pi * t / 10.0))

    def g(t):
        o, a, _ = ell(t)
        return o, float(np.cbrt(a[0] * a[1] * a[2]))
    return [g]


def _bn8():
    """Hành lang tĩnh của S2 (6 cầu so le, giữ nguyên) + một vật
    động bay DỌC hành lang ngược chiều (-x) tại y = 0, z = 2,8,
    tốc độ 0,9 m/s, r = 1,0. Cao độ 2,8 đặt vật phía trên hàng
    cầu hành lang (đỉnh cầu z = 2,6): khoảng cách tâm nhỏ nhất
    tới mỗi cầu tĩnh là sqrt(1,8^2 + 1,3^2) = 2,22 m > tổng bán
    kính 2,1 m (dư +0,12 m) nên vật KHÔNG xuyên cầu tường như
    bản đầu (bản đầu mượn quỹ đạo D1 chạy dọc mặt x = 5 tại
    z = 1,5, đâm xuyên hai cầu (4,5; -1,8) và (6; 1,8) — đã sửa
    sau rà soát MB-1 của Ba). Vật vẫn cắt đường bay của UAV
    theo nghĩa biên an toàn: nó bay ngược chiều ngay trên trục
    hành lang, khe hở vật lý tới đường bay thẳng (z = 1,5) chỉ
    0,3 m < biên 0,5 m, nên UAV vẫn phải cơ động khi gặp."""
    def dong(t):
        return np.array([12.0 - 0.9 * t, 0.0, 2.8]), 1.0
    tinh = [_fn_tinh(c, r)
            for c, r in _static_sc.scenario("S2_hanh_lang")["obstacles"]]
    return tinh + [dong]


def _bn9():
    """Hỗn hợp 4 vật: 2 tĩnh + 1 biến dạng (D3) + 1 cắt ngang.
    Hình học đã chỉnh để không cặp nào xuyên nhau (bản đầu vật
    cắt ngang xuyên cầu tĩnh (8; 2,4) −0,5 m và sượt vật biến
    dạng −0,05 m — rà soát MB-1 của Ba): mặt cắt của vật động
    dời từ x = 6,5 sang x = 6,7 (cách tâm vật biến dạng 1,7 m,
    đủ cho tổng bán kính tại pha gặp nhau) và cầu tĩnh thứ hai
    dời từ (8; 2,4) sang (8,8; 2,4) (mặt cắt cách cầu 2,1 m >
    tổng bán kính 2,0 m). Vai trò 4 vật giữ nguyên."""
    def cat(t):
        return np.array([6.7, -4.0 + 0.9 * t, 1.5]), 0.9
    return [_fn_tinh([3.0, -2.2, 1.5], 1.0),
            _fn_tinh([8.8, 2.4, 1.5], 1.1),
            *_D["D3_bien_dang"],
            cat]


def _bn10():
    """Tổng hợp khó: cổng hai cầu r = 1,2 tại (5; ±1,8; 1,5) chừa
    khe 1,2 m + vật biến dạng kiểu D3 tại (7,5; 0; 1,5) + vật
    cắt ngang tại x = 2,7. Mặt cắt x = 2,7 cách mặt cổng x = 5
    là 2,3 m > tổng bán kính 2,1 m nên vật cắt không sượt xuyên
    hai cầu cổng (bản đầu tại x = 3 sượt −0,1 m mỗi cầu — rà
    soát MB-1 của Ba)."""
    def bien_dang(t):
        return (np.array([7.5, 0.0, 1.5]),
                1.0 + 0.35 * np.sin(2.0 * np.pi * t / 6.0))

    def cat(t):
        return np.array([2.7, -4.0 + 1.0 * t, 1.5]), 0.9
    return [_fn_tinh([5.0, -1.8, 1.5], 1.2),
            _fn_tinh([5.0, 1.8, 1.5], 1.2),
            bien_dang,
            cat]


# ten -> (truth_fns, ten registry ellipsoid | None, ghi chú nguồn)
_BN = {
    "BN1": (list(_D["D1_cat_ngang"]), None,
            "một vật cầu cắt ngang — họ D: D1_cat_ngang"),
    "BN2": (list(_D["D2_doi_dau"]), None,
            "một vật cầu đối đầu — họ D: D2_doi_dau"),
    "BN3": (_bn3(), None,
            "hai vật cắt ngang lệch pha — định nghĩa tại script"),
    "BN4": (list(_D["D3_bien_dang"]), None,
            "vật biến dạng tại chỗ — họ D: D3_bien_dang"),
    "BN5": (family_f.sphere_eq_fns("F5_ket_hop"), "F5_ket_hop",
            "biến dạng + di chuyển — họ F: F5 (cầu r_eq)"),
    "BN6": (_bn6_cau_tuong_duong(), "F3_xoay_yaw",
            "thanh dài quay yaw — trục F1 + luật yaw F3 (cầu r_eq)"),
    "BN7": (family_f.sphere_eq_fns("F2_phong_dihuong"),
            "F2_phong_dihuong",
            "phình dị hướng giữ thể tích — họ F: F2 (cầu r_eq)"),
    "BN8": (_bn8(), None,
            "hành lang tĩnh + vật động bay dọc hành lang — họ S: S2 "
            "+ vật động tại script"),
    "BN9": (_bn9(), None,
            "hỗn hợp tĩnh + động + biến dạng, 4 vật — ghép tại script"),
    "BN10": (_bn10(), None,
             "khe hẹp + biến dạng + cắt ngang, 4 vật — ghép tại script"),
}

MA = "tn6"
KICH_BAN = [(ten, ten, _BN[ten][2]) for ten in
            ["BN1", "BN2", "BN3", "BN4", "BN5",
             "BN6", "BN7", "BN8", "BN9", "BN10"]]
T_MAX = None  # 60 s mặc định (tên không bắt đầu bằng "D")


def chuan_bi() -> None:
    for ten, (fns, _ell, _note) in _BN.items():
        dang_ky_kich_ban(ten, fns)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Thực nghiệm 6 — benchmark 10 tình huống (động + biến dạng)")
    ap.add_argument("--methods", nargs="+", default=PHUONG_PHAP_MAC_DINH,
                    choices=list(MAP_PHUONG_PHAP))
    ap.add_argument("--seeds", type=int, default=3,
                    help="số seed (bắt đầu từ 9000000); --quick ép 1")
    ap.add_argument("--quick", action="store_true",
                    help="kiểm khói: 1 seed + rút gọn số kịch bản")
    ap.add_argument("--out", default=None,
                    help="thư mục kết quả (mặc định results_tn6/ cạnh script)")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="chỉ liệt kê kịch bản rồi thoát")
    ap.add_argument("--ellipsoid", action="store_true",
                    help="A2/A3 chạy thêm lượt quan sát ellipsoid "
                         "(xem giới hạn trong docstring đầu file)")
    args = ap.parse_args()

    chuan_bi()
    if args.list_only:
        for nhan, ten, ghi_chu in KICH_BAN:
            ell = _BN[ten][1] or "(suy biến từ cầu)"
            print(f"{nhan:<28} runner={ten:<28} ellipsoid={ell:<22} "
                  f"{ghi_chu}")
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
    # --- lượt mặc định: biểu diễn cầu tương đương, mọi phương pháp
    for nhan, ten, ghi_chu in kich_ban:
        for pp in args.methods:
            for sd in seeds:
                log = chay_episode(pp, ten, sd, outdir, t_max=T_MAX)
                ban_ghi.append(dict(kich_ban=nhan, phuong_phap=pp,
                                    log=log, ghi_chu=ghi_chu))
    # --- lượt ellipsoid (chỉ khi --ellipsoid): chỉ A2/A3
    if args.ellipsoid:
        print("[TN6] Lượt ellipsoid: chỉ A2/A3 — A3 ở chế độ này "
              "CHƯA phải ST-DCA đầy đủ (runner không bọc adapter, "
              "rho plug-in kappa=2, đầu vào trùng A2).")
        for nhan, ten, _gc in kich_ban:
            ten_ell = _BN[ten][1] or ten
            if _BN[ten][1]:
                chu_ell = (f"ellipsoid thật qua registry: {ten_ell}"
                           if ten != "BN6" else
                           "ellipsoid: chạy THAY bằng F3_xoay_yaw "
                           "(thanh + yaw chưa có trong registry)")
            else:
                chu_ell = "ellipsoid suy biến từ cầu (vật vốn là cầu)"
            for pp in ("A2", "A3"):
                if pp not in args.methods:
                    continue
                for sd in seeds:
                    log = chay_episode(pp, ten_ell, sd, outdir,
                                       obs_mode="ellipsoid", t_max=T_MAX)
                    ban_ghi.append(dict(
                        kich_ban=("BN6 (ellipsoid: F3)" if ten == "BN6"
                                  else nhan),
                        phuong_phap=f"{pp} (ellipsoid)",
                        log=log, ghi_chu=chu_ell))
        for pp in args.methods:
            if pp not in ("A2", "A3"):
                ban_ghi.append(dict(
                    kich_ban="(tất cả)", phuong_phap=f"{pp} (ellipsoid)",
                    log=None,
                    ghi_chu="bỏ qua ở chế độ ellipsoid: chỉ A2/A3 "
                            "được chạy (A1/baseline bị runner từ chối)"))
    ghi_tong_hop(ban_ghi, outdir, f"tong_hop_{MA}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
