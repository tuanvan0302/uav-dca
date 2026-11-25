# HƯỚNG DẪN 6 THỰC NGHIỆM ST-DCA

*Bộ script này là lớp trình bày/chạy do người dùng chốt ngày 2026-11-24:
6 thực nghiệm, mỗi thực nghiệm đúng MỘT script tự chứa, tên gọi đơn giản.
Bản này đặt trong cây refactor của người dùng (2026-11-24): script ở
`code/experiments/`, chạy qua `experiments/experiment_runner.py` và nối thẳng vào
`core/` + `baselines/` của người dùng cùng gói `sim/` do đội dựng.
Thư viện kịch bản 49 ô (họ S/D, Series A/B, họ F) và giao thức thực
nghiệm 2.0 bên dưới KHÔNG thay đổi.*

Thư mục: `code/experiments/`. Chạy bằng Python đã cài đủ phụ thuộc của
dự án (cvxpy, osqp, clarabel, numpy, scipy). Mọi script tự chèn thư
mục `code/` (cha của `experiments/`) vào sys.path theo đường dẫn
tương đối của chính script, nên giữ nguyên bố cục
`code/{core,baselines,sim,experiments,tests}` khi mang đi; chạy từ
thư mục làm việc nào cũng được. Các lệnh dưới đây viết từ thư mục
`code/`.

## Giao diện chung (giống hệt cả 6 script)

```
python3 experiments/tnX_....py [--methods A0 A1 A2 A3 NGO CBF MPCQP] [--seeds N] [--quick] [--out DIR] [--list]
```

- `--methods`: mặc định đủ 7 — A0 (Bài 1 gốc), A1 (+ biên H1), A2
  (+ tầng hình dạng H2), A3 (ST-DCA đầy đủ), NGO (Ngo 2023),
  CBF (Varun 2021), MPCQP (MPC-QP).
- `--seeds N`: mặc định 3; seed theo quy ước lô chính thức
  9000000, 9000001, ... A1/A3 luôn kèm trần T_cap = 2,0 s (DV-H7)
  như hướng dẫn lô chính thức.
- `--quick`: kiểm khói — 1 seed, chỉ kịch bản đầu tiên của script.
- `--list`: chỉ liệt kê kịch bản của script rồi thoát (không chạy).
- Mọi episode chạy ở chế độ mặc định của lô chính thức: dự báo
  kalman, mô hình thời gian realtime (không đóng băng thời gian).
- Đầu ra: thư mục `results_tnX/` cạnh script (hoặc `--out`):
  một JSON log cho mỗi episode (đúng định dạng log của runner,
  đủ provenance: planner file + sha256, code_commit, cấu hình),
  và `tong_hop_tnX.csv` — bảng (kịch bản × phương pháp): số
  episode, tỉ lệ thành công, số va chạm, khe hở nhỏ nhất trung
  bình (m), độ dài đường trung bình (m), jerk RMS trung bình,
  thời gian giải p95 trung bình (s), ghi chú. Bảng tương tự in ra
  màn hình khi chạy xong. Phương pháp không tương thích với một
  chế độ chạy thì bị bỏ qua kèm ghi chú, script không sập.

## Từng thực nghiệm

### TN1 — `tn1_khong_vat_can.py` — không vật cản (1 kịch bản)

Bay thẳng (0, 0, 1,5) → (10, 0, 1,5) không vật cản (S0_trong của
Bài 1). Mốc kiểm tra bay cơ bản: mọi phương pháp phải tới đích với
độ dài đường sát 10,000 m và jerk thấp — lệch đường ở đây là lỗi,
không phải đánh đổi an toàn.

**Ghi nhận lịch sử — lỗi thư viện đã sửa (2026-11-24, DV-H12):**
trước đợt sửa này, A1/A2/A3 từng crash trên kịch bản trống do hai
khiếm khuyết có sẵn của thư viện ở đường adapter khi danh sách
vật cản rỗng (đồng hồ DV-H7 chưa khởi tạo ở adapter;
`seed_max_viol = None` ở harness ghi log). Lỗi từng được kiểm
chứng qua CLI chính thức của runner: crash y hệt, không phải
lỗi của script. Trong cây này cả hai đã sửa: `core/method_adapter.py`
là bản adapter của người dùng đã được đội sửa lỗi đồng hồ
(DV-H12), và `sim/sim_realtime.py` ghi `seed_max_viol` = 0,0 khi
không có vật cản. TN1 nay chạy đủ 7 phương pháp, không crash vì
hai lỗi này; hồ sơ sửa và hồi quy ở
`exp/results/empty_obs_fix/FIX_REPORT.md` trong cây dự án của đội,
deviation ghi tại mục DV-H12 của sổ DEVIATIONS.md. Mọi kịch bản
có vật cản đều không dính hai lỗi này từ trước.

### TN2 — `tn2_vat_can_tinh.py` — vật cản tĩnh, nhiều hình, 3 cấp độ (17 kịch bản)

5 hình dạng × 3 cấp độ + 2 episode ghép. Hình dạng: cầu, trụ đứng,
hộp, thanh ngang, tường có cổng.

- **de** (dễ): từng hình riêng lẻ kích thước chuẩn — cầu R = 1,2
  (khớp hệt vật của S1), trụ R = 0,8 cao 3,0, hộp 1,6×2,4×2,0,
  thanh dài 4,0, tường cổng rộng.
- **trung_binh**: từng hình riêng lẻ, to hơn ~1,2–1,4 lần tùy hình.
- **kho** (khó): cặp hình chừa **khe 1,2 m** ngay trên đường bay
  thẳng (chỉ vừa đủ qua với biên an toàn 0,5 m mỗi phía), tường
  cổng 1,2 m lệch tâm, và 2 episode ghép nhiều hình
  (`TN2_kho_ghep1`, `TN2_kho_ghep2`).

Lưu ý hai episode ghép **không phải ô khe**: `TN2_kho_ghep1`
(trụ + thanh lệch cao độ) — đường bay thẳng có khe hở 0,47 m,
vi phạm nhẹ biên an toàn 0,5 m khoảng 0,03 m (không va chạm;
cúi xuống nhẹ là qua); `TN2_kho_ghep2` (tường cổng + một cầu
nằm ngay trên đường thẳng *sau* cổng) — đây là ô "qua cổng rồi
né vật", không phải ô luồn khe.

**Giới hạn trung thực của TN2 — đọc trước khi trích số:** hạ tầng
chỉ hiểu hình cầu, nên mọi hình dạng được xấp xỉ bằng **tập cầu
con** (tiền lệ ô F1-MS của giao thức), và khe hở đo trên chính
biểu diễn cầu con đó:

| Hình | Cách xấp xỉ | Sai lệch so với hình thật |
|---|---|---|
| cầu | 1 cầu con | 0 (khớp hệt) |
| trụ đứng | chồng cầu con dọc trục, R_cầu = √(R² + 0,3²) | hợp cầu bao mặt trụ, phình ≤ ~0,05 m |
| hộp | lưới cầu con R = 0,9, tâm lùi 0,5 m mỗi mặt (tự kiểm bao phủ bằng lấy mẫu 6 mặt khi dựng) | hợp cầu bao kín hộp, phình khỏi mặt ≤ 0,4 m |
| thanh ngang | dãy cầu con R = 0,35 dọc trục, bước ≤ 0,35 | bề mặt lẹm vào ≤ ~0,05 m giữa hai tâm |
| tường có cổng | hai khối hộp hai bên cổng, lưới cầu như hộp | như hộp; **cổng theo biểu diễn hẹp hơn cổng hình thật 0,8 m** |

Khe 1,2 m ở cấp khó tính **trên biểu diễn cầu con** (mặt cầu con
đối mặt); khe theo hình thật rộng hơn tương ứng. Mọi sai lệch
phình ra đều theo hướng bảo thủ (vật biểu diễn to hơn vật thật),
trừ độ lẹm ~0,05 m của thanh/trụ đã ghi ở trên.

### TN3 — `tn3_vat_can_dong.py` — vật cản động, không biến dạng (4 kịch bản)

Vật cầu chỉ di chuyển, bán kính không đổi; t_max = 40 s như họ D.
Hai kịch bản lấy từ registry của Bài 1 (D1 cắt ngang, D2 đối đầu)
và hai kịch bản định nghĩa ngay trong script (hai vật cắt ngang
lệch pha; một vật bay chéo) — ghi chú nguồn in kèm trong tổng hợp.

### TN4 — `tn4_flightbench.py` — mô phỏng theo FlightBench (24 kịch bản)

Series A nguyên vẹn: 8 cấp độ khó (2 → 20 vật cản tĩnh ngẫu
nhiên) × 3 lần lặp. **Mức trung thực:** chỉ *mượn thang độ khó và
bộ chỉ số* của FlightBench; trường vật do bộ sinh của dự án sinh
theo seed cố định (tái lập tuyệt đối), **không** tái tạo môi
trường/dữ liệu gốc của FlightBench.

### TN5 — `tn5_flyingtrust.py` — mô phỏng theo FLYINGTRUST (9 kịch bản)

Series B nguyên vẹn: 3 nền tảng bay (P1 hạn chế v_max = 2,0 /
u_max = 3,0; P2 tiêu chuẩn 3,0 / 4,0; P3 linh hoạt 5,0 / 8,0) ×
3 kịch bản nền (S2, S4, D1). **Mức trung thực:** chỉ *mượn chiều
biến thiên nền tảng* của FLYINGTRUST, không dùng môi trường/dữ
liệu gốc.

### TN6 — `tn6_benchmark.py` — benchmark 10 tình huống (thực nghiệm chủ lực)

10 tình huống động + biến dạng, tái dùng hình học họ D và họ F
trong thư viện (không viết lại hình học):

| Tình huống | Nội dung | Nguồn hình học |
|---|---|---|
| BN1 | một vật cầu cắt ngang | họ D: D1 |
| BN2 | một vật cầu đối đầu | họ D: D2 |
| BN3 | hai vật cắt ngang lệch pha | định nghĩa tại script |
| BN4 | vật biến dạng tại chỗ | họ D: D3 |
| BN5 | biến dạng + di chuyển | họ F: F5 |
| BN6 | thanh dài quay yaw ±35° | bán trục F1 + luật yaw F3 |
| BN7 | phình dị hướng giữ thể tích | họ F: F2 |
| BN8 | hành lang tĩnh, vật động bay dọc hành lang | S2 + vật động tại script |
| BN9 | hỗn hợp tĩnh + động + biến dạng (4 vật) | ghép tại script |
| BN10 | khe hẹp 1,2 m + vật biến dạng + vật cắt ngang (4 vật) | ghép tại script |

**Chế độ mặc định:** mọi phương pháp chạy trên **biểu diễn cầu
tương đương r_eq** (quy ước §2.7/Q6 của giao thức) để so sánh
ngang hàng. Hệ quả: BN6 ở chế độ này chỉ còn là một cầu nhỏ đứng
yên (r_eq của thanh rất nhỏ) — độ khó thật của BN6 nằm ở chế độ
ellipsoid.

**Cờ `--ellipsoid`** (chỉ A2/A3): chạy thêm một lượt quan sát
ellipsoid. **Giới hạn bắt buộc phải nhớ:** ở chế độ này runner
chưa bọc method_adapter nên **A3 chưa phải ST-DCA đầy đủ** — rho
là rho plug-in κ = 2 của tầng hình dạng, đầu vào planner của A3
trùng A2. Đừng trích kết quả ellipsoid như kết quả ST-DCA đầy đủ.
Ánh xạ: BN5 → `F5_ket_hop`, BN7 → `F2_phong_dihuong` (ellipsoid
thật từ registry); BN6 → chạy thay bằng `F3_xoay_yaw` (ellipsoid
quay yaw thật; thanh + yaw chưa có trong registry) — hàng tổng
hợp ghi nhãn **"BN6 (ellipsoid: F3)"**, và hai hàng BN6 cầu /
BN6 (ellipsoid: F3) là *hai vật khác nhau* (thanh F1 dài 4,0 m
dày 0,3 m so với ellipsoid F3 dài 3,2 m dày 1,0 m), không so
trực tiếp như một cặp biểu diễn; các BN còn lại toàn vật cầu
nên quan sát ellipsoid suy biến từ cầu khớp đúng vật thật.
A0/A1/NGO/CBF/MPCQP không chạy ở chế độ này (ghi chú bỏ qua
trong tổng hợp).

**Kiểm xuyên nhau giữa các vật:** toàn bộ 10 tình huống đã được
quét theo thời gian cho mọi cặp vật (min của khoảng cách tâm −
tổng bán kính; vật biến dạng lấy bán kính tương đương theo thời
gian) — **không cặp nào âm** trong cửa sổ episode, các vật
không bao giờ xuyên nhau. Để đạt tính chất này, bốn tình huống
ghép tại script đã chỉnh hình học tối thiểu so với bản đầu (rà
soát thiết kế của Ba, MB-1): BN3 cho hai vật cắt ngang ở hai
mặt x khác nhau (5 và 7,5) thay vì cùng mặt x = 5 khiến tâm
trùng khít; BN8 bỏ quỹ đạo D1 chạy xuyên hai cầu tường hành
lang — vật động nay bay dọc hành lang ngược chiều ở z = 2,8,
phía trên hàng cầu, vẫn lẹm vào biên an toàn 0,5 m của đường
bay thẳng (khe hở vật lý 0,3 m) nên UAV vẫn phải cơ động; BN9
dời mặt cắt của vật động (x = 6,5 → 6,7) và một cầu tĩnh
((8; 2,4) → (8,8; 2,4)); BN10 dời nhẹ mặt cắt của vật động
(x = 3 → 2,7).

## Thứ tự chạy gợi ý trên server

1. Kiểm khói từng script: `python3 experiments/tnX_....py --quick --methods A0`
   (cả 6 phải xong ít nhất 1 episode, trạng thái `ok`).
2. TN1 đủ 7 phương pháp (rất nhanh) — xác nhận mốc bay thẳng.
3. TN3, TN6 lượt mặc định — hai thực nghiệm động chính.
4. TN2, TN4, TN5 — các lô tĩnh/nền tảng (TN4 cấp cao và TN2 cấp
   khó nặng nhất về thời gian giải).
5. TN6 `--ellipsoid` cho A2/A3 khi cần xem tầng hình dạng.

Thời gian giải tuyệt đối phụ thuộc máy; trên máy ảo phát triển,
một episode động từng đo tới hàng chục giây mỗi chu kỳ vì tranh
chấp CPU — số tuyệt đối chỉ có nghĩa khi đo trên server của anh.
