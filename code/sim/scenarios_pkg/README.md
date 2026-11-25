# Gói kịch bản benchmark ST-DCA (`scenarios_pkg/`)

Bộ kịch bản hoàn chỉnh cho gói so sánh SOTA, theo **giao thức thực
nghiệm ST-DCA bản đóng băng 1.2** (`exp/protocol/GIAO_THUC_THUC_NGHIEM.md`,
§2). Mọi kịch bản được chọn **theo tên** qua một sổ đăng ký duy nhất;
mọi kịch bản đều sinh từ seed/quy tắc tường minh và có **sự thật nền
dạng hàm truy vấn được theo thời gian** (không phải bảng lấy mẫu cố
định).

> Tên gói là `scenarios_pkg` (không phải `scenarios`) vì cây code
> đã có module `scenarios.py` của Bài 1 (nay ở
> `src/legacy_scenarios/`) — gói mới không được phép che khuất
> module cũ mà nhiều script đang import.

## Cấu trúc

| File | Vai trò |
|---|---|
| `registry.py` | Sổ đăng ký thống nhất: `get_scenario(name)`, `all_names()`, `resolve_for_runner(name)` |
| `family_f.py` | Họ F mới (F1–F5 ellipsoid + ô minh họa F1-MS), sự thật nền `(o(t), a(t), ψ(t))` |
| `validate.py` | Kiểm tra tính hợp lệ **độc lập cấu hình** (§2.5(i)(ii)) bằng A\* lưới mịn 0,1 m |
| `runner_smoke.py` | Chứng minh tích hợp: chạy thử A0 đóng-băng-thời-gian trên 1 kịch bản loạt A + 1 kịch bản họ F qua runner |
| `validation_results.json` | Kết quả kiểm hợp lệ của mọi kịch bản (sinh bởi `validate.py`) |
| `smoke_results/runner_smoke.json` | Kết quả chạy thử tích hợp (seed dải thử 9 000 000+, §3.3) |

Các họ S / D / loạt A / loạt B **tái dùng nguyên văn** module của
Bài 1 (`scenarios.py`, `dynamic_scenarios.py`,
`benchmark_scenarios.py`) — nguồn sự thật cho tham số kịch bản là
code, theo quy ước của §2. Sổ đăng ký chỉ bọc chúng vào giao diện
chung, không sửa một tham số nào.

## Giao diện kịch bản chuẩn

`get_scenario(name)` trả về:

- `start`, `goal`, `t_max` (họ D: 40 s; S / A / B / F: 60 s — §2);
- `truth_fns`: danh sách hàm `t -> (tâm (3,), bán kính)` — giao diện
  cầu mà `sim_realtime` / `obstacle_predict` tiêu thụ. Với họ F đây
  là **hình chiếu cầu thể-tích-tương-đương** `r_eq = (abc)^{1/3}`
  theo quyết định Q6 (§2.7) — giao diện của A0/A1 trên họ F;
- `ellipsoid_truth_fns` (chỉ họ F): hàm `t -> (tâm, bán trục (3,),
  yaw)` — sự thật nền ellipsoid đầy đủ cho A2/A3 và cho khâu đánh
  giá khe hở ellipsoid thật;
- `truth_fns_bounding` (chỉ họ F): hình chiếu **cầu bao**
  `r = a_max(t)` — giao diện của cấu hình A0b (§2.7);
- `obstacles` (kịch bản tĩnh), `platform` (chỉ loạt B:
  `(tên, v_max, u_max)`), `meta` (nguồn gốc, seed hình học).

Tổng số: **49 tên kịch bản** = 5 họ S + S1L + 4 họ D + 24 loạt A
(8 cấp × 3 rep) + 9 loạt B (3 nền tảng × {S2, S4, D1}) + 5 họ F +
F1-MS.

## Các họ kịch bản

### Họ S — tĩnh (§2.1, `scenarios.py` của Bài 1)

S0 trống; S1 một cầu r = 1,2 tại (5; 0; 1,5); S2 hành lang 6 cầu
r = 1,1 (đích (12, 0, 1,5) — theo code, lệch văn bản Bài 1 đã ghi
nhận ở §2); S3 lối hẹp (khe hình học 0,9 m); S4 trường ngẫu nhiên
`default_rng(7)` — **danh sách vật cản cụ thể đã in và đóng băng**
ở cuối tài liệu này (yêu cầu của §2.1 cho GĐ3). **S1L_thap** là mục
mới của giao thức: như S1 nhưng điểm đầu (0, 0, 0,9), chỉ cho A4.

### Họ D — động và biến dạng cầu (§2.2, `dynamic_scenarios.py`)

D1 cắt ngang (v = 0,9 m/s), D2 đối đầu lệch trục (1,1 m/s), D3 bán
kính dao động 0,65 ↔ 1,35 m (chu kỳ 6 s), D4 = D1 + D3. Sự thật nền
là hàm解析 theo t, truy vấn được tại mọi thời điểm.

### Loạt A — thang độ khó kiểu FlightBench (§2.3)

8 cấp, số vật cầu 2/4/6/8/10/12/16/20, 3 lần lặp mỗi cấp, sinh bởi
`default_rng(1000 + 100·cấp + rep)`, tâm x ∈ [1,2; 8,8],
y ∈ [−2,8; 2,8], z ∈ [0,9; 3,0], r ∈ [0,6; 1,2], loại tâm cách
đầu/đích < r + 1,2. Tên: `A_L{cấp}_r{rep}`. Chế độ thông tin: truth.

### Loạt B — biến thiên nền tảng kiểu FLYINGTRUST (§2.4)

3 cấp năng lực (P1: v_max/u_max = 2/3; P2: 3/4 — cấu hình gốc;
P3: 5/8) × kịch bản {S2, S4, D1} × chế độ kalman. Tên:
`B_{nền tảng}_{S2|S4|D1}`. Khác họ khác, mục loạt B mang thêm khóa
`platform` — khi chạy, (v_max, u_max) của nền tảng phải được truyền
vào `PlannerParams` của lần chạy (runner hiện chưa có cờ cho việc
này; xem mục Tích hợp).

### Họ F — ellipsoid biến dạng (§2.5, `family_f.py`) — MỚI

Benchmark công khai cho vật biến dạng chưa tồn tại; đây là bộ kịch
bản tự công bố của đề tài. Điểm đầu (0, 0, 1,5), đích (10, 0, 1,5),
t_max = 60 s, một ellipsoid `(o(t), a(t), ψ(t))`:

| ID | Sự thật nền |
|---|---|
| F1_thanh_dai | Thanh tĩnh: tâm (5; 0; 1,5), bán trục (0,15; 2,0; 0,15) — dài 4 m ngang đường bay, dày 0,3 m; ψ = 0. r_eq ≈ 0,356 m (hụt ~1,64 m dọc trục dài); cầu bao r = 2,0 |
| F2_phong_dihuong | Tĩnh tại (5, 0, 1,5); s(t) = (1−cos(2πt/12))/2; a_x = 1+0,8s (→1,8); a_y = a_z = 1/√a_x (bảo toàn thể tích abc = 1) |
| F3_xoay_yaw | Tĩnh tại (5, 0, 1,5); bán trục (1,6; 0,5; 0,5); ψ(t) = 35°·sin(2πt/10) |
| F4_lech_pha | Tĩnh tại (5, 0, 1,5); a_x = 1,1+0,45 sin(2πt/8); a_y lệch pha 120°; a_z = 0,9 |
| F5_ket_hop | Tâm như D1 (v = 0,9 m/s); a_y = 1,0+0,35 sin(2πt/6) ∈ [0,65; 1,35]; a_x = a_z = 0,9 |
| F1-MS | Ô minh họa: thanh F1 = hợp 9 cầu con r = 0,28 m tại y ∈ {−2; …; 2} (chỉ A3, 10 seed phụ, không tham gia phán quyết §5.3) |

> **Ghi nhận định lượng về F1-MS — hợp cầu KHÔNG phủ kín thanh
> (lệch nằm trong chính văn bản giao thức).** Lấy mẫu dày bề mặt
> ellipsoid F1 (rà soát độc lập của Vân, lưới 1201×1081 trong tham
> số hoá cầu đơn vị, đo khoảng cách tới tâm cầu con gần nhất): hợp
> 9 cầu con r = 0,28 m **không chứa trọn ellipsoid** — độ chòi tối
> đa **+0,0109 m**, khoảng **2% điểm bề mặt** nằm ngoài hợp cầu;
> điểm chòi lớn nhất nằm **giữa hai tâm cầu kề nhau** (y ≈ ±0,25;
> ±0,75; …) ở **phía cực trị x/z** của thanh. Các cầu con vẫn chồng
> lấp liên tục với nhau (khoảng cách tâm 0,5 < 2r = 0,56) và phủ
> kín mặt cắt ngang (0,15√2 ≈ 0,212 < 0,28), nhưng phát biểu "bao
> phủ thanh" là không đúng tuyệt đối ở mức ~1,1 cm. Lệch này nằm
> trong chính **§2.5 của giao thức** (r = 0,28 là con số đã đóng
> băng): gói này cài **đúng giao thức**, tự ý đổi bán kính mới là
> lệch giao thức. Hệ quả diễn giải: trong ô F1-MS, khe hở "có
> chứng nhận" của A3 có thể bị đánh giá cao hơn thực tế tối đa
> ~0,011 m tại các điểm chòi (nhỏ so với d = 0,5 m), và F1-MS là ô
> minh họa, không tham gia phán quyết §5.3. Thêm nữa, kiểm hợp lệ
> của F1-MS trong `validate.py` hiện đo trên **khoảng cách
> ellipsoid** (entry mang `ellipsoid_truth_fns` và
> `scenario_signed_distance` ưu tiên ellipsoid — đường tìm được và
> `path_len` của F1-MS trùng hệt F1), nên **hình học hợp-cầu của
> F1-MS chưa từng được kiểm riêng**. Phương án sửa tận gốc (đổi
> r = 0,30 qua thủ tục §9.3) được ghi ở mục đề xuất cuối tài liệu
> này; chừng nào chưa được duyệt, code giữ nguyên r = 0,28.

## Mức trung thực của loạt A / B so với benchmark gốc (xấp xỉ ở đâu)

Bài 1 v2 đã tuyên bố rõ và gói này giữ nguyên tuyên bố đó: hai
khung gốc **không chạy trực tiếp được** trong môi trường của đề
tài (FlightBench xây trên Flightmare/Unity + ROS/catkin;
FLYINGTRUST dùng chuỗi mô phỏng nặng hơn), và cả hai kiểm tra dẫn
đường *có cảm biến trong vòng lặp* — khác thiết lập nhận trạng thái
qua quan sát + dự báo của đề tài. Vì vậy:

- **Loạt A chỉ mượn thang độ khó phân cấp của FlightBench** (8 cấp
  mật độ vật cản) và bộ chỉ số của nó (tới đích, khe hở, thời gian
  bay, độ dài quỹ đạo, jerk RMS, thời gian giải — tính lại theo §1
  của giao thức). Môi trường, bản đồ và ngăn xếp lập kế hoạch của
  FlightBench **không** được tái lập; trường vật cản là trường cầu
  ngẫu nhiên do code của Bài 1 tự sinh với quy tắc seed công bố ở
  trên. Mọi văn bản phải gọi đây là "loạt A — thang độ khó *kiểu*
  FlightBench (giao thức tái cài đặt)", không gọi là kết quả
  FlightBench.
- **Loạt B chỉ mượn chiều biến thiên nền tảng của FLYINGTRUST**
  (3 cấp giới hạn vận tốc/gia tốc) chạy trên kịch bản của chính đề
  tài {S2, S4, D1}. Chuỗi mô phỏng và kịch bản gốc của FLYINGTRUST
  **không** được tái lập.
- Thông tin trong Bài 1 v2 + code chỉ đủ để tái lập trung thực ở
  đúng mức trên (số cấp, số vật, dải tham số sinh trường, giới hạn
  nền tảng đều có đủ); không có thông số nào bị bịa thêm. L8 của
  loạt A chưa từng hoàn tất ở Bài 1 (chi phí giải ~1,1–2,4 giờ/lần
  lặp theo chẩn đoán của Bài 1 v2) — giao thức §2.3 yêu cầu GĐ3
  hoàn tất L8 cho ít nhất A0 và A3 hoặc ghi deviation §7.4.

## Kiểm tra tính hợp lệ (độc lập cấu hình — §2.5(i)(ii))

`validate.py` kiểm tra, **không dùng bất kỳ cấu hình A0–A6/baseline
nào làm oracle**:

1. **A\* lưới mịn 0,1 m** trên trường sự thật nền phồng đều theo
   biên d = 0,5 m quanh bề mặt thật (khoảng cách ellipsoid = khoảng
   cách Euclid tới bề mặt qua điểm gần nhất). Kịch bản tĩnh: một
   trường; kịch bản động/biến dạng: các lớp thời gian bất lợi nhất
   (đỉnh phồng/giao nhau) **cộng** kiểm tra đường đi theo thời gian
   (trạng thái vật tại thời điểm dự kiến tới, v_nom = 1/2/3 m/s,
   phán quyết ở 2 m/s). Hai mức được báo cáo tách bạch: A\* với
   phồng đúng d (đúng chữ giao thức) và *chứng nhận đoạn thẳng*
   (A\* với phồng d + (√3/2)·0,1 — điều kiện đủ để mọi đoạn của
   đường giữ khe hở ≥ d, cùng lập luận với `astar_seed.py`).
2. **Kiểm tra số học cực trị**: bán trục/bán kính/yaw cực đại–cực
   tiểu của F1–F5 và D3 khớp bảng §2.5 (gồm abc = 1 của F2, biên độ
   yaw 35° của F3, lệch pha đỉnh 8/3 s của F4, r_eq của F1/F3).

Kết quả đầy đủ: `validation_results.json` (bảng tóm tắt ở cuối tài
liệu này). Kịch bản nào không hợp lệ sẽ bị sửa tham số và ghi lại
tại đây — quyết định thuộc về kiểm tra hình học, không thuộc về bộ
giải nào.

## Tích hợp runner

`sim_realtime.run_episode` hiện nạp kịch bản qua
`dynamic_scenarios.dynamic_scenario` (chỉ họ D).
`registry.resolve_for_runner(name)` là hàm thay thế trực tiếp (tra
họ D qua đường cũ, mọi họ khác qua sổ này) — tích hợp chính thức =
một điểm nối đó trong `run_episode`, thực hiện khi hạ tầng sẵn sàng
sửa `sim_realtime.py`. `runner_smoke.py` chứng minh tính ăn khớp
ngay bây giờ bằng cách gắn hàm thay thế này tại thời gian chạy
(không sửa file hạ tầng): A0, đóng băng thời gian, seed thử
9 000 000/9 000 001 trên `A_L1_r0` và `F1_thanh_dai`.

Lưu ý đánh giá họ F qua đường chạy hiện tại: `sim_realtime` tính
khe hở/va chạm theo `truth_fns` — với họ F đó là cầu
thể-tích-tương-đương, **đúng giao diện mà A0/A1 được định nghĩa**
trên họ F (§2.7: kết quả A0/A1 trên họ F đo thất bại của biểu diễn
cầu-fit). Khe hở so với **bề mặt ellipsoid thật** cho A2/A3 thuộc
đường đánh giá có tầng hình dạng; hàm khoảng cách dùng chung là
`validate.ellipsoid_signed_distance`, sự thật đầy đủ ở
`ellipsoid_truth_fns`.

Loạt B cần truyền (v_max, u_max) của nền tảng vào `PlannerParams` —
runner hiện chưa có cờ; khi tích hợp chính thức cần bổ sung ở tầng
runner, không thuộc phạm vi gói kịch bản này.

## Danh sách vật cản S4 đã đóng băng (§2.1 — `default_rng(7)`)

Tâm (x; y; z), bán kính r (m), z = 1,5 cho mọi vật:

| # | Tâm | r |
|---|---|---|
| 1 | (8,125859; 4,574924; 1,5) | 1,010274 |
| 2 | (4,526865; −0,798503; 1,5) | 1,049421 |
| 3 | (2,547388; 3,891056; 1,5) | 1,018828 |
| 4 | (10,754510; 2,163036; 1,5) | 0,905647 |
| 5 | (5,825827; −3,466392; 1,5) | 1,032019 |
| 6 | (9,175939; −2,676540; 1,5) | 0,916458 |
| 7 | (7,725876; 0,339846; 1,5) | 1,051275 |
| 8 | (5,259935; 5,152158; 1,5) | 0,886336 |

## Tóm tắt kết quả kiểm hợp lệ

Đã chạy `validate.py` cho toàn bộ 40 kịch bản có hình học riêng
(loạt B dùng lại hình học của S2/S4/D1 nên không kiểm tra trùng).
`validation_results.json` hiện tại là đầu ra nguyên trạng của
**một lần chạy duy nhất** của `validate.py` hiện hành (ngày
2026-11-22, sinh lại theo mục bắt buộc của rà soát độc lập — bản
giao trước đó là bản ghép hai lần chạy của hai phiên bản kiểm
tra); định dạng `traversal` nay đồng nhất theo tiêu chí tồn tại
cho cả 8 kịch bản động/biến dạng, và **không kết luận hợp lệ nào
thay đổi** so với bản ghép trước:

- **40/40 kịch bản HỢP LỆ** ở cả hai mức: A\* với phồng đúng d
  (`astar_inflated_d`) và chứng nhận đoạn thẳng
  (`astar_certificate`, phồng d + (√3/2)·0,1) ở **mọi** lớp thời
  gian bất lợi nhất; khe hở thật nhỏ nhất dọc đoạn của các đường
  tìm được đều ≥ 0,50 m (giá trị điển hình 0,50–0,52 m — đường A\*
  bám sát biên d như kỳ vọng).
- **Kiểm tra số học cực trị (§2.5(ii)): ĐẠT toàn bộ** — F1 r_eq =
  0,3557 (bảng: ≈0,356); F2 a_x ∈ [1; 1,8], abc = 1 với sai lệch
  ≤ 2,3×10⁻¹⁶; F3 biên độ yaw đúng 35,0°, r_eq = 0,7368 (bảng:
  ≈0,737); F4 khoảng đỉnh hai trục lệch 2,665 s (lý thuyết 8/3 s);
  F5 tốc độ tâm 0,9 m/s, a_y ∈ [0,65; 1,35]; D3 r ∈ [0,65; 1,35].
- **Kiểm tra đường đi theo thời gian** (kịch bản động/biến dạng):
  tiêu chí *tồn tại* — thử mọi đường ứng viên (mọi lớp bất lợi × 2
  mức phồng) × lệch giờ khởi hành δ ∈ {0; 1; 2; 3} s (chờ trước khi
  bay là tham số hoá thời gian hợp lệ) × tốc độ {1; 2; 3} m/s, đạt
  khi có tổ hợp sạch ở v = 2 m/s. Bốn kịch bản D1, F2, F3, F5 ở lần
  kiểm đầu tiên (một đường tĩnh duy nhất, δ = 0) cho kết quả âm
  tính giả: đường A\* duy nhất đó vòng đúng phía vật đang tiến tới
  (D1/F5) hoặc lách ngang vừa khít ở pha phồng mà bán trục ngang
  lại lớn nhất ở pha khác (F2), hoặc bị đầu trục dài đã xoay quét
  qua (F3) — trong khi đường khả thi theo thời gian tồn tại. Sau
  khi đổi sang tiêu chí tồn tại: cả bốn đều đạt, với lề trên d ở
  v = 2 m/s là D1 +0,679 m; F5 +0,621 m; F2 +0,089 m; F3 +0,085 m
  (F2/F3 lề mỏng — đúng tính chất thiết kế: đây là các kịch bản
  phân biệt khó). Không kịch bản nào phải sửa tham số.

Chi tiết từng kịch bản: `validation_results.json` (từng lớp thời
gian có `astar_inflated_d`, `seg_min_clearance_m`,
`astar_certificate`; kịch bản động có `traversal` với toàn bộ tổ
hợp ứng viên và `best_margin_over_d_at_v2_m`).

## Đề xuất sửa giao thức — bán kính cầu con F1-MS (chờ đợt 2.0, thủ tục §9.3)

Theo ghi nhận định lượng ở mục Họ F ở trên, hợp 9 cầu con
r = 0,28 m của ô F1-MS chòi tối đa +0,0109 m so với bề mặt
ellipsoid F1. **Đề xuất:** trong đợt sửa đổi giao thức bản 2.0
(theo thủ tục §9.3), sửa §2.5 đổi bán kính cầu con của F1-MS từ
**0,28 m thành 0,30 m** — theo chính số đo chòi tối đa 0,0109 m,
r = 0,30 phủ kín thanh với lề ~0,009 m tại điểm chòi lớn nhất đã
lấy mẫu — để phát biểu "bao phủ thanh" đúng tuyệt đối và khe hở
có chứng nhận của ô minh họa không còn bị đánh giá cao hơn thực
tế. Đề xuất này đi kèm đợt sửa đổi 2.0 để đội trình người dùng ký
một lần; **chừng nào chưa được duyệt, code và kiểm hợp lệ giữ
nguyên r = 0,28 theo giao thức 1.2 đã đóng băng**, và ghi nhận ở
trên là căn cứ diễn giải duy nhất cho ô F1-MS.
