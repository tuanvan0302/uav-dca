# Baselines Tier-1

Gói `baselines/` cài đặt ba phương pháp so sánh cho ST-DCA theo các
đặc tả **v1.1**:

| File | Phương pháp | Đặc tả | Bài gốc |
| --- | --- | --- | --- |
| `ngo2023.py` | Penalty-DCA của Ngo et al. | SPEC-BL-NGO-2023 v1.1 | IEEE CDC 2023, DOI 10.1109/CDC49753.2023.10383259 (arXiv:2303.07581v3) |
| `cbf2021.py` | Convexified-CBF của Varun, Vinod & Kolathaya | SPEC-BL-CBF-2021 v1.1 | Proc. ICC 2021, tr. 81–86 |
| `mpcqp.py` | MPC-QP tuyến tính hóa một lần (mốc tham chiếu, không phải bài báo) | SPEC-BL-MPCQP v1.1 | — |
| `common.py` | Luật chơi chung: mảng dự báo đóng băng + checksum, `need`, J chung, động học, dung sai solver | SPEC-FAIR-EXP4 (SPEC_cong_bang) v1.1 | — |
| `run_baseline.py` | Chạy một episode cho một baseline và ghi log JSON | Giao thức §7.2 | — |
| `test_baselines.py` | Các mục kiểm chứng bắt buộc ở cuối từng SPEC + checklist công bằng §14 | — | — |

Nguyên tắc phân cấp (SPEC_cong_bang §1.5): **giao thức thắng SPEC khi
xung đột**. Mọi phương pháp dùng cùng mô hình tích phân kép chính xác,
cùng hộp giới hạn, cùng J, cùng mảng dự báo **đóng băng** và cùng dung
sai solver chốt bằng số.

---

## 1. Giao diện

Cả ba module có cùng giao diện với `core.dca_planner.plan`, thêm tham số
cấu hình riêng `cfg`:

```python
plan(p0, v0, goal, obstacles, prm, seed_P=None, cfg=None) -> BaselineResult
```

- `obstacles`: đúng định dạng của `core.dca_planner.normalize_obstacles`
  (dict `{'centers', 'radii', 'rho'}` hoặc tuple `(tâm, bán kính)`).
  Mỗi baseline suy ra `(O, R, RHO)` bằng **chính hàm đó** qua
  `common.forecast_arrays`, rồi tính
  `need = R + d_safe + extra_margin + RHO`. Không phương pháp nào đọc
  trạng thái thật của vật cản; đầu vào duy nhất là mảng dự báo harness
  cấp (SPEC_cong_bang §4.3/M-8).
- `prm`: `PlannerParams` của Bài 1. Baseline chỉ đọc các trường dùng
  chung: K, dt, alpha, beta, gamma, v_max, u_max, z_floor, d_safe,
  extra_margin.
- `cfg`: `NgoConfig` / `CbfConfig` / `MpcqpConfig`. Giá trị mặc định là
  cấu hình chính đã đóng băng.
- Kết quả `BaselineResult` là lớp con của `PlanResult`, thêm túi `meta`.
  Các trường chuẩn (`P, V, U, success, min_clearance, total_slack,
  iterations, solve_time, obj_history`) giữ đúng ngữ nghĩa của
  `PlanResult`. `meta` mang các chỉ số phụ của SPEC_cong_bang §12:
  `time_to_first_feasible` (+ số vòng), `hit_cap`, `stop_reason`, cờ
  success nội bộ **kèm đơn vị** (Ngo: m; CBF: m²; MPC-QP: không ngưỡng),
  solver + dung sai thực dùng, `forecast_checksum` (sha256 của bộ ba
  `(O, R, RHO)` của lần gọi đó) và các cờ riêng (MPC-QP:
  `qp_infeasible`, `degenerate_cut`; CBF: `n_init_qp = 1`,
  `sum_slacks_m2`, `max_slack_m2`).
- **Không có trạng thái ẩn qua chu kỳ** (SPEC_cong_bang §5.3): không
  module nào giữ trạng thái giữa các lần gọi `plan()`. Bài toán CVXPY
  được dựng mới mỗi lần gọi và thời gian dựng tính vào `solve_time`.

### Chạy một episode

Từ thư mục `code/` (cần gói `sim/`, nằm ngoài `code/`):

```bash
python3 -m baselines.run_baseline --method CBF2021 \
    --scenario D1_cat_ngang --seed 9000000 --mode kalman \
    --time-model frozen --outdir <thư_mục_log>
```

- `--method`: `Ngo2023` | `CBF2021` | `MPCQP`.
- `--scenario`: họ động `D1_cat_ngang`…`D4_ket_hop`, hoặc kịch bản tĩnh
  `S0_trong`, `S1_mot_vat`, `S3_loi_hep`. Driver đăng ký các kịch bản
  tĩnh vào registry động **trong tiến trình**, chỉ những kịch bản có
  start/goal trùng họ động; S2/S4 có đích khác nên bị bỏ qua.
- Driver thay `sim_realtime.plan` bằng `plan()` của baseline, bọc thêm
  một lớp ghi metadata từng lần gọi.
- Log JSON gồm: `run_id`, `timestamp_utc`, `protocol`,
  `protocol_version`, `code_files` và `planner_file` (đường dẫn các file
  đã nạp), `machine`, `status`, `error`, `method`, `scenario`, `mode`,
  `time_model`, `seed`, `config`, `metrics`, `cycles`, và `plan_calls`
  (metadata của từng lần gọi `plan()`). Chu kỳ có lỗi được ghi thêm vào
  `<run_id>_diag.json`.

---

## 2. Tham số và nguồn của từng giá trị

### 2.1 Dùng chung (SPEC_cong_bang §2, §3, §7)

| Tham số | Giá trị | Nguồn |
| --- | --- | --- |
| K, Δt | 28; 0,25 s | `PlannerParams` |
| α, β, γ của J | 1; 0,5; 30 | `PlannerParams` (γ này là trọng số J, khác γ CBF) |
| v_max, u_max | 3 m/s; 4 m/s², hộp theo thành phần (lệch D-1) | `PlannerParams` |
| d_safe, extra_margin | 0,5 m; 0,03 m (trong `need`) | `PlannerParams` |
| Sàn z ≥ 0,8 m | **cứng** ở cả 3 baseline (Bài 1 phạt bằng biến dư — lệch D-6) | SPEC_cong_bang §10 |
| OSQP | eps_abs = eps_rel = 1e-4, max_iter = 20000, warm_start = True | `core.dca_planner.OSQP_OPTS` (SPEC_cong_bang §7; KHÔNG phải mặc định 1e-3 của OSQP) |
| ECOS / Clarabel | abstol = reltol = feastol = 1e-8 / tol_gap_abs = tol_gap_rel = tol_feas = 1e-8 | SPEC_cong_bang §7; thay thế bằng Clarabel ghi ở DV-B2 |

### 2.2 Ngo-2023 (SPEC-BL-NGO-2023 §5.1)

| Tham số | Giá trị | Nguồn |
| --- | --- | --- |
| τ₀ | **1,0** (Bài 1 dùng 10 — chênh lệch phải công bố, D-5) | Quyết định Q-N1 |
| µ, τ̄ | 4,0; 2×10⁴ (bằng Bài 1) | Quyết định Q-N2 |
| ε (δ_m) | 1e-6 **tương đối** | Quyết định Q-N3 |
| δ_loose (nhánh dừng (0)) | 1e-2 tương đối | Quyết định Q-N5 (nhánh tự thêm vì công bằng — A-3) |
| Ngưỡng Σ s khả thi | 1e-3 m (cùng `slack_tol` của Bài 1) | SPEC §5.1/§5.2 |
| Trần vòng lặp | 200, chạm trần gắn cờ `hit_cap` | Quyết định Q-N4 |
| Cờ success nội bộ | `total_slack ≤ max(slack_tol, 1e-2)` = 1e-2 m — chỉ metadata | cùng quy tắc với `dca_planner.plan` |
| Mặt cắt | tích lũy **không trần**, không điểm giữa | Bài gốc Mục V (M-10/M-11) |
| S-N1 (nhạy cảm) | mục tiêu gốc (5), ρf = 1, a = 0,1 (tự chọn — bài gốc không cho số) | SPEC §5.3 |
| S-N2 (nhạy cảm) | thêm ràng buộc cứng p_K = goal | SPEC §5.3 |

Điều kiện dừng (§5.2, dừng ở điều kiện đầu tiên thỏa):

- **(0)** Σ s ≤ 1e-3 ∧ δ_m ≤ 1e-2 (tương đối) — tự thêm, đối xứng Bài 1.
- **(i)** δ_m ≤ 1e-6 — tiêu chí gốc.
- **(ii)** τ = τ̄ ∧ Σ s ≤ 1e-3 — gốc, thêm vế Σ s.
- **(iii)** chạm trần 200 vòng.

τ tăng **mỗi** vòng theo Bước 3 gốc. `time_to_first_feasible` = vòng /
thời gian đầu tiên Σ s ≤ 1e-3 (định nghĩa SPEC §5.2); metadata ghi
thêm bản theo vi phạm khoảng cách thật ≤ 1e-3 m (`*_true_m`).

**Khi so `time_to_first_feasible` giữa các phương pháp:** trường chính
của Ngo dùng định nghĩa theo biến dư, còn CBF/MPC-QP dùng định nghĩa
phía bộ đánh giá (vi phạm khoảng cách thật ≤ 1e-3 m, SPEC_cong_bang
§6.2). Hai định nghĩa không đồng nhất, nên bảng so sánh phải dùng bản
`*_true_m` của Ngo.

### 2.3 CBF-2021 (SPEC-BL-CBF-2021 §4)

| Tham số | Giá trị | Nguồn |
| --- | --- | --- |
| µ | 4 | Nguyên văn Mục IV bài gốc |
| τ_max | 10⁵ | Nguyên văn Mục IV |
| δ_cost | 0,1 — tuyệt đối, trên J **không** phạt (quyết định A-8) | Nguyên văn Mục IV + A-5/A-8 |
| δ_viol | 0,001 — trên Σ s, đơn vị **m²** | Nguyên văn Mục IV + A-6 |
| γ (CBF) | **0,9** — TỰ CHỌN, bài gốc không cho số (D-4) | SPEC §4.2 |
| τ₀ | **1,0** — TỰ CHỌN, Thuật toán 1 không liệt kê (A-2) | SPEC §4.2 (Q-C1) |
| Trần vòng lặp | 50 (tự thêm — A-7; bài gốc ~8 vòng thực nghiệm) | SPEC §4.2 |
| Cờ success nội bộ | `total_slack ≤ δ_viol` = 1e-3 **m²** — chỉ metadata | SPEC §3 |
| Khởi tạo z | QP (16') bỏ ràng buộc an toàn, giải OSQP; `n_init_qp = 1` trong meta, không tính vào `iterations` | Bài gốc pt. (16), M-9 |
| Tâm trong (11d) | `ô_{m,k}` cho **cả hai vế** (nguyên văn, M-7/A-4) | Bài gốc pt. (9)/(11d)/(14)/(17) |
| S-C1 / S-C2 / S-C3 (nhạy cảm) | tâm k+1 cho vế x_{k+1} / mục tiêu (11a) với Q = I (vị trí), R = 0,1·I / `cost := J + τ·Σs` | SPEC §4.3 |

`time_to_first_feasible` của CBF tính theo định nghĩa phía bộ đánh giá
(vi phạm khoảng cách thật ≤ 1e-3 m, SPEC_cong_bang §6.2/§9.1), vì Σ s
của CBF ở đơn vị m² nên không so trực tiếp được. Σ s và max s dạng thô
(m²) vẫn được ghi trong metadata theo A-6.

### 2.4 MPC-QP (SPEC-BL-MPCQP §2)

Không có siêu tham số riêng. Mỗi (m, k) có một mặt cắt tại điểm tham
chiếu (lạnh: nội suy thẳng; ấm: `seed_P`). Giải đúng 1 QP (OSQP, dung
sai như §2.1); mặt cắt và sàn đều **cứng**, không biến dư. QP vô nghiệm
⇒ trả quỹ đạo tham chiếu `q` + cờ `qp_infeasible` (SPEC Bước 4 +
SPEC_cong_bang §9.2); (V, U) của `q` khi đó suy bằng hiệu hữu hạn trên
lưới Δt với V[0] ép bằng v0 (xem §4).

---

## 3. Ngân sách tinh chỉnh ≤ 15 cấu hình/phương pháp (DV-B1)

SPEC_cong_bang §8.2 chốt trần 15 cấu hình đánh giá trên gia đình hiệu
chuẩn (giao thức §4.3) cho mỗi phương pháp, nhưng không nói cấu hình
nhạy cảm có tính vào ngân sách hay không. **Quyết định cài đặt (bảo
thủ, DV-B1): tính cả vào.**

| Phương pháp | Lưới tinh chỉnh trên tập hiệu chuẩn | Cấu hình nhạy cảm | Tổng |
| --- | --- | --- | --- |
| Ngo-2023 | τ₀ ∈ {0,1; 1; 10} = 3 (µ/τ̄ cố định theo Q-N2, không tính) | S-N1, S-N2 = 2 | **5** |
| CBF-2021 | γ ∈ {0,5; 0,8; 0,9; 0,95; 0,99} tại τ₀ = 1 (5) + τ₀ ∈ {0,1; 10} tại γ = 0,9 (2) = 7 | S-C1, S-C2, S-C3 = 3 | **10** |
| MPC-QP | — | — | **0** |
| Bài 1 / ST-DCA | 0 (tham số đã công bố) | — | **0** |

Hệ quả: lưới CBF **không** chạy full factorial γ × τ₀ = 5 × 3 = 15, vì
cộng 3 cấu hình nhạy cảm sẽ thành 18 > 15. Lưới rút gọn ở trên vẫn chứa
đủ 5 giá trị γ và 3 giá trị τ₀ quanh điểm chính (γ = 0,9, τ₀ = 1). Gói
này chỉ cung cấp cơ chế (`cfg` ghi đè mọi tham số); không cấu hình nào
được chọn bằng kịch bản đánh giá (SPEC_cong_bang §8.1).

---

## 4. Quyết định cài đặt đáng chú ý

1. **Sàn cứng cho cả 3 baseline** (SPEC_cong_bang §10, lệch D-6):
   `P[:, 2] >= z_floor` là ràng buộc cứng, khác Bài 1 (phạt bằng biến
   dư). Nếu p0/v0 làm QP vô nghiệm vì sàn + hộp, mỗi baseline xử lý
   theo quy tắc thất bại trong SPEC của nó: MPC-QP trả `q`; Ngo/CBF trả
   quỹ đạo tốt nhất đã có, hoặc quỹ đạo đứng yên + `total_slack = inf`.
2. **CBF không dùng `seed_P`** dù harness có cấp: QP khởi tạo (16') là
   cơ chế khởi tạo định nghĩa của phương pháp (SPEC_cong_bang §5.1).
   Ngo và MPC-QP dùng `seed_P` theo quy tắc ấm chung.
3. **`cost` của CBF không gồm hạng tử phạt** (A-8): nếu gồm phạt, mỗi
   lần τ ×4 làm Δcost nhảy vọt và δ_cost = 0,1 gần như không bao giờ
   đạt, nên thuật toán luôn chạy tới trần. Cách đọc ngược lại có sẵn
   qua `cfg.cost_includes_penalty = True` (S-C3).
4. **Đơn vị `total_slack` khác nhau theo phương pháp** (SPEC_cong_bang
   §9.1): Ngo = m (Σ s của bài toán con, kẹp tại 0 vì nhiễu dung sai
   OSQP có thể cho −3e-6); CBF = m²; MPC-QP = 0 (không có biến dư;
   `success` = QP giải được). Cờ `success` của baseline là cờ nội bộ,
   **chỉ là metadata**; thành công mức episode do bộ đánh giá chung
   định nghĩa (giao thức §1.4).
5. **Solver nón: ECOS nếu có, không thì Clarabel** (DV-B2): code tự chọn
   ECOS khi khả dụng và ghi tên + dung sai thực dùng vào metadata từng
   lần gọi. Không trộn số của hai solver trong một bảng mà không ghi
   chú.
6. **MPC-QP khi QP vô nghiệm**: trả `q` với (V, U) suy bằng
   `np.gradient`/hiệu hữu hạn (V[0] ép = v0). Đây là đường dự phòng duy
   nhất; SPEC cấm thử lại/nới lỏng trong cùng một lần gọi.
7. **Ngo khi solver thất bại ở vòng đầu**: trả quỹ đạo đứng yên tại p0
   với `total_slack = inf` (cùng quy ước của `dca_planner.plan` khi
   không có nghiệm nào), để tầng mô phỏng áp chính sách chu kỳ thất bại
   thống nhất (SPEC_cong_bang §9.2).
8. **Ellipsoid bị từ chối tường minh**: `common.forecast_arrays` ném lỗi
   nếu đầu vào có khoá `'axes'`; baseline chỉ chạy vật cầu
   (SPEC_cong_bang §4.2).
9. **Cờ success nội bộ không chứng nhận an toàn.** Với CBF, ràng buộc
   (14)/(17) chỉ bảo đảm đệ quy h_{k+1} ≥ γ·h_k − s_k trên mảng dự báo,
   **không** bảo đảm ‖p_k − ô‖ ≥ need tại từng nút. Quỹ đạo có thể đi vào
   vùng h < 0 rồi phục hồi dần, nên khe hở thật có thể nhỏ hơn d dù cờ
   nội bộ báo thành công. Vì vậy SPEC_cong_bang §9.1 cấm dùng cờ nội bộ
   làm chỉ số so sánh.

---

## 5. Kiểm thử

Từ thư mục `code/`:

```bash
python3 -m baselines.test_baselines
```

`test_baselines.py` chạy 13 mục kiểm chứng:

| Mục | Nội dung kiểm |
| --- | --- |
| Ngo §7.1 | g(x) ≥ tuyến tính hoá của nó tại mọi mẫu (underestimator) |
| Ngo §7.2 | vòng đầu trùng Bài 1: tầng cơ chế (Clarabel 1e-10) tới 1e-4 m; tầng chạy thật (OSQP 1e-4) chỉ ghi số (DV-B3) |
| Ngo §7.3/§7.4 | hội tụ trên vật động cắt ngang; khả thi tại mọi nút khi success |
| Ngo S-N1 | mục tiêu gốc (5) qua SOCP chạy được |
| CBF §6.1 | (14) ⇒ (11d) trên 500 mẫu ngẫu nhiên |
| CBF §6.2 | bất biến tiến và phục hồi; đệ quy h_{k+1} ≥ γ·h_k − s_k tới 1e-5 |
| CBF §6.3 | dừng bởi tiêu chí (δ_viol, δ_cost) trước trần 50 vòng |
| CBF §6.4 | dấu hạng tử hằng số của (17): lhs(z, z, z) = (γ−1)·h(z) tới 1e-12 |
| MPC-QP §3.1 | không vật cản ⇒ trùng `dca_planner.plan` tới 1e-4 m |
| MPC-QP §3.2 | tham chiếu an toàn ⇒ nghiệm không vi phạm khe hở nút quá 1e-6 m |
| MPC-QP §3.3 | đúng 1 lần gọi solver mỗi `plan()` |
| MPC-QP §3.4 | tham chiếu xuyên tâm vật cản: cắt suy biến bị bỏ + ghi cờ; hai lần chạy trùng từng bit |
| Công bằng §14 | ba phương pháp nhận checksum `(O, R, RHO)` trùng nhau ở từng chu kỳ của cùng một episode |

Mục §14 chạy một episode qua `sim.sim_realtime`, nên cần gói `sim/`.
