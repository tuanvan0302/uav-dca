# -*- coding: utf-8 -*-
"""
Vòng lặp mô phỏng vòng kín HAI chế độ thời gian cho thực nghiệm ST-DCA.

Chế độ "frozen" (đối chứng cũ, tương đương simulate_dynamic.py của Bài 1):
  mỗi chu kỳ, thời gian mô phỏng chỉ tiến đúng dt bất kể bộ giải tính
  bao lâu. Khe hở đo được trong chế độ này LẠC QUAN vì thế giới đứng
  yên trong lúc bộ giải tính. Giao thức 1.2 §7.1 chỉ cho phép chế độ
  này trong smoke test (seed dải 9xxxxxx, §3.3).

Chế độ "realtime" (mô hình trễ thực thi có bù trễ, giao thức §7.1):
  Một chu kỳ bắt đầu tại thời gian mô phỏng t_c với UAV đang thực thi
  kế hoạch hiện hành. Độ trễ của một chu kỳ được định nghĩa thống nhất
  (giao thức §0/§7.1):

      T_latency = t_predict + t_seed + t_plan

  trong đó t_predict là thời gian của tầng dự báo (get_obs: Kalman /
  truth / frozen) đo bằng perf_counter quanh lời gọi, t_plan là
  thời gian của lời gọi plan() (trường solve_time của planner, §1.7),
  và t_seed là thời gian tính hạt giống A* theo need của
  method_adapter (DV-H8 §4.2 mục 6/9b — bước nối M2). Chỉ các cấu
  hình qua adapter (A1–A3) mới có t_seed; A0 và các baseline không
  qua adapter nên t_seed = 0 và công thức thu về dạng hai thành
  phần cũ, log của chúng không thêm khoá hạt giống nào.
  Cấm dùng t_plan trần thay T_latency ở bất kỳ đâu ngoài bảng hiệu
  năng. Độ trễ ƯỚC LƯỢNG của chu kỳ này (τ̂, dùng để neo) là T_latency
  ĐO THẬT của chu kỳ TRƯỚC (quy ước T_r của H1, ST-DCA Ch.6; chu kỳ
  đầu τ̂ = 0):

    1. Trạng thái neo: dự báo trạng thái UAV tại t_c + τ̂ bằng cách
       truyền trạng thái hiện tại dọc kế hoạch hiện hành (toán học
       thuần, chưa bay thật) — gồm cả đoạn hãm bão hòa nếu kế hoạch
       hiện hành cạn trước t_c + τ̂ (xem quy tắc hãm dưới đây).
    2. Bộ giải lập kế hoạch TỪ trạng thái neo, với quan sát vật cản
       neo tại t_c + τ̂. T_latency thật của chu kỳ được đo khi chạy.
    3. Trong [t_c, t_c + T_latency]: thế giới trôi theo sự thật nền;
       UAV thực thi tiếp kế hoạch hiện hành; khi kế hoạch hiện hành
       cạn, áp QUY TẮC HÃM BÃO HÒA của §7.1 (không còn trôi tự do
       u = 0 như bản trước — xem dưới). Pha trễ bị chặn bởi t_max:
       đồng hồ mô phỏng không bao giờ vượt t_max.
    4. Tại t_c + T_latency: áp lệnh của kế hoạch mới trong một
       bước dt (cũng bị chặn bởi t_max); kế hoạch mới thành kế hoạch
       hiện hành. Lệnh áp có phản hồi vận tốc theo trạng thái thật
       (mục THỰC THI BÁM TRẠNG THÁI dưới đây): lệch T_latency ≠ τ̂
       để lại một sai lệch vận tốc giữa trạng thái thật và hồ sơ
       của kế hoạch mới — sai lệch đó nay là sai số phản hồi, suy
       giảm mũ theo luật phản hồi thay vì được mang nguyên vẹn
       sang mọi kế hoạch sau như ở bản thực thi hở vòng trước.

  QUY TẮC HÃM BÃO HÒA (§7.1, bản 1.2 — clip THEO CHUẨN):
  khi kế hoạch cũ đã hết, áp mỗi bước Δt lệnh

      u = −(v/Δt) · min(1, u_max/‖v/Δt‖)

  tức hãm thẳng theo hướng −v với độ lớn min(‖v‖/Δt, u_max), bảo đảm
  ‖u‖ ≤ u_max ở mọi bước, lặp lại cho tới khi ‖v‖ ≤ 0,05 m/s, sau đó
  giữ vị trí (u = 0). (Dạng clip theo từng thành phần của bản 1.1 cho
  phép ‖u‖ tới √3·u_max ≈ 6,93 > u_max = 4 — đã bỏ. Quy tắc cũ của hạ
  tầng này — u = 0 ngay khi hết kế hoạch — với mô hình tích phân kép
  nghĩa là giữ nguyên vận tốc trôi thẳng vô hạn; rà soát của Thiên
  đo được đoạn trôi dài nhất 35,85 s trong lô smoke cũ. Đã sửa ở cả
  hai nơi dùng: thực thi thật (fly_along_plan) và truyền trạng thái
  neo (propagate_along_plan).)

  Chính sách khi bộ giải thất bại (total_slack = inf): không áp lệnh
  của kế hoạch hỏng; thực thi tiếp kế hoạch hiện hành thêm dt, giữ
  hạt giống ấm, thử lại chu kỳ sau; đếm vào n_planner_failures.

  THỰC THI BÁM TRẠNG THÁI — PHẢN HỒI TRẠNG THÁI (hiệu chỉnh mô
  hình thực thi ở tầng sim, 2026-11-22; planner KHÔNG đổi):
  Bản trước thực thi kế hoạch hiện hành kiểu HỞ VÒNG: lệnh áp là
  đúng dãy U của kế hoạch, đánh chỉ số theo cur_offset, bất kể
  trạng thái thật đang ở đâu so với hồ sơ (P, V) của kế hoạch.
  Vì lệnh đầu của kế hoạch mới được thiết kế cho trạng thái neo
  tại t_c + τ̂ nhưng chỉ áp tại t_c + T_latency, mỗi lần đổi kế
  hoạch dưới jitter trễ bơm vào một lệch vận tốc δ (độ biến thiên
  vận tốc của kế hoạch cũ trên khoảng lệch τ̂ ↔ T_latency); động
  học tuyến tính giữ δ NGUYÊN VẸN qua mọi lần áp lệnh hở vòng sau
  đó, nên vận tốc thực thi = hồ sơ kế hoạch + δ — mà hồ sơ của D4
  tự nó đã chạy sát hộp |v_i| ≤ v_max = 3 — và mọi trạng thái neo
  sau đó (truyền từ trạng thái thật) đều thừa hưởng δ: Địa tái tạo
  bằng trễ kịch bản hoá (một chu kỳ chậm 2,32 s giữa các chu kỳ
  nhanh) được 14 chu kỳ sập cùng chữ ký iter=1/slack=∞, v_anchor
  tới 4,38 m/s và vận tốc THỰC THI tới 4,39 m/s — quỹ đạo thực
  thi tự nó vi phạm (R3) dù mọi kế hoạch riêng lẻ đều hợp lệ
  (review_dia_tinh_tro_bugB.md mục 5; diagnosis_second_
  infeasibility.md). Đây là khuyết tật mô hình hoá của harness
  (thiếu phản hồi trạng thái khi áp kế hoạch), không phải lỗi
  planner: một bộ điều khiển tầng thấp thật bám theo quỹ đạo kế
  hoạch bằng phản hồi, nên lệch bị khử dần thay vì cộng dồn.
  Bản sửa: thực thi kế hoạch hiện hành có PHẢN HỒI VẬN TỐC theo
  trạng thái thật — tại nút k của kế hoạch hiện hành, lệnh áp là

      u = clip_{|·|≤u_max}( U[k] + Kv·(V[k] − v) ),   Kv = 4 1/s

  (clip theo từng thành phần về hộp |u_i| ≤ u_max — cùng hộp
  (R2) của planner; v là vận tốc THẬT tại đầu đoạn áp lệnh, V[k]
  là vận tốc tham chiếu của chính kế hoạch). Luật này áp ở cả ba
  nơi dùng lệnh của kế hoạch hiện hành: bước áp kế hoạch mới
  (k = 0), bay dọc kế hoạch trong pha trễ (fly_along_plan) và
  truyền trạng thái neo (propagate_along_plan) — neo vì thế là
  dự phóng của đúng luật thực thi sẽ áp, nhất quán hai phía.
  Ngoài chân trời kế hoạch vẫn dùng quy tắc hãm bão hòa §7.1
  như cũ. Hệ quả mô hình: lệch vận tốc neo–thực thi suy giảm mũ
  với thời hằng 1/Kv = 0,25 s thay vì hằng số vĩnh viễn; vận tốc
  thực thi bị kéo về hồ sơ vận tốc của chính kế hoạch — tức
  không vượt quá những gì dãy lệnh cộng động học cho phép, ngoài
  dung sai bộ giải của chính hồ sơ. Lệch VỊ TRÍ so với hồ sơ
  không được phản hồi đuổi theo: nó được vòng replan hấp thụ ở
  chu kỳ sau (mọi kế hoạch đều neo trên trạng thái thật dự
  phóng), đúng ngữ nghĩa "thực thi bước đầu" của Bài 1.
  Hai biến thể đã thử và LOẠI bằng số trước khi chốt luật này:
    - Bám PD cả vị trí lẫn vận tốc (Kp = 9, Kv = 6): ở mẫu hình
      ngược dấu τ̂ ≫ T_latency (chu kỳ đầu chậm 6,79 s do khởi
      tạo trên VM tranh chấp, chu kỳ sau nhanh 0,60 s), kế hoạch
      neo ở trạng thái dự phóng 6,79 s trong tương lai nhưng áp
      gần như ngay; hạng tử vị trí đuổi theo tham chiếu ở xa
      phía trước, bão hoà ở u_max và TỰ NÓ đẩy neo chu kỳ sau
      tới 4,01 m/s → sập + va chạm. Ca đối chứng có kiểm soát
      bằng trễ kịch bản hoá [6,79; 0,60; 0,22…] trên D2 seed
      9000000: bản hở vòng sạch hoàn toàn (0 sập, khe hở
      0,706 m), bản PD-đầy-đủ va chạm −0,350 m; cùng cơ chế ở lô
      smoke PD-đầy-đủ: D3 seed 9000002 neo tới 7,02 m/s rồi va
      chạm, D2 seed 9000000 neo 4,00 m/s, 1 sập rồi va chạm.
    - Áp nhất quán thời gian (giữ kế hoạch tới thời điểm neo khi
      tới sớm + nhập tại nút theo tuổi khi tới muộn) kết hợp PD
      đầy đủ: ở ca kịch bản hoá trên, chu kỳ đầu tới muộn 6,79 s
      làm kế hoạch đầu bị nhập ở nút cuối và UAV gần như đứng
      yên cạnh điểm xuất phát trong ~13 s — vật cản đối đầu D2
      tới nơi trong lúc chờ → va chạm −0,122 m dù 0 chu kỳ sập.
      Ngữ nghĩa "chờ tới thời điểm neo" lãng phí lần tính xong
      sớm và không phải cách hệ MPC thật xử lý τ̂ sai lệch.
  Mô hình trễ có bù trễ (τ̂ = T_latency chu kỳ trước) và planner
  giữ nguyên; chế độ frozen không qua đường thực thi này.

  Vì sao không dùng hai mô hình đơn giản hơn (đã thử, đã loại bằng số):
    - "Giữ nguyên gia tốc cũ suốt τ": mô hình rơm — một lần giữ u
      trong 5,18 s đưa UAV đi hơn 1 km.
    - "Áp u_0 của kế hoạch mới ngay khi tính xong, không bù trễ":
      lệch trạng thái tích lũy làm tổng |v_x|+|v_y|+|v_z| vượt 3 —
      ngưỡng vô nghiệm của ràng buộc cp.norm(V,'inf') trong
      dca_planner.py (xem README_HA_TANG.md, mục phát hiện hạ tầng) —
      khiến bộ giải thất bại 158/163 chu kỳ và UAV trôi 296 m. Mô
      hình bù trễ ở trên là cách hệ MPC thật xử lý độ trễ tính toán,
      và là mô hình duy nhất đo được tác động của trễ lên an toàn
      thay vì đo một giả tạo của vòng lặp.
    - "Thực thi hở vòng đúng dãy U của kế hoạch" (bản trước của
      chính file này): xem mục THỰC THI BÁM TRẠNG THÁI ở trên —
      lệch vận tốc do lệch τ̂ ↔ T_latency không bao giờ được khử,
      vận tốc thực thi và trạng thái neo vượt hộp (R3) tới 4,39 m/s
      dưới jitter trễ (tái tạo của Địa). Đã thay bằng luật phản
      hồi vận tốc theo trạng thái thật (hai biến thể bám mạnh hơn
      — PD đầy đủ và áp nhất quán thời gian — cũng đã thử và loại
      bằng số, xem cùng mục).

Đo lường (cả hai chế độ, cùng một cách):
  - Khe hở thật LIÊN TỤC: mẫu mỗi SUBSTEP = 0,025 s (40 Hz, 10 mẫu
    mỗi bước điều khiển) dọc mọi đoạn đã bay — kể cả đoạn chờ giải —
    vị trí UAV theo tích phân kép chính xác, vật cản theo sự thật nền
    tại đúng thời điểm mẫu. Va chạm khi khe hở < 0; episode dừng.
    (Đã kiểm chứng tương đương lưới φ của §1.3 tới < 0,1 mm.)
  - Chế độ frozen còn tính khe hở "legacy" theo cách cũ (mẫu tại
    frac 0,5 và 1,0) để đối chiếu trực tiếp số đã công bố của Bài 1.
  - Jerk RMS (§1.6): dãy lệnh thực thi được LẤY MẪU LẠI trên lưới Δt
    đều (u_k = lệnh đang có hiệu lực tại đầu bước k), rồi
    jerk_k = (u_{k+1} − u_k)/Δt, RMS trên k. (Cách cũ — hiệu hữu hạn
    theo thời điểm áp lệnh thật — phình tới 3,15 lần ở chế độ
    realtime vì các chunk một phần làm mẫu số co lại; đã sửa. Ở chế
    độ frozen hai cách trùng nhau vì lệnh chỉ đổi tại biên bước Δt.)
  - Năng lượng (proxy, §1.6): Σ ||u||^2 * (thời lượng áp lệnh) — dạng
    tích phân của Δt·Σ||u_k||², đã đối chiếu khớp §1.6 (mục A6).
  - Độ dài đường bay: tổng độ dài đoạn con ở độ phân giải 0,025 s
    (lệch nhỏ so với §1.5 — xem DEVIATIONS.md, mục A7).
  - Thành công = tới đích (≤ goal_tol và ||v|| ≤ 0,6) VÀ không va
    chạm; va chạm ghi tách riêng; THÀNH CÔNG AN TOÀN (§1.4) = tới
    đích VÀ clearance_min ≥ d = 0,5 m — chỉ số đầu ra chính thứ hai.
  - Mã trạng thái episode (§6.4): ok / solver_fail / timeout /
    crash / infra, suy từ kết cục episode (xem cuối run_episode);
    status_note giữ văn bản chi tiết. Episode không bay được bước
    nào bị đánh dấu no_flight (§6.4.2).
  - Trần: t_max mô phỏng theo họ kịch bản (§2: họ D = 40 s cho MỌI
    mô hình thời gian; họ S/F và loạt A/B = 60 s) và trần wall-clock
    episode (§7.3: 20 phút cho họ S/D/F) sinh mã timeout. Lưu ý: một
    lời gọi plan() đơn lẻ không ngắt được — trần wall-clock được
    kiểm tra tại đầu mỗi chu kỳ.

  CHẾ ĐỘ QUAN SÁT ELLIPSOID (hook get_obs, bổ sung 2026-11-22;
  tham số obs_mode của run_episode, mặc định "sphere"):
  Mặc định vòng lặp quan sát vật cản dưới dạng CẦU (get_obs: truth /
  kalman / frozen của obstacle_predict) — hành vi cũ giữ nguyên
  tuyệt đối ở obs_mode = "sphere". Với obs_mode = "ellipsoid",
  get_obs trả các dict ellipsoid ĐÚNG định dạng đầu ra của tầng
  hình dạng (shape_forecast: centers / axes / yaws / rho, bộ giải
  bỏ qua các khoá chẩn đoán phụ) theo cùng ba chế độ mode:
    - truth  -> shape_forecast.truth_ellipsoid_obstacles (ρ = 0);
    - kalman -> shape_forecast.kalman_ellipsoid_obstacles: lịch sử
      đo nhiễu (tâm std MEAS_STD, log-bán-trục std LOG_MEAS_STD,
      yaw std YAW_MEAS_STD — đúng mô hình nhiễu/giả thiết đầu vào
      mà tầng hình dạng tiêu thụ) -> dự báo hai lớp có guardrail;
    - frozen -> trạng thái ellipsoid tại thời điểm neo giữ nguyên
      trên cả chân trời, ρ = 0 (đúng ngữ nghĩa frozen của nhánh
      cầu: tuple tĩnh tại t_anchor).
  Nguồn hàm sự thật ellipsoid: kịch bản có ellipsoid_truth_fns
  trong sổ đăng ký scenarios_pkg (họ F, F1-MS) dùng chính các hàm
  đó (config ghi ellipsoid_obs_source = "registry"); mọi kịch bản
  còn lại SUY BIẾN từ hàm sự thật cầu của chính nó — a = (r, r, r),
  yaw = 0 ("degenerate_from_sphere") — nên bật ellipsoid trên họ
  S/D/A/B cho planner cùng một thế giới, chỉ khác biểu diễn.
  GIỚI HẠN CỐ Ý của hook này: nó chỉ đổi QUAN SÁT đưa vào planner.
  Sự thật nền để đo khe hở/va chạm trong vòng lặp vẫn là truth_fns
  dạng cầu của kịch bản — với họ F đó là hình chiếu cầu
  thể-tích-tương-đương theo §2.7/Q6 (giao diện đã đóng băng của
  vòng lặp); khe hở so với bề mặt ellipsoid thật thuộc tầng đánh
  giá (ví dụ scenarios_pkg.validate.ellipsoid_signed_distance),
  không đổi tại đây để giữ tính so sánh của mọi chỉ số đã kiểm
  chứng. Chi tiết vận hành: README_HA_TANG.md §3.4.

  DỰ PHÒNG HẠT GIỐNG DV-H10 (bổ sung 2026-11-23 theo phán quyết
  vòng kiểm chứng SOTA, adjudication_sota_verify.md mục 6; mã
  deviation MỚI — không phải phụ lục DV-H9, vì DV-H9 đã hiệu lực
  qua giao thức 2.0 đã ký): tham số seed_fallback của run_episode,
  mặc định False (tắt — hành vi và schema log giữ nguyên tuyệt
  đối); experiment_runner bật True cho A0, là cấu hình duy nhất
  thuộc đối tượng của DV-H10 (A1–A3 theo đường hạt giống DV-H8
  của method_adapter; baseline Tier-1 có giao diện hạt giống
  đóng băng theo SPEC_cong_bang §5 — Ngo2023/MPCQP có tiêu thụ
  seed_P nhưng không thuộc phạm vi DV-H10, CBF2021 không tiêu
  thụ seed_P). Khi bật: vị từ kích hoạt v1.1 (phán bổ sung
  DV-H10, adjudication_sota_verify.md mục BS-1 — thay công thức
  v1.0 "một lần gọi bất kỳ > 20 vòng", vốn kích hoạt cả ở các
  episode khỏe có lần gọi đơn lẻ đắt) chỉ đánh giá trong 3 lần
  gọi đầu của episode: kích hoạt nếu (a) lần gọi chu kỳ 0 chạm
  trần vòng lặp SEED_FALLBACK_ITERATION_CAP (105 vòng), hoặc
  (b) lần gọi chu kỳ 0 dùng > SEED_FALLBACK_MAX_ITERATIONS
  (20) vòng VÀ ít nhất một trong hai lần gọi chu kỳ 1/2 cũng
  dùng > 20 vòng. Kích hoạt tại thời điểm điều kiện thành
  đúng; từ chu kỳ KẾ TIẾP tới hết episode, hạt giống đưa vào
  plan() là hạt giống A* — astar_seed của script gốc Bài 1
  với thông số mặc định như
  mã gốc (d_safe, z_floor lấy từ PlannerParams của episode; mọi
  thông số còn lại để mặc định của module) — tính tại trạng thái
  neo trên ảnh chụp vật cản ở bước 0 của dự báo hiện hành, thay
  cho hạt giống thẳng / khởi động ấm. Kích hoạt một lần là CHỐT
  (latched) tới hết episode. astar_seed trả None (không có vật
  cản / không tìm được đường) thì chu kỳ đó giữ hạt giống cũ —
  đúng hợp đồng "phía gọi giữ nguyên hành vi cũ" của module —
  và chốt vẫn giữ. Hai trường log bắt buộc ghi vào metrics:
  seed_fallback_triggered (bool) và seed_fallback_cycle (chỉ số
  chu kỳ mà lần gọi vượt ngưỡng đã xảy ra); config ghi
  seed_fallback_dvh10 + seed_fallback_max_iterations. Số vòng
  lặp từng lần gọi vốn đã có trong bản ghi chu kỳ (iterations).
"""
from __future__ import annotations

import dataclasses
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# (Cây package 2026-11-24: module này thuộc gói sim/ trong cây
# refactor của người dùng; mọi import bên dưới là tuyệt đối
# core.*/sim.* — thư mục code/ phải nằm trên sys.path, do phía
# gọi bảo đảm: runner, script TN và test đều chèn sẵn.)
# LƯU Ý NẠP PLANNER (sửa 2026-11-22 — xem inertness_bugB.md, phụ
# lục truy vết): file này KHÔNG tự chèn đường dẫn nào lên sys.path
# (cơ chế cũ làm mọi lần chạy qua runner nạp bản planner GỐC thay
# vì bản làm việc, âm thầm). `from core.dca_planner import ...`
# dưới đây trỏ tới: (a) module mà experiments/experiment_runner đã nạp
# MINH vào sys.modules["core.dca_planner"] từ --planner (khi chạy
# qua runner), hoặc (b) core/dca_planner.py của cây này (khi
# import trực tiếp, ví dụ từ test). Không còn đường nạp ngầm nào
# trỏ về bản gốc.

from core.astar_seed import astar_seed                     # noqa: E402
from core.dca_planner import PlannerParams, plan          # noqa: E402
from sim.dynamic_scenarios import dynamic_scenario        # noqa: E402
from core.obstacle_predict import kalman_obstacles, truth_obstacles  # noqa: E402
from core.shape_forecast import (kalman_ellipsoid_obstacles,  # noqa: E402
                            truth_ellipsoid_obstacles)

SUBSTEP = 0.025        # s — mật độ lấy mẫu khe hở liên tục (40 Hz)
BRAKE_V_EPS = 0.05     # m/s — ngưỡng ‖v‖ coi như đã dừng (§7.1)
TRACK_KV = 4.0         # 1/s — hệ số phản hồi vận tốc (mục THỰC THI BÁM TRẠNG THÁI)
# Điều kiện ổn định rời rạc của luật phản hồi vận tốc (Địa, rà soát
# bản sửa harness 2026-11-22): ánh xạ sai số vận tốc sau một bước
# đầy đủ có hệ số (1 − Kv·Δt) — sai số co đơn điệu khi
# 0 < Kv·Δt < 2. Với Δt = 0,25 s (PlannerParams mặc định),
# Kv·Δt = 1,0: deadbeat ở bước đầy đủ, hệ số thuộc [0; 0,9] ở các
# chunk một phần. KHI ĐỔI Δt phải giữ Kv·Δt trong khoảng (0, 2) —
# giữ Kv = 4 với Δt ≥ 0,5 s sẽ làm luật phản hồi mất ổn định.
SAFETY_D = 0.5         # m — biên an toàn d cho "thành công an toàn" (§1.4)
WALL_CLOCK_CAP_S = 20 * 60.0   # s — trần wall-clock episode họ S/D/F (§7.3)
SEED_FALLBACK_MAX_ITERATIONS = 20
SEED_FALLBACK_ITERATION_CAP = 105
# Ngưỡng kích hoạt DV-H10 (phán quyết vòng kiểm chứng SOTA
# 2026-11-23, exp/reviews/adjudication_sota_verify.md mục 6 +
# phán bổ sung DV-H10 v1.1 mục BS-1): vị từ kích hoạt chỉ xét 3
# lần gọi đầu của episode — (a) lần gọi chu kỳ 0 chạm trần
# SEED_FALLBACK_ITERATION_CAP (105 vòng = 7 mức phạt × 15 vòng
# của planner), hoặc (b) lần gọi chu kỳ 0 dùng
# > SEED_FALLBACK_MAX_ITERATIONS vòng và ít nhất một trong hai
# lần gọi chu kỳ 1/2 cũng > ngưỡng đó. Kích hoạt thì từ chu kỳ
# kế tiếp, hạt giống của episode chuyển sang A* và CHỐT tới
# hết episode; qua chu kỳ 2 mà chưa kích hoạt thì cơ chế tắt
# cho phần còn lại của episode. Căn cứ tách lớp (BS-1): chu kỳ
# 0 đắt là bình thường (không warm-start; episode khỏe tới
# 30–45 vòng) nhưng episode khỏe rơi về ≤ 15 vòng ngay ở chu
# kỳ 1–2, còn mọi ca phình đều đắt ở chu kỳ 0 VÀ còn đắt ở chu
# kỳ 1 hoặc 2. Cơ chế chỉ hiệu lực khi run_episode được bật
# seed_fallback=True — experiment_runner bật cho A0 (xem mục
# DỰ PHÒNG HẠT GIỐNG DV-H10 ở docstring đầu file).


def default_t_max(scenario_name: str) -> float:
    """t_max mô phỏng theo họ kịch bản (§2): họ D = 40 s; còn lại 60 s."""
    return 40.0 if scenario_name.startswith("D") else 60.0


def _resolve_scenario(name: str):
    """Chọn kịch bản theo tên cho run_episode (tích hợp 2026-11-22).

    Họ D — và mọi tên dynamic_scenarios đã biết, kể cả kịch bản
    tĩnh do driver khác đăng ký vào _SCENARIOS — giữ ĐÚNG hành vi
    cũ: dict của dynamic_scenarios, entry = None. Mọi họ còn lại
    (S + S1L, loạt A, loạt B, họ F — tổng 49 tên) qua sổ đăng ký
    thống nhất scenarios_pkg.registry: resolve_for_runner(name)
    cho dict {name, start, goal, truth_fns} mà vòng lặp tiêu thụ
    (họ F ở giao diện cầu thể-tích-tương-đương §2.7/Q6), và
    get_scenario(name) cho bản ghi đầy đủ (họ, platform, t_max
    đăng ký). Trả (sc_dict, entry | None).
    """
    try:
        return dynamic_scenario(name), None
    except KeyError:
        pass
    from sim.scenarios_pkg.registry import get_scenario, resolve_for_runner
    return resolve_for_runner(name), get_scenario(name)


def degenerate_ellipsoid_fns(truth_fns):
    """Hàm sự thật ellipsoid SUY BIẾN từ hàm sự thật cầu: cùng tâm,
    bán trục (r, r, r), yaw = 0 tại mọi t. Dùng cho các kịch bản
    không khai ellipsoid_truth_fns trong sổ đăng ký (họ S/D/A/B)
    khi bật chế độ quan sát ellipsoid — planner thấy cùng một thế
    giới, chỉ khác biểu diễn hình học."""
    out = []
    for fn in truth_fns:
        def make(fn):
            def g(t):
                c, r = fn(t)
                return (np.asarray(c, dtype=float).copy(),
                        np.full(3, float(r)), 0.0)
            return g
        out.append(make(fn))
    return out


def frozen_ellipsoid_obstacles(ell_fns, t_anchor: float, K: int) -> list:
    """Quan sát ellipsoid ở chế độ frozen: trạng thái (tâm, bán
    trục, yaw) tại thời điểm neo t_anchor giữ nguyên trên cả chân
    trời, ρ = 0 — đúng ngữ nghĩa của nhánh frozen cầu (tuple tĩnh
    tại t_anchor). Dict đúng định dạng tầng hình dạng."""
    out = []
    for fn in ell_fns:
        c, a, psi = fn(t_anchor)
        out.append(dict(
            centers=np.repeat(
                np.asarray(c, dtype=float).reshape(1, 3), K + 1, axis=0),
            axes=np.repeat(
                np.asarray(a, dtype=float).reshape(1, 3), K + 1, axis=0),
            yaws=np.full(K + 1, float(psi)),
            rho=np.zeros(K + 1)))
    return out


def ellipsoid_obstacles(mode: str, ell_fns, t_anchor: float, K: int,
                        dt: float, rng: np.random.Generator) -> list:
    """Quan sát ellipsoid cho get_obs theo chế độ mode của vòng lặp
    (truth / kalman / frozen) — xem mục CHẾ ĐỘ QUAN SÁT ELLIPSOID
    ở docstring đầu file. Đầu ra là dict của tầng hình dạng
    (shape_forecast), không định dạng thứ hai nào khác."""
    if mode == "truth":
        return truth_ellipsoid_obstacles(ell_fns, t_anchor, K, dt)
    if mode == "kalman":
        return kalman_ellipsoid_obstacles(ell_fns, t_anchor, K, dt, rng)
    if mode == "frozen":
        return frozen_ellipsoid_obstacles(ell_fns, t_anchor, K)
    raise ValueError(f"chế độ không rõ: {mode}")


def brake_command(v: np.ndarray, dt: float, u_max: float) -> np.ndarray:
    """Lệnh hãm bão hòa THEO CHUẨN của §7.1 cho một bước Δt.

    u = −(v/Δt)·min(1, u_max/‖v/Δt‖); ‖v‖ ≤ 0,05 ⇒ u = 0 (giữ vị trí).
    Bảo đảm ‖u‖ ≤ u_max ở mọi bước.
    """
    v = np.asarray(v, dtype=float)
    if np.linalg.norm(v) <= BRAKE_V_EPS:
        return np.zeros(3)
    w = v / dt
    n = np.linalg.norm(w)
    if n <= 0.0:
        return np.zeros(3)
    return -w * min(1.0, u_max / n)


def track_command(u_ff, ref_v, v, u_max):
    """Lệnh thực thi có PHẢN HỒI VẬN TỐC theo trạng thái thật.

    u = U[k] + Kv·(V[k] − v), clip theo từng thành phần về hộp
    |u_i| ≤ u_max (cùng hộp (R2) của planner). v là vận tốc THẬT
    tại đầu đoạn áp lệnh; V[k] là vận tốc tham chiếu của kế hoạch
    hiện hành tại nút k. Khi vận tốc thật trùng tham chiếu, lệnh
    rút về đúng U[k] của kế hoạch. Chỉ kênh vận tốc được phản hồi
    — kênh vị trí cố ý KHÔNG đuổi theo tham chiếu (bằng chứng ở
    mục THỰC THI BÁM TRẠNG THÁI trong docstring đầu file: hệ số
    vị trí làm luật bám đuổi một tham chiếu neo xa và tự sinh
    vận tốc ngoài hộp ở mẫu hình τ̂ ≫ T_latency).
    """
    u = (np.asarray(u_ff, dtype=float)
         + TRACK_KV * (np.asarray(ref_v, dtype=float)
                       - np.asarray(v, dtype=float)))
    return np.clip(u, -u_max, u_max)


def resample_controls(segments, dt: float, t_end: float):
    """Lấy mẫu lại dãy lệnh thực thi trên lưới Δt đều (§1.6).

    segments: danh sách (t_bắt_đầu, thời_lượng, u) phủ [0, t_end].
    Trả mảng (N, 3): u_k là lệnh đang có hiệu lực tại đầu bước k
    (thời điểm k·Δt). Bước cuối một phần vẫn tính là một bước.
    """
    if t_end <= 0.0 or not segments:
        return np.zeros((0, 3))
    n_steps = int(np.ceil(t_end / dt - 1e-12))
    grid = np.zeros((n_steps, 3))
    segs = sorted(segments, key=lambda s: s[0])
    j = 0
    for k in range(n_steps):
        t = k * dt
        while j + 1 < len(segs) and segs[j + 1][0] <= t + 1e-12:
            j += 1
        grid[k] = segs[j][2]
    return grid


def jerk_rms_on_grid(segments, dt: float, t_end: float) -> float:
    """jerk RMS theo §1.6 trên dãy u thực thi lấy mẫu lại theo lưới Δt."""
    grid = resample_controls(segments, dt, t_end)
    if len(grid) < 2:
        return 0.0
    jerks = np.diff(grid, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(jerks ** 2, axis=1))))


def _propagate(p, v, u, h):
    """Tích phân kép chính xác trong h giây dưới gia tốc hằng u."""
    return p + h * v + 0.5 * h * h * u, v + h * u


def _fly_segment(p, v, u, duration, t0, truth_fns, state):
    """Bay một đoạn duration giây dưới lệnh u bắt đầu tại thời gian t0.

    Cập nhật state: min_clear, path_len, energy. Trả (p, v, t_end, collided).
    """
    if duration <= 0.0:
        return p, v, t0, False
    n = max(1, int(np.ceil(duration / SUBSTEP)))
    h = duration / n
    pc, vc = p.copy(), v.copy()
    collided = False
    for i in range(n):
        pn, vn = _propagate(pc, vc, u, h)
        state["path_len"] += float(np.linalg.norm(pn - pc))
        for pp, tt in ((pc, t0 + i * h), (pn, t0 + (i + 1) * h)):
            for fn in truth_fns:
                o, r = fn(tt)
                c = float(np.linalg.norm(pp - o) - r)
                if c < state["min_clear"]:
                    state["min_clear"] = c
                if c < 0.0:
                    collided = True
        pc, vc = pn, vn
    state["energy"] += float(np.dot(u, u)) * duration
    return pc, vc, t0 + duration, collided


def _astar_snapshot_obstacles(obs) -> list:
    """Ảnh chụp vật cản TĨNH tại thời điểm neo cho hạt giống A*
    của DV-H10: mỗi vật một cặp (tâm, bán kính) ở bước 0 của dự
    báo hiện hành — đúng ngữ nghĩa danh sách vật cản mà script
    gốc Bài 1 (simulate.py) truyền cho astar_seed. Dict cầu lấy
    centers[0] / radii[0]; dict ellipsoid lấy centers[0] và bán
    kính bao = bán trục lớn nhất ở bước 0; tuple (o, r) của chế
    độ quan sát frozen giữ nguyên."""
    snap = []
    for ob in obs:
        if isinstance(ob, dict):
            c = np.asarray(ob["centers"][0], dtype=float)
            if "radii" in ob:
                r = float(np.asarray(ob["radii"], dtype=float)[0])
            else:
                r = float(np.max(np.asarray(ob["axes"][0], dtype=float)))
            snap.append((c, r))
        else:
            o, r = ob
            snap.append((np.asarray(o, dtype=float), float(r)))
    return snap


def run_episode(name: str, mode: str = "kalman", seed: int = 0,
                prm: PlannerParams | None = None,
                time_model: str = "realtime",
                t_max: float | None = None,
                goal_tol: float = 0.35,
                safety_d: float = SAFETY_D,
                wall_clock_cap_s: float | None = None,
                diag_sink: list | None = None,
                anchor_sink: list | None = None,
                obs_mode: str = "sphere",
                seed_fallback: bool = False) -> dict:
    """Chạy một episode vòng kín. time_model: 'realtime' | 'frozen'.

    obs_mode: 'sphere' (mặc định — quan sát cầu của
    obstacle_predict, hành vi cũ giữ nguyên tuyệt đối) hoặc
    'ellipsoid' (get_obs trả dict ellipsoid của tầng hình dạng —
    xem mục CHẾ ĐỘ QUAN SÁT ELLIPSOID ở docstring đầu file và
    README_HA_TANG.md §3.4). Nguồn hàm sự thật ellipsoid và chế
    độ được ghi vào config trả về (obs_mode, ellipsoid_obs_source).

    Kịch bản chọn theo TÊN qua _resolve_scenario: họ D giữ đường
    dynamic_scenarios như cũ; họ S, loạt A, loạt B và họ F qua sổ
    đăng ký scenarios_pkg.registry (họ F ở giao diện cầu
    thể-tích-tương-đương §2.7/Q6). Kịch bản loạt B mang khóa
    platform (tên, v_max, u_max) của §2.4: hai giới hạn này được
    ghi đè vào bản sao của prm cho episode đó và ghi vào config
    trả về (scenario_platform, planner_effective).

    t_max mặc định theo họ kịch bản (§2): họ D = 40 s cho cả hai mô
    hình thời gian (bản cũ cho realtime 90 s — đã sửa theo rà soát);
    họ khác 60 s. wall_clock_cap_s mặc định 20 phút (§7.3, họ S/D/F).

    diag_sink: nếu là một list, mỗi chu kỳ bộ giải THẤT BẠI
    (total_slack = inf) được ghi thêm một bản ghi chẩn đoán đầy đủ
    (trạng thái neo, v0, đầu vào plan(), kết quả solver) vào list
    đó — runner ghi ra file log phụ *_diag.json, schema log chính
    không đổi. Bản ghi cũng trả về ở khoá "diagnostics".

    anchor_sink: nếu là một list, MỌI lần gọi plan() (không chỉ
    chu kỳ sập) được ghi một bản ghi trạng thái neo (p_anchor,
    v_anchor đầy đủ, phân loại hộp theo dung sai tiền định của
    QUYET_DINH_THIET_KE.md Quyết định 1 §1.5 Ô 1: trong-hộp khi
    max_i |v0_i| <= v_max; vùng xám (v_max, v_max + 3e-3]; ngoài
    hộp khi > v_max + 3e-3) vào list đó — runner ghi ra file log
    phụ *_anchors.json khi bật cờ --anchor-log, schema log chính
    không đổi. Bản ghi cũng trả về ở khoá "anchor_calls". Đây là
    điều kiện tiên quyết hạ tầng của Ô 1 DV-H4: tiền đề trong-hộp
    phải trích xuất được từ log của cả hai lô trước/sau. Ghi thuần
    túy quan sát — không đổi hành vi vòng lặp.

    seed_fallback: cơ chế dự phòng hạt giống DV-H10 (mặc định
    False = tắt; xem mục DỰ PHÒNG HẠT GIỐNG DV-H10 ở docstring
    đầu file để biết đặc tả đầy đủ, đối tượng áp dụng và hai
    trường log bắt buộc). experiment_runner chỉ bật cho A0.
    """
    if time_model not in ("realtime", "frozen"):
        raise ValueError(f"time_model không rõ: {time_model}")
    if obs_mode not in ("sphere", "ellipsoid"):
        raise ValueError(f"obs_mode không rõ: {obs_mode}")
    t_max_source = "explicit" if t_max is not None else "family_default"
    if t_max is None:
        t_max = default_t_max(name)
    if wall_clock_cap_s is None:
        wall_clock_cap_s = WALL_CLOCK_CAP_S
    prm = prm or PlannerParams()
    sc, sc_entry = _resolve_scenario(name)
    platform = sc_entry.get("platform") if sc_entry else None
    if platform is not None:
        # Loạt B (§2.4): nền tảng mang giới hạn bay riêng
        # (v_max, u_max) — ghi đè vào BẢN SAO của prm, không sửa
        # đối tượng của người gọi; bộ tham số hiệu lực được ghi
        # vào config trả về (planner_effective, scenario_platform).
        prm = dataclasses.replace(prm, v_max=float(platform[1]),
                                  u_max=float(platform[2]))
    dt = prm.dt
    u_max = prm.u_max
    truth_fns = sc["truth_fns"]
    # Nguồn hàm sự thật ellipsoid cho chế độ quan sát ellipsoid:
    # bản ghi sổ đăng ký có ellipsoid_truth_fns (họ F, F1-MS) thì
    # dùng chính nó; còn lại suy biến từ hàm sự thật cầu của kịch
    # bản. Ở obs_mode = "sphere" hai biến này không được dùng.
    ell_fns = None
    ellipsoid_obs_source = None
    if obs_mode == "ellipsoid":
        reg_ell = (sc_entry.get("ellipsoid_truth_fns")
                   if sc_entry else None)
        if reg_ell:
            ell_fns = reg_ell
            ellipsoid_obs_source = "registry"
        else:
            ell_fns = degenerate_ellipsoid_fns(truth_fns)
            ellipsoid_obs_source = "degenerate_from_sphere"
    goal = sc["goal"]
    p = sc["start"].astype(float).copy()
    v = np.zeros(3)
    rng = np.random.default_rng(seed)
    wall_start = time.perf_counter()

    state = dict(min_clear=np.inf, path_len=0.0, energy=0.0)
    min_clear_legacy = np.inf
    cycles = []
    segments = []           # (t_start, duration, u) mọi đoạn lệnh đã thực thi
    cur_U = None            # kế hoạch hiện hành (dãy lệnh (K,3))
    cur_P = None            # quỹ đạo vị trí tham chiếu của kế hoạch hiện hành
    cur_V = None            # hồ sơ vận tốc tham chiếu của kế hoạch hiện hành
    cur_offset = 0.0        # số giây đã tiêu thụ trong kế hoạch hiện hành
    seed_P = None
    fb_triggered = False    # DV-H10: chốt dự phòng A* đã bật hay chưa
    fb_cycle = None         # DV-H10: chu kỳ mà điều kiện kích hoạt thành đúng
    fb_iter_cycle0 = None   # DV-H10 v1.1: số vòng lặp của lần gọi chu kỳ 0
    prev_latency = 0.0      # T_latency chu kỳ trước (cho τ̂, quy ước H1)
    t_now = 0.0
    reached = False
    collided = False
    status_note = "t_max"

    def propagate_along_plan(p0, v0, duration):
        """Truyền trạng thái dọc kế hoạch hiện hành (không bay thật).

        Trong chân trời kế hoạch dùng đúng luật phản hồi vận tốc
        (track_command) như đường thực thi thật — neo là dự phóng
        của luật sẽ áp. Đoạn vượt chân trời dùng quy tắc hãm bão
        hòa §7.1 theo từng bước Δt — nhất quán với đường thực thi
        thật.
        """
        pc, vc, off, rem = p0.copy(), v0.copy(), cur_offset, duration
        while rem > 1e-12:
            if cur_U is not None and off < len(cur_U) * dt - 1e-12:
                idx = min(int(off // dt), len(cur_U) - 1)
                chunk = min(rem, (idx + 1) * dt - off)
                u = track_command(cur_U[idx], cur_V[idx], vc, u_max)
            else:
                chunk = min(rem, dt)
                u = brake_command(vc, dt, u_max)
            pc, vc = _propagate(pc, vc, u, chunk)
            off += chunk
            rem -= chunk
        return pc, vc

    def fly_along_plan(duration):
        """Bay thật dọc kế hoạch hiện hành, cập nhật mọi sổ sách.

        Bị chặn bởi t_max: đồng hồ mô phỏng không vượt t_max.
        Trả số giây bay NGOÀI chân trời kế hoạch (delay_beyond_plan)
        — nay là bay dưới quy tắc hãm §7.1, không còn trôi tự do.
        """
        nonlocal p, v, t_now, cur_offset, collided
        beyond, rem = 0.0, duration
        while rem > 1e-12 and t_now < t_max - 1e-12:
            room = t_max - t_now
            if cur_U is not None and cur_offset < len(cur_U) * dt - 1e-12:
                idx = min(int(cur_offset // dt), len(cur_U) - 1)
                chunk = min(rem, (idx + 1) * dt - cur_offset, room)
                u = track_command(cur_U[idx], cur_V[idx], v, u_max)
                is_beyond = False
            else:
                chunk = min(rem, dt, room)
                u = brake_command(v, dt, u_max)
                is_beyond = True
            t_seg = t_now
            p, v, t_now, hit = _fly_segment(p, v, u, chunk, t_now,
                                            truth_fns, state)
            segments.append((t_seg, chunk, u.copy()))
            if is_beyond:
                beyond += chunk
            cur_offset += chunk
            rem -= chunk
            if hit:
                collided = True
                break
        return beyond

    def fly_one_step(u, duration):
        """Bay một đoạn lệnh đơn (bước áp kế hoạch mới / bước dự phòng),
        bị chặn bởi t_max. Trả True nếu va chạm."""
        nonlocal p, v, t_now, collided
        dur = min(duration, max(0.0, t_max - t_now))
        if dur <= 0.0:
            return False
        t_seg = t_now
        p, v, t_now, hit = _fly_segment(p, v, u, dur, t_now,
                                        truth_fns, state)
        segments.append((t_seg, dur, np.asarray(u, dtype=float).copy()))
        collided = collided or hit
        return hit

    def get_obs(t_anchor):
        if obs_mode == "ellipsoid":
            return ellipsoid_obstacles(mode, ell_fns, t_anchor,
                                       prm.K, dt, rng)
        if mode == "truth":
            return truth_obstacles(truth_fns, t_anchor, prm.K, dt)
        if mode == "kalman":
            return kalman_obstacles(truth_fns, t_anchor, prm.K, dt, rng)
        if mode == "frozen":
            return [(fn(t_anchor)[0], fn(t_anchor)[1]) for fn in truth_fns]
        raise ValueError(f"chế độ không rõ: {mode}")

    while t_now < t_max - 1e-12:
        # --- Trần wall-clock episode (§7.3), kiểm tra tại đầu chu kỳ ---
        if time.perf_counter() - wall_start > wall_clock_cap_s:
            status_note = "wall_clock_timeout"
            break
        t_cycle_start = t_now
        tau_hat = prev_latency if time_model == "realtime" else 0.0

        # --- Neo: trạng thái + quan sát tại t_c + τ̂ (realtime) ---
        if time_model == "realtime" and tau_hat > 0.0:
            p_anchor, v_anchor = propagate_along_plan(p, v, tau_hat)
            t_anchor = t_now + tau_hat
            if cur_P is not None:
                idx = min(int((cur_offset + tau_hat) // dt), len(cur_P) - 1)
                seed_use = np.vstack([cur_P[idx:], np.repeat(
                    cur_P[-1:], idx, axis=0)]) if idx > 0 else cur_P.copy()
            else:
                seed_use = seed_P
        else:
            p_anchor, v_anchor, t_anchor = p, v, t_now
            seed_use = seed_P

        # --- Tầng dự báo: bấm giờ riêng cho t_predict (§7.1) ---
        t_pred0 = time.perf_counter()
        obs = get_obs(t_anchor)
        t_predict = time.perf_counter() - t_pred0

        # --- Dự phòng hạt giống DV-H10: nếu chốt đã bật ở một chu
        #     kỳ TRƯỚC, hạt giống của lần gọi này là hạt giống A*
        #     tại trạng thái neo (ảnh chụp vật cản bước 0 của dự
        #     báo hiện hành), thay cho hạt giống thẳng / khởi
        #     động ấm vừa suy ở trên. astar_seed trả None -> giữ
        #     seed_use cũ của chu kỳ này (hợp đồng của module:
        #     không tìm được đường thì phía gọi giữ hành vi cũ);
        #     chốt vẫn giữ nguyên.
        if seed_fallback and fb_triggered:
            fb_seed = astar_seed(
                p_anchor, goal, _astar_snapshot_obstacles(obs),
                prm.K + 1, d_safe=prm.d_safe, z_floor=prm.z_floor)
            if fb_seed is not None and fb_seed.shape == (prm.K + 1, 3):
                seed_use = fb_seed

        # --- Bộ giải: t_plan là solve_time nội bộ của plan() (§1.7);
        #     đếm cảnh báo solver phát ra trong lúc giải (§1.9) ---
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = plan(p_anchor, v_anchor, goal, obs, prm, seed_P=seed_use)
        n_solver_warnings = len(caught)
        t_plan = float(res.solve_time)
        # --- Hạt giống DV-H8 (bước nối M2): nếu plan() là bản đã
        #     bọc bởi method_adapter (A1–A3), adapter phơi
        #     last_seed_info của CHÍNH lần gọi này trên hàm đã bọc;
        #     t_seed đi vào T_latency theo §4.2 mục 6/9b (khớp đồng
        #     hồ H1 trong adapter, vốn đã cộng t_seed vào T_r).
        #     Đường không qua adapter (A0, baseline) không có thuộc
        #     tính này -> t_seed = 0 và bản ghi chu kỳ KHÔNG thêm
        #     khoá hạt giống (giữ trơ tuyệt đối cho A0).
        seed_info = getattr(plan, "last_seed_info", None)
        t_seed = float(seed_info["t_seed"]) if seed_info else 0.0
        latency = t_predict + t_seed + t_plan   # T_latency (§0/§7.1)
        delay = latency if time_model == "realtime" else 0.0
        planner_failed = bool(np.isinf(res.total_slack))

        # --- Kích hoạt DV-H10 v1.1 (phán bổ sung 2026-11-23,
        #     mục BS-1): chỉ đánh giá trong 3 lần gọi đầu (chu
        #     kỳ 0, 1, 2) — (a) lần gọi chu kỳ 0 chạm trần
        #     SEED_FALLBACK_ITERATION_CAP (105 vòng); hoặc (b)
        #     lần gọi chu kỳ 0 dùng > SEED_FALLBACK_MAX_ITERATIONS
        #     (20) vòng VÀ ít nhất một trong hai lần gọi chu kỳ
        #     1/2 cũng > 20 vòng. Điều kiện thành đúng ở chu kỳ
        #     nào thì bật chốt từ chu kỳ KẾ TIẾP (hạt giống của
        #     chính chu kỳ này không đổi) và fb_cycle ghi chu kỳ
        #     đó. Qua chu kỳ 2 mà chưa kích hoạt thì không đánh
        #     giá nữa — cơ chế tắt cho phần còn lại của episode.
        if seed_fallback and not fb_triggered:
            _fb_idx = int(len(cycles))
            _fb_iters = int(res.iterations)
            if _fb_idx == 0:
                fb_iter_cycle0 = _fb_iters
                if _fb_iters >= SEED_FALLBACK_ITERATION_CAP:
                    fb_triggered = True
                    fb_cycle = 0
            elif _fb_idx in (1, 2):
                if (fb_iter_cycle0 is not None
                        and fb_iter_cycle0 > SEED_FALLBACK_MAX_ITERATIONS
                        and _fb_iters > SEED_FALLBACK_MAX_ITERATIONS):
                    fb_triggered = True
                    fb_cycle = _fb_idx

        cyc = dict(cycle=len(cycles), t_start=t_cycle_start,
                   t_predict=float(t_predict), t_plan=t_plan,
                   solve_time=t_plan, T_latency=float(latency),
                   tau_hat=float(tau_hat),
                   v_anchor_max_abs_comp=float(np.max(np.abs(v_anchor))),
                   iterations=int(res.iterations),
                   total_slack=float(res.total_slack),
                   delay_applied=float(delay), delay_beyond_plan=0.0,
                   planner_failed=planner_failed,
                   n_solver_warnings=int(n_solver_warnings))
        if seed_info is not None:
            # Sáu trường hạt giống DV-H8 của chu kỳ (§4.2 mục 9b) —
            # chỉ khi qua adapter; khớp từng chu kỳ với
            # config.method_adapter.run_summary.seed_records.
            # seed_max_viol = None ⟺ chu kỳ không có vật cản nào
            # (adapter trả None: max trên tập rỗng không định
            # nghĩa). Trường log ở đây là float nên ánh xạ sang
            # 0.0 — theo quy ước dấu của trường (dương = độ sâu vi
            # phạm need; mọi tiêu chí kiểm toán đều ở dạng <= ~0,
            # ví dụ <= 1e-9 của Ô 3 DV-H8), 0.0 là giá trị "không
            # vi phạm": không giả mạo độ hở như một số âm (không
            # có vật nào để mà hở), không phá thống kê như -inf.
            # Sửa lỗi 2026-11-24: A2 crash float(None) trên kịch
            # bản không vật cản. Bản ghi thô seed_records của
            # adapter giữ nguyên null.
            _smv = seed_info["seed_max_viol"]
            cyc.update(
                seed_source=str(seed_info["seed_source"]),
                seed_trigger=str(seed_info["seed_trigger"]),
                t_seed=t_seed,
                seed_max_viol=0.0 if _smv is None else float(_smv),
                seed_budget_exceeded=bool(
                    seed_info["seed_budget_exceeded"]),
                seed_nodes=int(seed_info["seed_nodes"]))

        # --- Cửa sổ trễ DV-H7 (bước nối vào đường sản xuất): nếu
        #     plan() là bản đã bọc bởi method_adapter cho A1/A3,
        #     adapter phơi last_window_info của CHÍNH lần gọi này
        #     (window_info() của ReplanClock trong margin_h1: T_r
        #     không chặn, T_r_eff sau trần, capped, T_cap). Bốn
        #     trường được gắn vào bản ghi chu kỳ để cờ drift_capped
        #     TỒN TẠI trong log đường sản xuất (điều kiện cứng của
        #     Phong cho Ô 3 của DV-H7) — kể cả khi không đặt trần
        #     (T_cap = None -> drift_capped = False, T_r_eff = T_r).
        #     Đường không qua adapter H1 (A0, baseline, A2) không
        #     có thuộc tính này -> bản ghi chu kỳ KHÔNG thêm khoá
        #     (giữ trơ tuyệt đối cho A0).
        window_info = getattr(plan, "last_window_info", None)
        if window_info is not None:
            cyc.update(
                T_r=float(window_info["T_r"]),
                T_r_eff=float(window_info["T_r_eff"]),
                drift_capped=bool(window_info["capped"]),
                T_cap=(None if window_info["T_cap"] is None
                       else float(window_info["T_cap"])))

        # --- Ghi trạng thái neo MỌI lần gọi plan() (log phụ, điều
        # kiện tiên quyết của Ô 1 DV-H4 — xem docstring): phân loại
        # hộp theo dung sai tiền định 3e-3 của Quyết định 1 §1.5.
        if anchor_sink is not None:
            _v0_max = float(np.max(np.abs(v_anchor)))
            _box_tol = 3e-3
            if _v0_max <= prm.v_max:
                _box_class = "in_box"
            elif _v0_max <= prm.v_max + _box_tol:
                _box_class = "gray_zone"
            else:
                _box_class = "out_of_box"
            anchor_sink.append(dict(
                cycle=int(len(cycles)), t_start=float(t_cycle_start),
                time_model=time_model,
                p_anchor=np.asarray(p_anchor, dtype=float).tolist(),
                v_anchor=np.asarray(v_anchor, dtype=float).tolist(),
                v_anchor_max_abs_comp=_v0_max,
                v_max=float(prm.v_max), box_tol=_box_tol,
                box_class=_box_class,
                iterations=int(res.iterations),
                total_slack=float(res.total_slack),
                success=bool(res.success),
                planner_failed=planner_failed))

        # --- Chẩn đoán chu kỳ sập (log phụ, schema log chính giữ
        # nguyên): lưu TOÀN BỘ đầu vào của lời gọi plan() vừa thất
        # bại để tái lập ngoại tuyến từng chu kỳ.
        if planner_failed and diag_sink is not None:
            def _obs_rec(ob):
                if isinstance(ob, dict):
                    rec = dict(centers=np.asarray(
                        ob["centers"], dtype=float).tolist())
                    if "axes" in ob:
                        # Dict ellipsoid (chế độ quan sát ellipsoid):
                        # ghi bán trục + yaw thay cho bán kính; các
                        # khoá chẩn đoán của tầng hình dạng không
                        # ghi (chỉ hình học quyết định bài toán con).
                        rec["axes"] = np.asarray(
                            ob["axes"], dtype=float).tolist()
                        rec["yaws"] = np.asarray(
                            ob.get("yaws", 0.0), dtype=float).tolist()
                    else:
                        rec["radii"] = np.asarray(
                            ob["radii"], dtype=float).tolist()
                    rec["rho"] = np.asarray(
                        ob["rho"], dtype=float).tolist()
                    return rec
                o_, r_ = ob
                return dict(center=np.asarray(o_, dtype=float).tolist(),
                            radius=float(r_))
            diag_sink.append(dict(
                cycle=int(len(cycles)), t_start=float(t_cycle_start),
                time_model=time_model,
                p_current=np.asarray(p, dtype=float).tolist(),
                v_current=np.asarray(v, dtype=float).tolist(),
                p_anchor=np.asarray(p_anchor, dtype=float).tolist(),
                v_anchor=np.asarray(v_anchor, dtype=float).tolist(),
                v_anchor_max_abs_comp=float(np.max(np.abs(v_anchor))),
                goal=np.asarray(goal, dtype=float).tolist(),
                tau_hat=float(tau_hat), cur_offset=float(cur_offset),
                seed_P=(None if seed_use is None else
                        np.asarray(seed_use, dtype=float).tolist()),
                obs=[_obs_rec(ob) for ob in obs],
                result=dict(iterations=int(res.iterations),
                            total_slack=float(res.total_slack),
                            success=bool(res.success),
                            solve_time=float(res.solve_time),
                            min_clearance=float(res.min_clearance)),
                warning_messages=[str(w.message) for w in caught]))

        # --- Pha trễ: bay dọc kế hoạch hiện hành trong lúc chờ ---
        beyond = fly_along_plan(delay) if delay > 0.0 else 0.0
        cyc["delay_beyond_plan"] = float(beyond)
        if collided:
            status_note = "collision_during_delay"
            cycles.append(cyc)
            break

        # --- Áp kế hoạch mới (hoặc dự phòng khi thất bại) ---
        p_before = p.copy()
        if planner_failed and time_model == "realtime":
            # Không áp kế hoạch hỏng: bay tiếp dọc kế hoạch hiện hành
            # thêm dt (hãm bão hòa §7.1 nếu kế hoạch đã cạn).
            fly_along_plan(dt)
            if collided:
                status_note = "collision_after_planner_failure"
                cycles.append(cyc)
                break
        elif time_model == "realtime":
            # Bước áp kế hoạch mới: lệnh có phản hồi vận tốc tại
            # nút 0 — tiền hướng U[0] cộng hiệu chỉnh Kv·(V[0] − v)
            # theo vận tốc THẬT lúc áp (V[0] của kế hoạch chính là
            # vận tốc neo). Xem mục THỰC THI BÁM TRẠNG THÁI.
            u = track_command(res.U[0], res.V[0], v, u_max)
            fly_one_step(u, dt)
            cur_U = np.asarray(res.U, dtype=float)
            cur_P = np.asarray(res.P, dtype=float)
            cur_V = np.asarray(res.V, dtype=float)
            cur_offset = dt
            if not planner_failed:
                seed_P = np.vstack([res.P[1:], res.P[-1:]])
        else:
            u = np.asarray(res.U[0], dtype=float)
            fly_one_step(u, dt)
            if not planner_failed:
                seed_P = np.vstack([res.P[1:], res.P[-1:]])
        if time_model == "frozen":
            dur_step = segments[-1][1] if segments else dt
            for frac in (0.5, 1.0):
                pm = p_before + frac * (p - p_before)
                tm = t_now - dur_step + frac * dur_step
                for fn in truth_fns:
                    o, r = fn(tm)
                    min_clear_legacy = min(min_clear_legacy,
                                           float(np.linalg.norm(pm - o) - r))
        cycles.append(cyc)
        prev_latency = latency
        if collided:
            status_note = "collision"
            break
        if np.linalg.norm(p - goal) <= goal_tol and np.linalg.norm(v) <= 0.6:
            reached = True
            status_note = "reached"
            break

    # --- Kết cục và mã trạng thái §6.4 ---
    no_flight = len(segments) == 0
    if np.isfinite(state["min_clear"]):
        min_clear = float(state["min_clear"])
    elif no_flight:
        # §6.4.2: thất bại trước bước bay đầu tiên — khe hở tại điểm đầu
        min_clear = float(min(np.linalg.norm(p - fn(0.0)[0]) - fn(0.0)[1]
                              for fn in truth_fns))
    else:
        min_clear = None
    if reached or collided:
        status = "ok"            # kết cục bay thật (tới đích / va chạm)
    elif cycles and cycles[-1]["planner_failed"]:
        status = "solver_fail"   # chu kỳ cuối thất bại, nhiệm vụ dở dang
    else:
        status = "timeout"       # hết t_max mô phỏng hoặc trần wall-clock

    solve_times = [c["t_plan"] for c in cycles]
    metrics = dict(
        status=status,
        status_note=status_note,
        no_flight=bool(no_flight),
        success=bool(reached and not collided),
        safe_success=bool(reached and min_clear is not None
                          and min_clear >= safety_d),
        collision=bool(collided),
        reached_goal=bool(reached),
        min_clearance_continuous=min_clear,
        min_clearance_legacy=(float(min_clear_legacy)
                              if np.isfinite(min_clear_legacy) else None),
        path_length_m=float(state["path_len"]),
        flight_time_s=float(t_now),
        jerk_rms=jerk_rms_on_grid(segments, dt, float(t_now)),
        energy_proxy=float(state["energy"]),
        n_cycles=len(cycles),
        n_planner_failures=int(sum(1 for c in cycles if c["planner_failed"])),
        n_solver_warnings=int(sum(c["n_solver_warnings"] for c in cycles)),
        solve_time_mean=float(np.mean(solve_times)) if solve_times else 0.0,
        solve_time_p50=float(np.percentile(solve_times, 50)) if solve_times else 0.0,
        solve_time_p95=float(np.percentile(solve_times, 95)) if solve_times else 0.0,
        solve_time_max=float(np.max(solve_times)) if solve_times else 0.0,
        t_predict_mean=(float(np.mean([c["t_predict"] for c in cycles]))
                        if cycles else 0.0),
        latency_mean=(float(np.mean([c["T_latency"] for c in cycles]))
                      if cycles else 0.0),
        latency_max=(float(np.max([c["T_latency"] for c in cycles]))
                     if cycles else 0.0),
        iterations_mean=float(np.mean([c["iterations"] for c in cycles])) if cycles else 0.0,
        slack_mean=(float(np.mean([c["total_slack"] for c in cycles
                                   if np.isfinite(c["total_slack"])]))
                    if any(np.isfinite(c["total_slack"]) for c in cycles) else None),
        slack_max=(float(np.max([c["total_slack"] for c in cycles
                                 if np.isfinite(c["total_slack"])]))
                   if any(np.isfinite(c["total_slack"]) for c in cycles) else None),
        final_total_slack=(float(cycles[-1]["total_slack"]) if cycles else None),
        final_dist_to_goal=float(np.linalg.norm(p - goal)),
        mean_delay_applied=(float(np.mean([c["delay_applied"] for c in cycles]))
                            if cycles else 0.0),
        mean_delay_beyond_plan=(float(np.mean([c["delay_beyond_plan"] for c in cycles]))
                                if cycles else 0.0),
    )
    if seed_fallback:
        # Hai trường log bắt buộc của DV-H10 — chỉ tồn tại khi cơ
        # chế được bật; đường chạy không bật giữ schema cũ tuyệt
        # đối (điều kiện trơ của kiểm chứng DV-H10 phần 1).
        metrics["seed_fallback_triggered"] = bool(fb_triggered)
        metrics["seed_fallback_cycle"] = (None if fb_cycle is None
                                          else int(fb_cycle))
    episode_config = dict(
                            obs_mode=obs_mode,
                            ellipsoid_obs_source=ellipsoid_obs_source,
                            scenario_family=(sc_entry.get("family")
                                             if sc_entry else "D"),
                            scenario_platform=(
                                dict(name=platform[0],
                                     v_max=float(platform[1]),
                                     u_max=float(platform[2]))
                                if platform is not None else None),
                            planner_effective=dataclasses.asdict(prm),
                            substep_s=SUBSTEP, t_max_s=t_max,
                            t_max_source=t_max_source,
                            wall_clock_cap_s=wall_clock_cap_s,
                            goal_tol=goal_tol, safety_d_m=safety_d,
                            brake_v_eps=BRAKE_V_EPS,
                            execution_law=("velocity_state_feedback"
                                           if time_model == "realtime"
                                           else "frozen_time"),
                            track_kv=TRACK_KV,
                            delay_model=("compensated_stale_plan"
                                         if time_model == "realtime"
                                         else "frozen_time"))
    if seed_fallback:
        episode_config["seed_fallback_dvh10"] = True
        episode_config["seed_fallback_max_iterations"] = (
            SEED_FALLBACK_MAX_ITERATIONS)
    return dict(metrics=metrics, cycles=cycles,
                diagnostics=(list(diag_sink) if diag_sink is not None
                             else []),
                anchor_calls=(list(anchor_sink)
                              if anchor_sink is not None else []),
                config=episode_config)
