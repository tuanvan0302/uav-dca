# AGENTS.md — Quy tắc làm việc với mã nguồn SOTA

File này dành cho mọi agent (và người) sửa code trong gói SOTA. Mỗi quy tắc
được viết thành một đoạn ngắn; khi hai quy tắc xung đột, quy tắc cụ thể hơn
thắng quy tắc chung.

---

## 1. Ngôn ngữ: cái gì tiếng Việt, cái gì tiếng Anh

**Quy tắc gốc.** Comment, docstring và tài liệu `.md` viết bằng **tiếng Việt
có dấu**, chi tiết. Mọi thứ còn lại — tên file, tên module, tên hàm, tên lớp,
tên biến, tên hằng, khóa dict, tên cột log, thông điệp commit — viết bằng
**tiếng Anh**, chỉ dùng ký tự ASCII.

**Vì sao tách như vậy.** Định danh tiếng Anh giữ code đọc được với công cụ,
thư viện và người ngoài đội; tiếng Việt trong comment giữ được lập luận toán
học và quyết định thiết kế ở mức chi tiết mà cả đội đọc nhanh nhất.

**Không viết định danh tiếng Việt không dấu.** `e_bam`, `ha_tang`, `do_tre`
khó đọc và dễ nhầm nghĩa. Dùng tên tiếng Anh tương ứng: `tracking_error`,
`infra`, `latency`. Nếu thuật ngữ gốc là tiếng Việt, ghi nó trong docstring:
`tracking_error: sai số bám (e_bám) của bộ điều khiển, đơn vị m.`

**Thông điệp lỗi và log.** Thông điệp `raise`/`warnings`/`print` dùng tiếng
Anh, ngắn, có giá trị cụ thể (`f"K must be >= 1, got {K}"`). Phần giải thích
dài vì sao lỗi xảy ra để ở comment hoặc docstring tiếng Việt.

**Ngoại lệ trong test.** Mô tả ca kiểm thử trong `tests/` và
`baselines/test_baselines.py` — tên ca trong `_check(...)`, thông điệp
`assert`, dòng in chẩn đoán — được viết tiếng Việt vì chúng là tài liệu,
dẫn chiếu chương báo cáo/SPEC. Nhãn trạng thái và dòng tổng kết thì luôn
tiếng Anh (`[PASS]`, `[FAIL]`, `15/15 tests passed.`) để grep được.

**Tên do gói `sim/` định nghĩa.** Tên kịch bản (`S1_mot_vat`,
`D1_cat_ngang`…) do gói `sim/` định nghĩa; các gói khác trong `code/` chỉ
dùng lại các chuỗi đó. Muốn đổi thì đổi ở `sim/` trước.

**Thuật ngữ chuyên ngành.** Trong comment tiếng Việt được giữ nguyên thuật
ngữ tiếng Anh đã quen (QP, solver, slack, warm start, gauge, conformal
prediction). Không dịch gượng; nhưng dùng nhất quán một cách gọi trong toàn
gói.

---

## 2. Đặt tên

**File và module.** `snake_case`, tiếng Anh, danh từ mô tả nội dung:
`dca_planner.py`, `shape_forecast.py`, `margin.py`. Không đưa mã ticket, mã
lệch hay ngày vào tên file (`test_astar_seed_dvh8.py` là dạng cần tránh);
mã ticket để trong docstring của module.

**File test.** Test đơn vị: `tests/test_<module>.py`, trùng tên module được
kiểm thử (`test_safety_margin.py` ↔ `core/safety_margin.py`). Test tích hợp
nhiều module: `tests/test_<tính_năng>.py` (`test_dynamic.py`,
`test_ellipsoid_obs.py`). Hàm test đặt tên theo hành vi được kiểm:
`test_drift_matches_chapter6_example`, không dùng `test_1`, `test_case_a`.

**Hàm và phương thức.** `snake_case`, bắt đầu bằng động từ khi hàm có tác
dụng hoặc tính toán: `compute_margin`, `build_qp`, `load_conformal_table`.
Hàm trả về giá trị boolean bắt đầu bằng `is_`, `has_`, `can_`.

**Lớp.** `PascalCase`, danh từ: `PlannerParams`, `ConformalTable`,
`ReplanClock`. Lớp cấu hình kết thúc bằng `Config` hoặc `Params`; lớp kết quả
kết thúc bằng `Result`.

**Hằng số.** `UPPER_SNAKE_CASE` ở cấp module: `OSQP_OPTS`, `BRAKE_V_EPS`.
Không định nghĩa hằng trong thân hàm nếu nó được dùng ở nơi khác.

**Riêng tư.** Tiền tố `_` cho hàm/biến chỉ dùng trong module. Module khác
không được import tên có `_`; nếu cần dùng chung, bỏ `_` và đưa vào API công
khai (như `normalize_obstacles`, `kalman_cv_forecast`, `is_segment_clean`).

**Đổi tên.** `code/` là nguồn duy nhất; không giữ alias cho tên cũ. Đổi
tên hàm, hằng, file hay khóa log thì sửa mọi nơi gọi, test, `Dockerfile`
và tài liệu trong cùng một thay đổi.

**Ký hiệu toán học.** Được phép dùng tên ngắn trùng ký hiệu trong báo cáo:
`K`, `dt`, `P`, `V`, `U`, `O`, `R`, `RHO`, `tau`. Đây là ngoại lệ có chủ đích
với PEP 8. Điều kiện: docstring phải ghi nghĩa, shape và đơn vị của từng ký
hiệu.

**Đơn vị trong tên.** Khi đơn vị không hiển nhiên, gắn hậu tố: `solve_time_s`,
`latency_ms`, `angle_rad`. Không trộn đơn vị trong cùng một hàm.

---

## 3. Comment và docstring (tiếng Việt, chi tiết)

**Docstring module.** Mỗi file `.py` mở đầu bằng docstring tiếng Việt nêu:
module giải quyết bài toán gì, công thức chính, module liên quan, và cách
chạy (nếu là script/test). `core/dca_planner.py` và `core/safety_margin.py`
là mẫu tham khảo.

**Docstring hàm công khai.** Bắt buộc với mọi hàm/lớp không có tiền tố `_`.
Dùng kiểu Google: tiêu đề mục giữ tiếng Anh (`Args:`, `Returns:`, `Raises:`)
để công cụ đọc được, nội dung viết tiếng Việt.

```python
def delta_drift(v_hat: float, a_max: float, T_r: float,
                tracking_error: float = TRACKING_ERROR_DEFAULT) -> float:
    """Hạng tử trôi δ_trôi = v̂ · T_r + ½ · a_max · T_r² + e_bám  (m).

    Hàm không tự áp trần; khi dùng trần, phía gọi truyền
    ReplanClock.effective_window() làm T_r.

    Args:
        v_hat: tốc độ ước lượng của vật cản (m/s).
        a_max: cận trên gia tốc của vật cản (m/s²).
        T_r: cửa sổ trễ (s).
        tracking_error: sai số bám e_bám của bộ điều khiển (m).

    Returns:
        Độ dịch tối đa của vật cản trong cửa sổ trễ (m).

    Raises:
        ValueError: khi v_hat, a_max hoặc T_r âm.
    """
```

Docstring cũ viết dạng văn xuôi (không có `Args:`) vẫn hợp lệ nếu đủ nghĩa,
shape và đơn vị; chuyển sang kiểu Google khi sửa hàm đó.

**Mảng phải có shape và đơn vị.** Mọi tham số/giá trị trả về kiểu
`np.ndarray` ghi shape và đơn vị: `P: (K+1, 3) vị trí, m`. Đây là nguồn lỗi
phổ biến nhất trong code số học.

**Comment giải thích "vì sao", không lặp "cái gì".** Không viết
`# tăng tau`; viết `# Còn vi phạm sau max_iter vòng → tăng phạt để ép
khả thi (phạt chính xác)`. Con số ma thuật luôn đi kèm nguồn gốc, như
`extra_margin = 0.03  # ≈ u_max·dt²/8, độ võng cung parabol giữa hai nút`.

**Dẫn nguồn.** Khi code hiện thực một mục của SPEC, báo cáo hoặc bài báo,
ghi tham chiếu cụ thể (`SPEC_cong_bang §7`, `Chương 6`, `DV-B2`). Khi code cố
ý lệch SPEC, ghi mã lệch và lý do ngay tại chỗ lệch.

**Trung thực về trạng thái.** Không viết "đã kiểm chứng" nếu chưa chạy thật.
Ghi rõ: chưa build, chưa có số đo, giá trị tạm. Comment sai còn nguy hiểm hơn
không có comment.

**Giữ comment đồng bộ với code.** Sửa logic thì sửa comment trong cùng thay
đổi. Comment chết (mô tả hành vi cũ, tham số không còn dùng) phải xóa.

**Comment mô tả hiện tại, không kể lịch sử.** Không ghi "bản cũ", "trước
đây", tên người rà soát, ngày sửa hay số dòng của file khác — những thứ đó
thuộc về git log. Nếu một test tồn tại để chặn một lỗi, mô tả lỗi đó như
một ca cụ thể ("nếu yaws là mảng (3,) thì planner crash"), không kể nó được
phát hiện khi nào. Không trỏ tới file nằm ngoài `code/`; mã SPEC và mã lệch
(`DV-B2`) thì được.

**TODO.** Dạng `# TODO(<người/mã ticket>): <việc cần làm>` bằng tiếng Việt.
Không để TODO không có chủ.

---

## 4. Tiêu chuẩn code cộng đồng

**PEP 8 + PEP 257.** Là nền tảng. Ngoại lệ duy nhất được chấp nhận là ký hiệu
toán học ở mục 2.

**Formatter và linter.** Dùng `ruff`, độ dài dòng 88; cấu hình nằm trong
`pyproject.toml` (`[tool.ruff]`). Từ thư mục `SOTA/`, cả hai lệnh sau phải
sạch trước khi kết thúc thay đổi:

```bash
ruff format code
ruff check code
```

**Type hints.** Bắt buộc cho chữ ký hàm công khai. Mỗi file bắt đầu bằng
`from __future__ import annotations`. Dùng `np.ndarray` (hoặc
`numpy.typing.NDArray[np.float64]`) cho mảng; shape ghi trong docstring.

**Import.** Thứ tự: thư viện chuẩn → bên thứ ba → nội bộ gói, mỗi nhóm cách
một dòng trống. Import tuyệt đối từ gốc `code/` (`from core.ellipsoid import
...`). Không dùng `import *`.

**Encoding.** File lưu UTF-8. Dòng `# -*- coding: utf-8 -*-` không cần với
Python 3 nhưng được giữ ở file cũ; file mới có thể bỏ.

**Không dùng giá trị mặc định ngầm của solver.** Dung sai, số vòng lặp và tên
solver luôn được truyền tường minh từ hằng số chung (`OSQP_OPTS` định nghĩa
một lần ở `core/dca_planner.py`, `baselines/common.py` import lại) và được
ghi vào log/metadata.

**Tính tái lập.** Mọi nguồn ngẫu nhiên nhận `seed` hoặc
`np.random.Generator` từ tham số; không gọi `np.random.seed()` toàn cục.
Phiên bản thư viện ghim trong `requirements.txt`.

---

## 5. Dễ bảo trì

**Một hàm, một việc.** Hàm dài hơn ~50 dòng hoặc có hơn ba mức lồng nhau là
dấu hiệu cần tách. Tách theo bước của thuật toán (dựng QP, giải, cập nhật
điểm tuyến tính hóa), không tách ngẫu nhiên.

**Cấu hình qua dataclass.** Tham số thuật toán gom trong `@dataclass`
(`PlannerParams`, `MarginConfig`), mỗi trường có comment tiếng Việt ghi ý
nghĩa và đơn vị. Không thêm tham số rời rạc vào chữ ký hàm khi đã có dataclass.

**Giữ ổn định giao diện.** Chữ ký `plan(p0, v0, goal, obstacles, prm,
seed_P=None)` là hợp đồng chung giữa planner, các baseline (có thêm
`cfg=None`) và `sim_realtime`. Đổi nó phải cập nhật mọi nơi gọi và test
trong cùng một thay đổi.

**Không trùng lặp logic dùng chung.** Ngưỡng `need`, hàm mục tiêu `J`, động
học tích phân kép và chuẩn hóa vật cản chỉ có một bản hiện thực. Baseline mới
gọi lại bản đó thay vì viết lại.

**Đổi tên file/hàm cũ.** Code mới tuân theo quy tắc này ngay. Code cũ chỉ đổi
tên trong một thay đổi riêng, có cập nhật toàn bộ nơi gọi, `Dockerfile` và
tài liệu; không trộn việc đổi tên với sửa logic.

**Không xóa im lặng.** Trước khi xóa hàm, tham số hay file, tìm mọi nơi tham
chiếu (`grep`) và ghi lý do trong mô tả thay đổi.

---

## 6. Kiểm thử

**Cách chạy.** Từ thư mục `code/`:

```bash
python3 -m tests.test_safety_margin    # một test
python3 -m baselines.test_baselines    # bộ test baseline
```

Danh sách đầy đủ nằm trong `CMD` của `Dockerfile`. Phần lớn test cần các gói
`sim/`, `experiments/` và dữ liệu `core/conformal_tables/`, đều nằm trong
`code/`.

**Mỗi thay đổi logic đi kèm test.** Test ưu tiên dùng chính ví dụ số trong
báo cáo/SPEC để kết quả có thể đối chiếu bằng tay. So sánh số thực dùng dung
sai tường minh (`abs(a - b) < 1e-12` hoặc `np.testing.assert_allclose`).

**Test chạy ngoại tuyến.** Không cần mạng, GPU hay file ngoài gói. Dữ liệu
test nhỏ, sinh tại chỗ với seed cố định.

**Giữ điểm vào `__main__`.** Mỗi file test có khối `if __name__ == "__main__":`
để chạy được bằng `python3 -m tests.<name>` mà không cần pytest.

**Báo cáo kết quả trung thực.** Agent chỉ được nói "test đạt" sau khi đã chạy
và thấy kết quả. Nếu không chạy được (thiếu solver, thiếu môi trường), nói rõ
là chưa chạy và vì sao.

---

## 7. Checklist trước khi kết thúc thay đổi

- Tên file, hàm, biến mới là tiếng Anh, ASCII, đúng kiểu chữ ở mục 2.
- Comment và docstring mới là tiếng Việt có dấu, ghi shape, đơn vị, nguồn.
- Hàm công khai có type hints và docstring kiểu Google.
- `ruff check` và `ruff format --check` sạch trên các file đã sửa.
- Test liên quan đã chạy và đạt (hoặc ghi rõ lý do chưa chạy).
- Comment, README và `Dockerfile` phản ánh đúng code sau thay đổi.
