Tiêu đề: Three-Stage Bearing Health Classification (Rung STFT + Nhiệt, TTF‑aligned)
Tác giả: Long Truong, Phu Le Nguyen
Thời lượng: 12–15 phút

---

Slide 1 — Bối cảnh & Động lực (1’)
- Vấn đề: chẩn đoán sức khỏe ổ lăn theo thời gian thực, giai đoạn Degrading khó nhận biết; rủi ro rò rỉ thời gian khi chia cửa sổ ngẫu nhiên.
- Mục tiêu: quy trình đánh giá sát triển khai (leakage‑aware, TTF‑aligned) + mô hình gọn, đa phương thức rung–nhiệt.

Gợi ý nói: Nêu hai “nút thắt” thực tế: (i) rò rỉ thời gian, (ii) giai đoạn Degrading mơ hồ/thiếu mẫu ở giai đoạn muộn.

---

Slide 2 — Đóng góp chính (1’)
- Giao thức đánh giá leakage‑aware, TTF‑aligned, báo cáo present‑class.
- Pipeline gọn đa phương thức: STFT rung + đặc trưng nhiệt 6‑D; late fusion.
- Suy luận file‑wise bằng mean‑logit (ổn định, dễ triển khai).
- Ablation tập trung (Table 3) làm rõ vai trò nhiệt, khởi tạo từ stratified.

Gợi ý nói: Nhấn mạnh “đánh giá sát triển khai” là điểm khác biệt cốt lõi.

---

Slide 2.5 — Lý do chọn thiết kế/giao thức (1’)
- Vì sao TTF‑aligned leakage‑aware: phản ánh trình tự hỏng hóc, tránh rò rỉ cửa sổ chồng lấn, báo cáo present‑class khi lớp vắng.
- Vì sao đa phương thức rung–nhiệt: rung mạnh ở giai đoạn sớm, nhiệt rõ ở giai đoạn muộn → bổ sung lẫn nhau.
- Vì sao mean‑logit file‑wise: ổn định theo tệp, chống nhiễu cửa sổ, nhất quán với yêu cầu suy luận ngoài hiện trường.

Gợi ý nói: Nêu các thất bại khi dùng split ngẫu nhiên và vote thuần tuý cửa sổ.

---

Slide 3 — Dữ liệu & Cấu hình (1’)
- RUNS=1, 129 files; sampling 25.6 kHz; cửa sổ 1.0 s, overlap 50%.
- Ba giai đoạn theo TTF% (ngưỡng trong `Table~\ref{tab:stages}`): Healthy, Degrading, Fault.

Tài liệu: `main.tex` (Experimental Configuration), bảng `tab:stages`.

---

Slide 4 — Experimental Setup (TTF‑aligned) (1’)
- Chia theo TTF%: train [0,60], val [60,70], test [70,100.1]; tránh chồng lấn cửa sổ train/test.
- Hình minh họa protocol: `figures` trong `main.tex:fig:protocol_compare`.

Tài liệu: `configs/logs_stft_full_temporal_gating.yaml` (temporal_ttf).

---

Slide 5 — Feature Engineering (1’)
- Rung: STFT `n_fft=2048`, `hop_length=512` (Hann), resize 160×160, per‑frequency normalization, ghép 2×H×W.
- Nhiệt: 6‑D (mean, std, slope × 2 kênh).

Tài liệu: `main.tex` (Feature Engineering).

---

Slide 6 — Kiến trúc & Late Fusion (1’)
- CNN 2D cho rung, tuyến tính cho nhiệt; concat → linear head.
- Hình Block Diagram: `main.tex:fig:block`.

Gợi ý nói: Thiết kế tiêu chuẩn, ưu tiên gọn nhẹ, dễ tái lập.

---

Slide 7 — Quy trình 2 giai đoạn (1’)
- Stratified (0.6/0.2/0.2) để ổn định học và chọn mô hình.
- Fine‑tune temporal (TTF‑aligned) khởi tạo từ checkpoint stratified.

Tài liệu: `configs/logs_stft_train_strat.yaml`, `configs/logs_stft_full_temporal_gating.yaml:32` (init_from).

---

Slide 7.5 — Quy trình thực hiện chi tiết (1’30s)
- Chuẩn bị dữ liệu: kiểm tra manifest, đồng bộ kênh, cửa sổ 1.0s/50% overlap.
- Pha phát triển (stratified): dò cấu hình CNN gọn, xác lập chuẩn hoá theo tần số, chọn checkpoint tốt nhất.
- Pha tinh chỉnh (temporal): khởi tạo từ stratified, FT theo TTF, bật AMP, giới hạn `val_max_windows` để tăng tốc.
- Đánh giá: lát Early/Late/Full‑range, dùng present‑class khi lớp vắng, sinh hình confusion/F1.

Gợi ý nói: Nhấn mạnh kiểm soát rò rỉ và ghi log/tái lập cấu hình.

---

Slide 8 — Suy luận File‑wise (30s)
- Trung bình pre‑softmax logits theo cửa sổ trong mỗi file (mean‑logit) → argmax.

Tài liệu: `main.tex:sec:filewise`.

---

Slide 9 — Kết quả Stratified (30s)
- Macro‑F1 ≈ 0.7762; Fault dễ, Degrading khó hơn.

Tài liệu: `main.tex:Results (Stratified)`.

---

Slide 10 — Kết quả Temporal (1’)
- Early (70–90%): Accuracy 0.8846; Degrading F1 0.9388.
- Late (90–100.1%): Accuracy 0.9231; Fault F1 0.9600.
- Hình: `figures/temporal/confusion_matrix.png`, `figures/temporal/f1_present.png`.

---

Slide 11 — Full‑range 0–100.1% (30s)
- Vote: Accuracy 0.8915; Macro‑F1 0.8739. (Mean‑logit: Acc 0.7752; Macro‑F1 0.8006)
- Hình: `figures/fullrange/confusion_matrix_full.png`, `figures/fullrange/f1_present_full.png`.

---

Slide 12 — Ablation (Table 3) (1’30s)
- Vib‑only “mù” Fault ở lát muộn (F1 ≈ 0); multi‑modal giữ Fault F1 cao.
- Khởi tạo từ stratified > train from scratch.
- Nhạy cảm lát kiểm thử: [70,100.1]% giảm Deg F1 do ranh giới Healthy–Degrading.
- SVM 8D mạnh ở [70,90]% (Deg‑only), yếu ở lát rất muộn.

Tài liệu: `main.tex:tab:abl_summary`.

---

Slide 12.5 — Phát hiện chính từ ablation (1’)
- Multi‑modal > vib‑only ở lát muộn: Fault F1 tăng rõ, giữ được Degrading ở mức cao.
- Khởi tạo từ stratified giúp FT ổn định, ít lệ thuộc LR/epoch.
- Cấu hình high‑res 224×224 không vượt cấu hình 160×160 (chi phí cao hơn, lợi ích hạn chế).

Gợi ý nói: Liên hệ Table 3 khi giải thích từng ý.

---

Slide 13 — Bàn luận ngắn (45s)
- Vì sao nhiệt giúp giai đoạn muộn (tổng hợp tải + ma sát, xu hướng rõ dần).
- Vì sao ranh giới Healthy–Degrading khó (quá độ, chồng lấn phân bố, thiếu hỗ trợ lớp).

---

Slide 14 — Hạn chế (30s)
- RUNS=1; ngưỡng 60/90% thực nghiệm; giả định đồng bộ rung–nhiệt; cần present‑class khi lớp vắng mặt.
- Có thể còn tương quan lân cận gần biên tách TTF; cần chèn khoảng cách thời gian khi đánh giá nghiêm ngặt.
- Nhiệt độ có độ trễ/độ trơn cao: cần bối cảnh chuỗi dài hơn để cải thiện ranh giới Healthy–Degrading.

---

Slide 15 — Bài học triển khai (30s)
- Tránh rò rỉ: chia TTF liên tục, không chồng lấn; file‑wise aggregation.
- Cần resampling/alignment nếu tốc độ kênh khác nhau.

---

Slide 16 — Kế hoạch tương lai (30s)
- Mở rộng cross‑bearing/đa bộ dữ liệu để đánh giá tổng quát.
- Cải thiện ranh giới Healthy–Degrading: phân tích độ nhạy ngưỡng, mô hình chuỗi/ngữ cảnh dài.

---

Slide 17 — Kết luận (30s)
- Giao thức TTF‑aligned leakage‑aware + multi‑modal gọn cho triển khai.
- Kết quả mạnh ở lát muộn (Fault F1 0.9600) và full‑range (Macro‑F1 0.8739).

---

Slide 18 — Tái lập & Mã nguồn (30s)
- Cấu hình stratified: `configs/logs_stft_train_strat.yaml`.
- Cấu hình temporal: `configs/logs_stft_full_temporal_gating.yaml` → `runs/logs_stft_temporal_gating_final`.
- Hình/bảng nằm trong `figures/...` và `main.tex`.

---

Phụ lục A — Điểm mạnh & Điểm yếu (1’)
- Điểm mạnh:
  - Giao thức TTF‑aligned leakage‑aware, đánh giá sát triển khai; present‑class rõ ràng.
  - Pipeline gọn, dễ tái lập; mean‑logit đơn giản, ổn định cấp tệp.
  - Chứng cứ ablation thuyết phục về lợi ích đa phương thức và khởi tạo từ stratified.
- Điểm yếu:
  - RUNS=1 hạn chế tổng quát hóa; ngưỡng 60/90% mang tính thực nghiệm.
  - Chưa khai thác chuỗi dài/transformer; phụ thuộc resize 160×160.

---

Phụ lục B — Lý do/Quyết định thiết kế (1’)
- Chọn 160×160 để cân bằng chi phí–hiệu quả; high‑res không mang lại lợi ích ổn định.
- Chọn n_fft=2048, hop=512 (Hann): cân bằng phân giải thời‑tần cho cửa sổ 1s ở 25.6 kHz.
- Không dùng class‑weights khi FT temporal (tránh thiên lệch muộn), ưu tiên balanced sampling.
- Giữ AMP và `val_max_windows=50` để rút ngắn vòng lặp mà không ảnh hưởng xu hướng.

---

Phụ lục C — Kinh nghiệm & Khuyến nghị (1’)
- Luôn kiểm định không rò rỉ: tách TTF liên tục, tránh cửa sổ chồng lấn train/test.
- Báo cáo present‑class khi lớp vắng; tránh đánh giá sai lệch.
- Ghi cấu hình/đường dẫn checkpoint trong bài để tái lập.


Phụ lục — Gợi ý Q&A
- Q: Vì sao dùng mean‑logit thay vì vote? A: Ổn định hơn với phân phối logit, giảm nhiễu cửa sổ.
- Q: Tại sao 160×160 thay vì 224×224? A: Cân bằng chi phí/hiệu quả; ablation high‑res không vượt cấu hình cuối.
- Q: Có rò rỉ nào còn lại? A: Đã dùng TTF liên tục; nêu khả năng tương quan lân cận và cách giảm thiểu (khoảng cách thời gian, không chồng lấn đánh giá).
