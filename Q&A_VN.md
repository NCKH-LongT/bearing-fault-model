Q&A — Trả Lời Phản Biện (Tiếng Việt)
====================================

Tài liệu này tổng hợp trả lời ngắn gọn cho các câu hỏi thường gặp từ reviewer và liệt kê các thí nghiệm bổ sung dự kiến thực hiện để tăng tính thuyết phục.

1) External Validity: RUNS=1 và nguy cơ overfitting
- Câu hỏi: Chỉ có một run (RUNS=1, 129 file), làm sao đảm bảo ResNet‑18 + late‑fusion không bị “học vẹt” vào một vòng bi cụ thể? Vì sao không kiểm chứng chéo trên CWRU/Paderborn?
- Trả lời (phạm vi hiện tại): Chúng tôi không khẳng định tổng quát tuyệt đối khi RUNS=1. Thay vào đó, giảm lạc quan do thời gian bằng tách theo trục TTF (train sớm, test muộn), khống chế dung lượng mô hình (ResNet‑18 nhỏ, weight decay, label smoothing, early stopping, balanced sampling), và quyết định ở mức file bằng trung bình logits (pre‑softmax). Quy trình 2 pha (stratified→temporal) cho thấy mô hình chuyển được từ early‑life sang late‑life mà không rò rỉ thời gian.
- Vì sao chưa cross‑dataset: CWRU/Paderborn là fault‑seeded, ngắn, khác cảm biến/nhãn/lấy mẫu (không có Degrading theo TTF); chuyển thẳng dễ trộn domain shift với sai khác nhiệm vụ. Chúng tôi sẽ báo cáo baseline vibration‑only, chống rò rỉ trên các bộ này với window/STFT hài hoà để sanity‑check, đồng thời giữ trọng tâm đánh giá theo TTF.
- Kế hoạch củng cố (hành động):
  - Holdout theo run/bearing khi có thêm run: leave‑one‑run‑out (LOTO) / bearing‑wise.
  - Cross‑validation theo khối thời gian trong cùng run: xoay các lát early/mid/late; báo cáo mean±std.
  - Holdout theo điều kiện vận hành: giữ nguyên tập tải/tốc độ chưa thấy cho test (nếu có metadata).
  - Sanity cross‑dataset (vibration‑only): CWRU, Paderborn với window/STFT nhất quán và chống rò rỉ.
  - Ablation loại trừ overfit: quét dung lượng (width 16–128), vib‑only vs +temp, kiểm độ bền trên các bin TTF.

2) Auto‑train rounds có làm RUNS>1?
- Câu hỏi: Nhiều round auto_r* có tương đương nhiều run dữ liệu không?
- Trả lời: Không. Mỗi round là một lần train→eval độc lập để dò siêu tham số trên CÙNG dữ liệu; RUNS vẫn bằng 1. Chúng tôi chọn checkpoint stratified tốt nhất và fine‑tune theo temporal.

3) Tính mới so với kho mã upstream (mdzalfirdausi)
- Câu hỏi: Mới hơn gì so với CNN upstream?
- Trả lời: Pipeline hướng hiện trường với: (i) đánh giá leakage‑aware căn theo TTF (kèm báo cáo theo lớp hiện diện ở lát cuối đời), (ii) fusion đa cảm biến với 6‑D nhiệt độ, (iii) quyết định file‑wise bằng trung bình logits (pre‑softmax), (iv) chuẩn hoá STFT phù hợp chuỗi run‑to‑failure (z‑score theo cửa sổ + theo tần số theo khung). Trọng tâm chuyển từ độ chính xác cửa sổ đơn sang đánh giá file‑level không rò rỉ thời gian.

4) Vì sao Degrading khó và cải thiện thế nào
- Câu hỏi: Vì sao Degrading khó phân loại và hướng cải thiện?
- Trả lời: Degrading chồng lấn Healthy gần ranh sớm/trung (TTF≈0.6), có shift điều kiện, và nhãn theo ngưỡng. Cải thiện: (1) đầu chuỗi TCN/LSTM/GRU/Transformer trên dãy cửa sổ ngắn, (2) xu hướng nhiệt đa tỉ lệ (slope ngắn/trung, EMA/STL), (3) curriculum FT quanh TTF∈[0.5,0.8], có thể dùng focal loss vừa phải, (4) nhãn mềm hoặc pseudo‑label nhất quán lân cận, (5) hiệu chỉnh/calibration và abstention gần ranh quyết định.

5) Vì sao nhấn mạnh đánh giá leakage‑aware theo thời gian
- Câu hỏi: Lợi ích của TTF‑split?
- Trả lời: Cửa sổ chồng lấn làm khung liền kề rất giống nhau; tách ngẫu nhiên theo cửa sổ dễ rò rỉ và thổi phồng kết quả. Chúng tôi ép TTF‑split (train sớm, test muộn), báo cáo theo lớp hiện diện ở lát cuối đời (không có Healthy), và cung cấp time‑metrics (Acc/Macro‑F1 theo bin TTF). Giới hạn số cửa sổ ở validation chỉ để tăng tốc; test dùng gộp file‑wise đầy đủ. Sẽ bổ sung thử nghiệm “temporal gap” giữa các lát để giảm tương quan sát ranh.

6) Vì sao trung bình logits (pre‑softmax) thay vì xác suất
- Câu hỏi: Lý do chọn mean‑logit?
- Trả lời: Trung bình logits ít bị lệch bởi dự đoán over‑confident ở mức cửa sổ và phù hợp với bề mặt quyết định tuyến tính trong không gian logit; thực nghiệm cho nhãn file ổn định hơn averaging probabilities hoặc majority vote.

7) Căn chỉnh cửa sổ nhiệt độ và rung
- Câu hỏi: Nhiệt có cắt đồng bộ với rung không?
- Trả lời: Có. Bốn kênh lấy mẫu đồng bộ; cắt cùng chỉ số [s:e] cho rung và nhiệt. Nếu triển khai thực có sampling khác, sẽ thêm bước resample/căn thời gian rõ ràng.

8) Báo cáo bổ sung để tăng tính thuyết phục
- Thêm ablation: vib‑only vs +temp; quét dung lượng; OneCycle vs Cosine; nhạy cảm kích thước ảnh; nhạy cảm khoảng trống thời gian.
- Thêm sanity cross‑dataset (vibration‑only) trên CWRU/Paderborn với tách chống rò rỉ và STFT hài hoà.
- Thêm biểu đồ time‑metrics (Acc/Macro‑F1 vs TTF) kèm khoảng tin cậy.
- Thêm thí nghiệm holdout theo run/bearing khi có thêm run.

9) Ablation đặc trưng nhiệt, rò rỉ sát ranh TTF, và “cuối đời” vs “toàn vòng đời”
- Câu hỏi: Hình 1 cho thấy Degrading có F1≈0.64, thấp hơn Fault=1.00. Có ablation chứng minh 6 đặc trưng nhiệt giúp hơn vib‑only không? 6 chỉ số đã đủ mô tả “xu hướng nhiệt” chưa? Ở mốc 60% TTF, cửa sổ sát ranh vẫn tương quan cao — nhóm xử lý thế nào; có cần “temporal gap”? Cuối cùng, đánh giá Degrading vs Fault ở cuối đời có phản ánh triển khai khi Healthy chiếm đa số? FAR dự kiến nếu còn nhầm Healthy↔Degrading?
- Trả lời:
  - Ablation: Sẽ bổ sung (i) vib‑only, (ii) +temp(6‑D), (iii) +multi‑scale nhiệt (nhiều chân trời slope/EMA). Kỳ vọng cải thiện chính ở Degrading (TTF 60–90%). Giữ nguyên cấu hình/seed; báo cáo file‑wise và theo bin TTF.
  - Độ đủ của 6‑D: 6‑D bắt xu hướng thô theo cửa sổ; để giàu động học hơn sẽ thêm slope đa thang, EMA/drift dài hạn, hoặc tách xu hướng STL; có thể gắn đầu chuỗi ngắn trên CNN 2D để tận dụng ngữ cảnh thời gian.
  - Rò rỉ sát ranh: Không có cửa sổ vượt ranh, nhưng cửa sổ sát ranh còn tương quan. Sẽ báo cáo độ nhạy với “temporal gap” (loại ±ΔTTF, ví dụ 0.5–2.0%) và đánh giá với cửa sổ không chồng lấn để giảm tương quan.
  - Cuối đời vs toàn vòng đời: Lát muộn phù hợp cảnh báo cận hỏng. Với vận hành toàn dải, sẽ (i) báo cáo Healthy vs Non‑Healthy ở sớm/trung, (ii) ước lượng FAR theo luật cảnh báo bền vững (K‑of‑N, ngưỡng đã hiệu chỉnh), và (iii) dùng calibration để hạn chế false alarm. Các số liệu sẽ bổ sung vào time‑metrics theo bin TTF.

---

Tài liệu đi kèm bản thảo chính (main.tex) để phục vụ phản biện và chuẩn bị camera‑ready.
