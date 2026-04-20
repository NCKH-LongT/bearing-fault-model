# Kết quả & đối sánh (từ thư mục `figures/`)

Tài liệu này tổng hợp các “artifact” có sẵn trong thư mục `figures/` (report `.txt`, hình confusion matrix/F1, đường cong huấn luyện) và đối sánh hiệu quả theo các giai đoạn/ablation.

## 1) Stratified (development, chia theo file)

Nguồn: `figures/stratified/report.txt`, cùng các hình `figures/stratified/confusion_matrix.png`, `figures/stratified/f1_per_class.png`.

Kết quả (3 lớp, file-wise):
- Accuracy: **0.7241**
- Macro-F1: **0.7762**
- F1 theo lớp:
  - Healthy: **0.6923** (support 16)
  - Degrading: **0.6364** (support 8)
  - Fault: **1.0000** (support 5)

Nhận xét:
- Ở setting “dev” (chia theo file, không ép theo thời gian), mô hình học rất tốt lớp **Fault** (F1 = 1.0), và đạt mức khá cho Healthy/Degrading.
- Đây là baseline phù hợp để chọn cấu hình/siêu tham số trước khi chuyển sang đánh giá theo thời gian (deployment-oriented).

## 2) Temporal (deployment-oriented, late-life slice)

Nguồn: `figures/temporal/report.txt`, `figures/temporal/report_early_70_90.txt`, `figures/temporal/report_late_90_100.txt`, và các hình `figures/temporal/confusion_matrix.png`, `figures/temporal/f1_per_class.png`.

### 2.1 Báo cáo tổng (trong `figures/temporal/report.txt`)

Kết quả (3 lớp, file-wise; **Healthy không xuất hiện trong slice**):
- Accuracy: **0.3333**
- Macro-F1: **0.3324**
- F1 theo lớp:
  - Healthy: **0.0000** (support 0)
  - Degrading: **0.0741** (support 26)
  - Fault: **0.9231** (support 13)

Diễn giải:
- Mô hình **rất mạnh ở Fault** (F1 ~ 0.92) nhưng **rất yếu ở Degrading** trong phần “late-life” rộng.
- Vì Healthy vắng mặt, Macro-F1 3 lớp bị “kéo xuống”; khi báo cáo cho temporal slice, nên nhấn mạnh thêm metric theo **present classes** (chỉ Degrading/Fault) để công bằng.

### 2.2 Tách vùng thời gian: [70, 90] vs [90, 100]

Nguồn:
- `figures/temporal/report_early_70_90.txt`: (chỉ Degrading xuất hiện)
  - Accuracy: **0.0385**
  - Degrading F1: **0.0741** (recall 0.0385)
- `figures/temporal/report_late_90_100.txt`: (chỉ Fault xuất hiện)
  - Accuracy: **0.9231**
  - Fault F1: **0.9600**

Nhận xét:
- Ở vùng **rất-late** ([90,100]%), tín hiệu Fault rõ ràng nên mô hình dự đoán tốt.
- Ở vùng **Degrading sớm** ([70,90]%), tín hiệu mơ hồ hơn và dễ nhầm → đây là nút thắt chính nếu mục tiêu là “cảnh báo sớm”.

## 3) Temporal\_alltest (đánh giá phủ toàn bộ 129 files)

Nguồn: `figures/temporal_alltest/report.txt` và các hình trong `figures/temporal_alltest/`.

Kết quả (3 lớp, file-wise; support đầy đủ 129 files):
- Accuracy: **0.6899**
- Macro-F1: **0.5892**
- F1 theo lớp:
  - Healthy: **0.7958** (support 77)
  - Degrading: **0.0488** (support 39)
  - Fault: **0.9231** (support 13)

Nhận xét:
- Khi đánh giá trên toàn bộ timeline, mô hình tiếp tục cho thấy **mạnh ở Healthy/Fault** nhưng gần như “sụp” ở Degrading (F1 ~ 0.05).
- Điều này phù hợp với quan sát ở temporal: phân biệt **Degrading** (đặc biệt giai đoạn đầu) là bài toán khó nhất.

## 4) Ablation (so sánh các biến thể trong `figures/ablation/`)

Thư mục `figures/ablation/` chủ yếu chứa hình (confusion matrix + F1 plots) cho các thí nghiệm ablation. Vì hình `.png` không kèm file `.txt` chứa số liệu, các con số dưới đây được lấy từ phần thuyết minh trong repo (`paper_extract.txt`, `summary.md`) và đối chiếu với các hình tương ứng trong `figures/ablation/`.

### 4.1 Multi-modal (vibration + temperature) vs vibration-only

Các hình liên quan:
- Very-late slice: `figures/ablation/multi_modal_85100_*.png` vs `figures/ablation/vib_only_85100_*.png`
- Broad late slice: `figures/ablation/multi_modal_broad_*.png` vs `figures/ablation/vib_only_broad_*.png`

Kết luận đối sánh (macro-F1 theo *present classes*, Healthy vắng mặt):
- Very-late slice **[85, 100.1]% TTF**:
  - vibration-only: **Macro-F1 ≈ 0.2593**
  - vibration+temperature: **Macro-F1 ≈ 0.9467**
- Broad late slice **[70, 100.1]% TTF**:
  - vibration-only: **Macro-F1 ≈ 0.4000**
  - vibration+temperature: **Macro-F1 ≈ 0.3324**

Diễn giải nhanh:
- Nhiệt độ giúp rất mạnh ở vùng **rất-late** (Fault rõ ràng) → cải thiện phân biệt Degrading/Fault.
- Nhưng khi “mở rộng” slice sang vùng Degrading sớm, có thể xuất hiện trade-off (multi-modal không luôn vượt vibration-only).

### 4.2 Hai pha huấn luyện (stratified → temporal) vs train từ scratch

Hình liên quan:
- `figures/ablation/temporal_ft_r2_*.png` (fine-tune theo temporal có khởi tạo từ stratified)
- `figures/ablation/temporal_scratch_*.png` (fine-tune/train từ scratch)

Kết luận (macro-F1 theo *present classes* trên slice late-life, ví dụ [85, 100.1]%):
- Two-phase (init từ stratified): **present-label Macro-F1 ≈ 0.947**
- From scratch: **present-label Macro-F1 ≈ 0.250**

Ý nghĩa:
- Fine-tune theo thời gian nên bắt đầu từ một mô hình “đã học” đặc trưng tổng quát (stratified) để tránh collapse về lớp trội khi dữ liệu late-life bị lệch lớp.

## 5) Tóm tắt nhanh (điểm mạnh/yếu)

- Điểm mạnh: mô hình nhận diện **Fault** rất tốt (F1 ~ 0.92–0.96 ở các slice late-life).
- Điểm yếu cốt lõi: lớp **Degrading** (đặc biệt vùng [70,90]%) rất khó → F1 thấp trong các báo cáo temporal/alltest.
- Ý nghĩa triển khai: nếu mục tiêu là **cảnh báo sớm**, cần tập trung cải thiện Degrading (thiết kế nhãn, đặc trưng/sequence, hoặc thêm “context” dài hơn thay vì chỉ 1s window).

## 6) Gợi ý cải thiện lớp Degrading (ưu tiên “cảnh báo sớm”)

### 6.1 Kiểm soát nhãn (labeling) & đánh giá
- **Rà lại ngưỡng stage theo TTF**: thử các ngưỡng khác cho Degrading (ví dụ 50/85, 60/90, 70/90) để xem Degrading “sớm” có đang bị dán nhãn quá rộng/khó hay không.
- **Xem Degrading như bài toán thứ bậc (ordinal)**: Healthy < Degrading < Fault. Ordinal loss hoặc regression TTF + ngưỡng thường ổn định hơn 3-class “phẳng” khi ranh giới mơ hồ.
- **Báo cáo theo present classes** cho các temporal slice thiếu lớp (Healthy vắng mặt) để tránh Macro-F1 3 lớp làm méo nhận xét.

### 6.2 Tăng “ngữ cảnh” (context) thay vì chỉ 1s/window
- **Tăng window hoặc dùng multi-window context**: thay vì 1.0s độc lập, có thể encode nhiều window liên tiếp (ví dụ 5–20s) rồi mới dự đoán.
- **Pooling theo file thông minh hơn mean-logit**: dùng attention pooling / confidence-weighted pooling để nhấn mạnh các đoạn có tín hiệu Degrading rõ.

### 6.3 Làm giàu đặc trưng nhiệt độ (vì slope 1s thường rất nhiễu)
- Hiện tại nhiệt độ là 6-D thống kê/slope trong 1s; nên thử:
  - **trend dài hơn** (moving average/slope trên 30–300s),
  - **delta giữa các window** (tốc độ tăng nhiệt theo thời gian),
  - hoặc thêm đặc trưng “difference” giữa bearing và ambient (temp\_bearing − temp\_atm).

### 6.4 Tối ưu huấn luyện để tăng recall của Degrading
- Trong báo cáo temporal, Degrading có kiểu “precision cao, recall rất thấp” → mô hình dự đoán Degrading quá ít. Các cách thường hiệu quả:
  - **Focal loss** hoặc class-balanced loss để phạt mạnh các lỗi ở Degrading.
  - **Cost-sensitive inference**: tăng “ngưỡng” để dự đoán Fault, hoặc cộng bias cho logit lớp Degrading khi deploy (tối ưu theo mục tiêu cảnh báo sớm).
  - **Hard example mining**: tập trung các mẫu gần ranh 70–90% TTF (nhóm dễ nhầm) trong fine-tuning.

### 6.5 Điều chỉnh protocol temporal (giữ “thực tế” nhưng giảm domain shift)
- Thử thêm **buffer/gap** giữa train và test (ví dụ train [0,60], val [60,65], test [70,100]) để kiểm tra độ nhạy.
- Nếu mục tiêu là cảnh báo sớm, có thể cân nhắc fine-tune thêm một phần nhỏ ở vùng gần ranh (semi-supervised/self-training) nhưng vẫn giữ test “chưa thấy”.

## 7) Lộ trình giải pháp theo cấp độ (từ đơn giản → phức tạp)

### Cấp 0 (không sửa code, chỉ đổi cách báo cáo)
- Báo cáo thêm **present-classes Macro-F1** cho các temporal slice thiếu lớp (ví dụ chỉ Degrading/Fault).
- Tách kết quả theo vùng thời gian (ví dụ `[70,90]` vs `[90,100]`) để thấy rõ “cảnh báo sớm” khó ở đâu.

### Cấp 1 (nhẹ, chủ yếu đổi config/command)
- Tăng `window_seconds` (ví dụ 2–5s) và/hoặc giảm overlap bằng cách tăng `hop_seconds` để giảm nhiễu và giảm phụ thuộc giữa các window liền kề.
- Thử aggregation khác khi eval (ví dụ `vote` thay vì `mean`) để xem recall Degrading có cải thiện không.

### Cấp 2 (nhẹ, sửa ít code/feature)
- Làm giàu nhiệt độ: thay slope 1s bằng **trend dài hơn** (mean/slope trên 30–300s), thêm đặc trưng `(temp_bearing - temp_atm)`, và/hoặc delta giữa các window.
- Aggregation “thông minh” hơn: thay mean-logit bằng **confidence-weighted mean** (window càng tự tin càng có trọng số cao).

### Cấp 3 (trung bình, tối ưu huấn luyện/ra quyết định để ưu tiên Degrading)
- Dùng **Focal loss** hoặc class-balanced loss để tăng recall Degrading khi dữ liệu lệch lớp/ranh mơ hồ.
- Cost-sensitive inference: cộng bias vào logit lớp Degrading hoặc điều chỉnh ngưỡng để ưu tiên phát hiện Degrading (chấp nhận false alarm tăng).

### Cấp 4 (trung bình–khó, tăng context theo thời gian)
- Thay “mỗi window độc lập” bằng mô hình chuỗi (TCN/Transformer/LSTM) trên nhiều window liên tiếp (ví dụ 20–120s) rồi dự đoán stage.

### Cấp 5 (khó, thay cách đặt bài toán/nhãn)
- Học theo thứ bậc (Healthy < Degrading < Fault) hoặc regression dự đoán `TTF%/health index` rồi cắt ngưỡng.
- Định nghĩa Degrading dựa trên change-point/physics-inspired indicators (RMS, kurtosis, envelope spectrum, temperature trend) thay vì ngưỡng TTF cố định.

Khuyến nghị nếu bạn muốn cải thiện nhanh cho “cảnh báo sớm”:
- Ưu tiên thử theo thứ tự: **Cấp 2 (trend nhiệt dài hơn) → Cấp 3 (focal/cost-sensitive) → Cấp 4 (sequence)**.
