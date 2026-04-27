# Các Vấn Đề Khi Đồng Bộ Paper

Ngày cập nhật: 2026-04-27

Tài liệu này ghi lại trạng thái hiện tại sau khi đã chốt bộ run chuẩn cho paper và làm sạch `main.tex`.

## 1. Các điểm đã xử lý

- `main.tex` đã chuyển sang bộ chuẩn hiện tại.
  - Stratified tham chiếu: `runs/logs_stft_strat/auto_r22/best.pt`
  - Temporal chuẩn: `configs/best_temporal.yaml`
  - Full-range chuẩn: `configs/best_fullrange_eval.yaml`

- Caption và diễn giải temporal trong `main.tex` đã được sửa.
  - Hình `figures/temporal/*` hiện được mô tả đúng là lát deployment `[70,100]%`.
  - Không còn gọi nhầm hình này là lát chỉ riêng late-life.

- Phần mô tả `No class-weights` không còn xuất hiện như một cấu hình cuối của paper.
  - `nocw` chỉ còn là artifact ablation lịch sử.

- Bảng baseline và bảng ablation trong `main.tex` đã được chuẩn hóa lại theo artifact đang giữ.
  - Classical baseline chính dùng `runs/classical/svm_vib8_stratified/report_test.txt` và `runs/classical/svm_vib8_strat_train_temporal_eval/report_test.txt`.
  - Deep vibration-only chính dùng `runs/ablations/strat_vib_only/eval/report.txt` và `runs/ablations/temporal_vib_only_fair/eval/report.txt`.
  - Các hàng classical slice-wise cũ đã bị loại khỏi bảng ablation để tránh trộn protocol.

- Bug `report_present.txt` trong `eval_logs.py` đã được sửa và artifact đã regenerate lại.
  - Logic `present-class` đã đổi từ `set(y_true) | set(y_pred)` sang chỉ `set(y_true)`.
  - `figures/temporal/report_present.txt` không còn giữ `healthy` với `support = 0`.
  - Temporal `[70,100]%` hiện khớp artifact với `Macro-F1_present = 0.9044`.

## 2. Vấn đề còn mở

- Stratified tốt nhất vẫn phụ thuộc vào checkpoint lịch sử `auto_r22`.
  - Việc train lại stratified từ đầu hiện chưa tái tạo ổn định đúng chất lượng checkpoint này.
  - Nếu muốn giữ số paper hiện tại, nên xem `auto_r22` là mốc tham chiếu cố định.

## 3. Bộ số chuẩn hiện đang dùng trong paper

- Stratified:
  - `Accuracy = 0.7241`
  - `Macro-F1 = 0.7762`
- Temporal `[70,100]%`:
  - `Accuracy = 0.8974`
  - `Macro-F1_present = 0.9044`
  - `F1 degrading = 0.9200`
  - `F1 fault = 0.8889`
- Temporal sớm `[70,90]%`:
  - `Accuracy = 0.8846`
  - `F1 degrading = 0.9388`
- Temporal muộn `[90,100]%`:
  - `Accuracy = 0.9231`
  - `F1 fault = 0.9600`
- Full-range vote `[0,100]%`:
  - `Accuracy = 0.8915`
  - `Macro-F1 = 0.8739`

## 4. Vị trí artifact chuẩn

- Stratified:
  - `figures/stratified`
- Temporal:
  - `figures/temporal`
- Full-range:
  - `figures/fullrange`
