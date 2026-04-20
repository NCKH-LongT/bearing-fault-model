# Cách chạy 3 baseline để lấy số liệu so sánh

Tài liệu này ghi lại cách chạy 3 baseline chuẩn hiện tại của repo để lấy số liệu so sánh cho model mới:

1. `svm_vib8`
2. `vibration_only_cnn`
3. `current_best_multimodal`

## 1. Ba baseline này có dùng chung dataset và manifest không?

Có.

Cả 3 baseline này đều được tổ chức để so sánh trên cùng nền dữ liệu của repo, tức là cùng:

- `data/manifest.csv`
- cùng logic chia split
- cùng nhãn
- cùng nguyên tắc đánh giá file-wise

Cụ thể:

- `svm_vib8` dùng `data/manifest.csv` thông qua config trong `classical_baselines/configs/`
- `vibration_only_cnn` dùng `data/manifest.csv` thông qua config trong `configs/ablation/`
- `current_best_multimodal` dùng `data/manifest.csv` thông qua config trong `configs/`

Điều này có nghĩa là:

- classical model và deep model có thể khác feature/model family
- nhưng vẫn đang được so sánh trên cùng bài toán, cùng dữ liệu, cùng protocol

Đó là điều làm cho phép so sánh có tính khách quan.

## 2. Cách chạy gọn nhất

Repo hiện đã có runner chung:

- `scripts/run_comparison_baseline.py`

Runner này giúp gọi baseline theo tên thay vì phải nhớ nhiều config rời rạc.

### Liệt kê baseline đã đăng ký

```powershell
python scripts/run_comparison_baseline.py --list
```

### Xem thông tin một baseline

```powershell
python scripts/run_comparison_baseline.py --baseline svm_vib8 --show
python scripts/run_comparison_baseline.py --baseline vibration_only_cnn --show
python scripts/run_comparison_baseline.py --baseline current_best_multimodal --show
```

## 3. Chạy baseline classical đơn giản: SVM vib 8

### Stratified

```powershell
python scripts/run_comparison_baseline.py --baseline svm_vib8 --protocol stratified --action train_eval
```

### Temporal

```powershell
python scripts/run_comparison_baseline.py --baseline svm_vib8 --protocol temporal --action train_eval
```

### File kết quả chính

- `runs/classical/svm_vib8_stratified/report_test.txt`
- `runs/classical/svm_vib8_temporal/report_test.txt`

Artifact thường có:

- `model.pkl`
- `report_test.txt`
- `confusion_matrix_test.csv`
- `predictions_test.csv`

## 4. Chạy baseline deep đơn cảm biến: Vibration-only CNN

### Stratified

```powershell
python scripts/run_comparison_baseline.py --baseline vibration_only_cnn --protocol stratified --action train_eval
```

### Temporal

```powershell
python scripts/run_comparison_baseline.py --baseline vibration_only_cnn --protocol temporal --action train_eval
```

### File kết quả chính

- `runs/ablations/strat_vib_only/eval/report.txt`
- `runs/ablations/temporal_vib_only/eval/report.txt`

Artifact thường có:

- `best.pt`
- `train_log.csv`
- `eval/report.txt`
- `eval/report_present.txt`
- `eval/confusion_matrix.png`

## 5. Chạy baseline mạnh hiện tại: Current best multi-modal

### Stratified

```powershell
python scripts/run_comparison_baseline.py --baseline current_best_multimodal --protocol stratified --action train_eval
```

### Temporal

```powershell
python scripts/run_comparison_baseline.py --baseline current_best_multimodal --protocol temporal --action train_eval
```

### Full-range nếu cần số liệu toàn vòng đời

```powershell
python scripts/run_comparison_baseline.py --baseline current_best_multimodal --protocol fullrange --action train_eval
```

Nếu muốn đánh giá full-range bằng `vote` thay vì `mean`:

```powershell
python scripts/run_comparison_baseline.py --baseline current_best_multimodal --protocol fullrange --action eval --agg vote
```

### File kết quả chính

- `runs/logs_stft_strat/eval/report.txt`
- `runs/logs_stft_temporal_gating_final/eval/report.txt`
- `runs/logs_stft_temporal_gating_fullrange/eval/report.txt`

## 6. Bộ lệnh tối thiểu để có bảng so sánh chính

Nếu mục tiêu là có số liệu tối thiểu để so sánh model mới với 3 baseline chuẩn, hãy chạy:

```powershell
python scripts/run_comparison_baseline.py --baseline svm_vib8 --protocol stratified --action train_eval
python scripts/run_comparison_baseline.py --baseline svm_vib8 --protocol temporal --action train_eval

python scripts/run_comparison_baseline.py --baseline vibration_only_cnn --protocol stratified --action train_eval
python scripts/run_comparison_baseline.py --baseline vibration_only_cnn --protocol temporal --action train_eval

python scripts/run_comparison_baseline.py --baseline current_best_multimodal --protocol stratified --action train_eval
python scripts/run_comparison_baseline.py --baseline current_best_multimodal --protocol temporal --action train_eval
```

Nếu paper của bạn còn báo cáo thêm kết quả toàn vòng đời, chạy thêm:

```powershell
python scripts/run_comparison_baseline.py --baseline current_best_multimodal --protocol fullrange --action train_eval
python scripts/run_comparison_baseline.py --baseline current_best_multimodal --protocol fullrange --action eval --agg vote
```

## 7. Nên lấy số ở đâu để đưa vào bảng so sánh?

Ưu tiên đọc từ các file report:

- Classical:
  - `runs/classical/svm_vib8_stratified/report_test.txt`
  - `runs/classical/svm_vib8_temporal/report_test.txt`
- Deep vibration-only:
  - `runs/ablations/strat_vib_only/eval/report.txt`
  - `runs/ablations/temporal_vib_only/eval/report.txt`
- Current best multi-modal:
  - `runs/logs_stft_strat/eval/report.txt`
  - `runs/logs_stft_temporal_gating_final/eval/report.txt`
  - `runs/logs_stft_temporal_gating_fullrange/eval/report.txt`

Các số thường cần lấy:

- `accuracy`
- `macro avg`
- F1 theo từng lớp

Nếu slice bị thiếu lớp, ưu tiên dùng thêm:

- `report_present.txt`

để tránh hiểu sai Macro-F1 trong các vùng temporal chỉ còn 1 hoặc 2 lớp.

## 8. Nếu không muốn dùng runner chung

Bạn vẫn có thể chạy trực tiếp từ config gốc.

### Classical SVM

```powershell
python classical_baselines/train_classical.py --config classical_baselines/configs/svm_vib8_stratified.yaml
python classical_baselines/train_classical.py --config classical_baselines/configs/svm_vib8_temporal.yaml
```

### Vibration-only CNN

```powershell
python train_logs.py --config configs/ablation/logs_stft_train_strat_vib.yaml
python eval_logs.py --config configs/ablation/logs_stft_train_strat_vib.yaml --ckpt runs/ablations/strat_vib_only/best.pt

python train_logs.py --config configs/ablation/logs_stft_temporal_vib_ft.yaml
python eval_logs.py --config configs/ablation/logs_stft_temporal_vib_ft.yaml --ckpt runs/ablations/temporal_vib_only/best.pt
```

### Current best multi-modal

```powershell
python train_logs.py --config configs/logs_stft_train_strat.yaml
python eval_logs.py --config configs/logs_stft_train_strat.yaml --ckpt runs/logs_stft_strat/best.pt

python train_logs.py --config configs/logs_stft_full_temporal_gating.yaml
python eval_logs.py --config configs/logs_stft_full_temporal_gating.yaml --ckpt runs/logs_stft_temporal_gating_final/best.pt
```

## 9. Kết luận ngắn

Có, 3 baseline này đều được chuẩn bị để dùng chung dataset và manifest của bài toán hiện tại.

Nói ngắn gọn:

- chung `manifest`
- chung split logic
- chung file-wise evaluation
- khác `feature/model family`

Đó là cách đúng để lấy số liệu so sánh cho paper.
