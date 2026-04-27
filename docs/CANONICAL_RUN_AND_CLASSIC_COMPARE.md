# Bộ Run Chuẩn Và Cách So Sánh Với Baseline Classic

Tài liệu này chốt một bộ run chuẩn duy nhất để dùng cho paper, đồng thời ghi lại các lệnh tối thiểu để chạy baseline classic và so sánh với mô hình hiện tại.

## 1. Bộ chuẩn để viết paper

Chỉ dùng 3 thành phần sau làm chuẩn:

- Stratified chuẩn để cite:
  - checkpoint: `runs/logs_stft_strat/auto_r22/best.pt`
  - config tham chiếu: `configs/best_stratified_ref.yaml`
  - report: `runs/logs_stft_strat/auto_r22/eval/report.txt`
- Temporal chuẩn để chạy lại:
  - config train/eval: `configs/best_temporal.yaml`
  - checkpoint sau khi train: `runs/paper_sync/temporal/best.pt`
  - report: `runs/paper_sync/temporal/eval/report.txt`
- Full-range chuẩn để đánh giá toàn vòng đời:
  - config eval: `configs/best_fullrange_eval.yaml`
  - checkpoint dùng để eval: `runs/paper_sync/temporal/best.pt`
  - report vote: `runs/paper_sync/fullrange/eval_vote/report.txt`

## 2. Ý nghĩa từng config chuẩn

- `best_stratified_ref.yaml`
  - Đây là snapshot đúng của run tốt nhất `auto_r22`.
  - Dùng để viết paper và đối chiếu hyperparameter.
  - Không nên xem đây là cấu hình mặc định để retrain độc lập từ đầu.
- `best_temporal.yaml`
  - Đây là config chuẩn để anh/chị tự chạy lại temporal train.
  - Nó khởi tạo từ `auto_r22`, tức là đúng logic đang dùng cho paper.
- `best_fullrange_eval.yaml`
  - Không phải config để tạo một "best run mới".
  - Nó chỉ phục vụ đánh giá checkpoint temporal chuẩn trên toàn miền TTF.

## 3. Các lệnh cần dùng cho mô hình chính

### 3.1 Stratified chuẩn

Không cần train lại nếu mục tiêu là giữ đúng chuẩn paper hiện tại.

Nếu chỉ muốn xem config và checkpoint chuẩn:

```powershell
Get-Content configs/best_stratified_ref.yaml
Get-ChildItem runs/logs_stft_strat/auto_r22
```

### 3.2 Temporal train chuẩn

```powershell
python train_logs.py --config configs/best_temporal.yaml
python eval_logs.py --config configs/best_temporal.yaml --ckpt runs/paper_sync/temporal/best.pt
```

### 3.3 Full-range eval chuẩn

Mean-logit:

```powershell
python eval_logs.py --config configs/best_fullrange_eval.yaml --ckpt runs/paper_sync/temporal/best.pt
```

Majority vote:

```powershell
python eval_logs.py --config configs/best_fullrange_eval.yaml --ckpt runs/paper_sync/temporal/best.pt --agg vote
```

## 4. Các số chuẩn đang dùng trong paper

- Stratified `auto_r22`
  - Accuracy `0.7241`
  - Macro-F1 `0.7762`
- Temporal `[70,100]%`
  - Accuracy `0.8974`
  - F1 Degrading `0.9200`
  - F1 Fault `0.8889`
- Temporal sớm `[70,90]%`
  - Accuracy `0.8846`
  - F1 Degrading `0.9388`
- Temporal muộn `[90,100]%`
  - Accuracy `0.9231`
  - F1 Fault `0.9600`
- Full-range vote `[0,100]%`
  - Accuracy `0.8915`
  - Macro-F1 `0.8739`

## 5. Baseline classic cần chạy để so sánh

Baseline classic nên dùng là SVM 8D:

- Stratified:
  - `classical_baselines/configs/svm_vib8_stratified.yaml`
- Temporal công bằng để so với mô hình chính:
  - `classical_baselines/configs/svm_vib8_strat_train_temporal_eval.yaml`

Không dùng `classical_baselines/configs/svm_vib8_temporal.yaml` cho bảng chính của paper, vì temporal-train thuần không nhìn thấy lớp Fault trong train.

## 6. Lệnh chạy baseline classic

### 6.1 Stratified

```powershell
python scripts/run_comparison_baseline.py --baseline svm_vib8 --protocol stratified --action train_eval
```

Kết quả chính:

- `runs/classical/svm_vib8_stratified/report_test.txt`

### 6.2 Temporal công bằng

```powershell
python scripts/run_comparison_baseline.py --baseline svm_vib8 --protocol temporal --action train_eval
```

Kết quả chính:

- `runs/classical/svm_vib8_strat_train_temporal_eval/report_test.txt`

## 7. Nếu muốn chạy đủ bộ so sánh cho paper

### 7.1 Mô hình chính

```powershell
python train_logs.py --config configs/best_temporal.yaml
python eval_logs.py --config configs/best_temporal.yaml --ckpt runs/paper_sync/temporal/best.pt
python eval_logs.py --config configs/best_fullrange_eval.yaml --ckpt runs/paper_sync/temporal/best.pt --agg vote
```

### 7.2 Baseline classic

```powershell
python scripts/run_comparison_baseline.py --baseline svm_vib8 --protocol stratified --action train_eval
python scripts/run_comparison_baseline.py --baseline svm_vib8 --protocol temporal --action train_eval
```

### 7.3 Baseline deep vibration-only

```powershell
python scripts/run_comparison_baseline.py --baseline vibration_only_cnn --protocol stratified --action train_eval
python scripts/run_comparison_baseline.py --baseline vibration_only_cnn --protocol temporal --action train_eval
```

## 8. Kết luận sử dụng

Nếu mục tiêu là viết paper gọn và nhất quán:

- chỉ cite stratified từ `auto_r22`
- chỉ train lại temporal bằng `configs/best_temporal.yaml`
- chỉ eval full-range bằng `configs/best_fullrange_eval.yaml`
- chỉ dùng SVM classical theo protocol `temporal` công bằng để so sánh chính

Config exploratory cũ đã được loại khỏi reviewer repo; với baseline classical, cũng không dùng `svm_vib8_temporal.yaml` cho bảng chính của paper.
