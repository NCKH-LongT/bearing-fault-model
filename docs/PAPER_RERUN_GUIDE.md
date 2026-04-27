# Hướng Dẫn Chạy Lại Paper

Tài liệu này mô tả cách khuyến nghị để tái sinh artifact của paper từ một môi trường làm việc sạch.

## 1. Môi trường

Nếu có sẵn, hãy dùng virtual environment của project:

```powershell
.venv\Scripts\Activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Thiết lập được kỳ vọng cho lần đồng bộ paper hiện tại:

- Python từ `.venv`
- PyTorch có CUDA
- GPU đã dùng ở lần chạy lại gần nhất: RTX 2060 6GB

## 2. Đầu vào bắt buộc

Trước khi chạy pipeline paper, cần kiểm tra:

- `data/manifest.csv` tồn tại và khớp với dataset hiện tại.
- `runs/logs_stft_strat/auto_r22/best.pt` tồn tại.

Các con số hiện tại trong paper giả định `auto_r22` là checkpoint stratified chuẩn.

## 3. Dọn các thư mục output

Nếu muốn chạy lại paper từ đầu, chỉ nên xóa output của `paper_sync`, không xóa các thí nghiệm cũ:

```powershell
Remove-Item -Recurse -Force runs\paper_sync -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force figures\temporal -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force figures\fullrange -ErrorAction SilentlyContinue
```

Không được xóa `runs/logs_stft_strat/auto_r22`.

## 4. Luồng chạy lại được khuyến nghị

Quy trình nên dùng:

1. Tái sử dụng checkpoint stratified chuẩn.
2. Chỉ train lại model temporal.
3. Đánh giá lại temporal mean.
4. Đánh giá lại full-range mean.
5. Đánh giá lại full-range vote.
6. Đồng bộ artifact đã chọn vào `figures/`.

Chạy lệnh:

```powershell
python scripts/run_paper_sync.py --python .venv/Scripts/python.exe --sync-figures
```

Lệnh này dùng:

- `configs/best_temporal.yaml` cho train/eval temporal
- `configs/best_fullrange_eval.yaml` cho eval full-range
- `runs/logs_stft_strat/auto_r22/best.pt` làm checkpoint khởi tạo từ stratified

## 5. Tùy chọn: sinh lại cả artifact stratified

Nếu muốn có thêm một run stratified mới chỉ để đối chiếu:

```powershell
python scripts/run_paper_sync.py --python .venv/Scripts/python.exe --run-stratified --sync-figures
```

Lưu ý:

- Đây không phải luồng khuyến nghị để tái lập bộ số mạnh nhất trong paper.
- Một lần retrain stratified mới có thể kém hơn `auto_r22`.

## 6. Chạy thủ công từng bước

Nếu muốn chạy thủ công từng bước:

### 6.1 Fine-tune temporal

```powershell
python train_logs.py --config configs/best_temporal.yaml
```

Output kỳ vọng:

- `runs/paper_sync/temporal/best.pt`

### 6.2 Đánh giá temporal mean

```powershell
python eval_logs.py --config configs/best_temporal.yaml --ckpt runs/paper_sync/temporal/best.pt
```

Output kỳ vọng:

- `runs/paper_sync/temporal/eval/`

### 6.3 Đánh giá full-range mean

```powershell
python eval_logs.py --config configs/best_fullrange_eval.yaml --ckpt runs/paper_sync/temporal/best.pt
```

Output kỳ vọng:

- `runs/paper_sync/fullrange/eval/`

### 6.4 Đánh giá full-range vote

```powershell
python eval_logs.py --config configs/best_fullrange_eval.yaml --ckpt runs/paper_sync/temporal/best.pt --agg vote
```

Output kỳ vọng:

- `runs/paper_sync/fullrange/eval_vote/`

### 6.5 Đồng bộ figures đã chọn

```powershell
python scripts/run_paper_sync.py --python .venv/Scripts/python.exe --skip-train --sync-figures
```

Lệnh này sẽ copy output đã chọn vào:

- `figures/stratified`
- `figures/temporal`
- `figures/fullrange`

## 7. Các con số chính kỳ vọng

Sau khi sync thành công, kiểm tra:

- `figures/temporal/report.txt`
  - Accuracy `0.8974`
- `figures/temporal/report_early_70_90.txt`
  - Accuracy `0.8846`
  - F1 của `degrading` là `0.9388`
- `figures/temporal/report_late_90_100.txt`
  - Accuracy `0.9231`
  - F1 của `fault` là `0.9600`
- `figures/fullrange/report.txt`
  - Accuracy `0.8915`
  - Macro-F1 `0.8739`

## 8. Các config chuẩn

Dùng các config này cho paper:

- `configs/best_stratified_ref.yaml`
- `configs/best_temporal.yaml`
- `configs/best_fullrange_eval.yaml`

Các file `paper_sync_*.yaml` vẫn được giữ lại như lớp cấu hình nền, nhưng không còn là tên chuẩn chính để trích trong paper.

Các config exploratory cũ đã được loại khỏi reviewer repo. Nếu cần đối chiếu biến thể, chỉ dùng các config còn lại trong `configs/ablation/*.yaml`.

## 9. Các điểm cần lưu ý

- Macro-F1 present-class của temporal trong bản thảo hiện đang được diễn giải thủ công từ các lớp thực sự hiện diện, không lấy trực tiếp từ `report_present.txt`.
- Kết quả headline của full-range hiện dựa trên `majority vote`, không phải mean-logit.
- Việc retrain stratified hiện chưa đủ ổn định để thay `auto_r22` làm checkpoint khởi tạo chuẩn.
