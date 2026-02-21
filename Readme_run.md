# Hướng dẫn chạy (Debug & Thực tế) cho Bearing Fault Pipeline

Tài liệu này hướng dẫn bạn chạy nhanh (debug) để kiểm tra pipeline và chạy đầy đủ để lấy số liệu/biểu đồ phục vụ bài báo. Mặc định kết quả được lưu trong `runs/logs_stft/`.

## 1) Chuẩn bị môi trường

- Yêu cầu: Python 3.9+; đã cài PyTorch (theo CUDA/CPU hệ thống), NumPy, PyYAML, scikit-learn, matplotlib.
- Khuyến nghị dùng virtualenv:

```powershell
python -m venv .venv
.venv\Scripts\Activate
pip install --upgrade pip
# Cài gói cần thiết (ví dụ)
pip install numpy pyyaml scikit-learn matplotlib
# Cài PyTorch phù hợp hệ thống (tham khảo trang chủ PyTorch cho đúng CUDA)
pip install torch torchvision torchaudio
```

### Chạy bằng GPU (RTX 2060 6GB)

1) Cài PyTorch có CUDA (ví dụ CUDA 11.8, phù hợp RTX 2060):
```powershell
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```
Nếu dùng bản CUDA mới hơn (cu121), thay `cu118` bằng `cu121` theo trang PyTorch.

2) Kiểm tra CUDA nhận GPU (PowerShell):
- Cách 1 (một dòng):
```powershell
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
- Cách 2 (here-string PowerShell):
```powershell
@'
import torch
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
else:
    print('device: CPU')
'@ | python -
```

3) Gợi ý cấu hình khi dùng GPU trên Windows:
- `train.num_workers: 2` (hoặc 4 nếu ổn định; Windows thường ít worker hơn Linux)
- `train.batch_size`: bắt đầu 32; nếu OOM, giảm 24 → 16 → 8
- `debug.seconds_cap`: khi thử nhanh để đỡ CPU/IO, giữ `5.0`. Chạy thật đặt `null`.

4) Theo dõi bộ nhớ GPU khi train/eval:
```powershell
nvidia-smi
```

### Sau bước 1: chạy từng câu lệnh (quick start)

1) Kiểm tra dữ liệu đã có trong `data/` (tùy chọn):
```powershell
Get-ChildItem -File data | Select-Object -First 5
```

2) Tạo `data/manifest.csv` (nếu chưa có):
```powershell
$files = Get-ChildItem -File "data/LogFile_*.csv" | Sort-Object Name
$n = $files.Count
$rows = for ($i=0; $i -lt $n; $i++) {
  $f = $files[$i]
  $p = if ($n -gt 1) { [math]::Round(($i * 100.0) / ($n - 1), 3) } else { 0.0 }
  $cls = if ($p -lt 60) { 'healthy' } elseif ($p -lt 90) { 'degrading' } else { 'fault' }
  [PSCustomObject]@{ file=$f.Name; run_id='run1'; ttf_percent=$p; fault_type=$cls }
}
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 "data/manifest.csv"
```

3) Xem nhanh vài dòng `manifest.csv`:
```powershell
Get-Content -TotalCount 5 data/manifest.csv
```

4) Chạy huấn luyện nhanh (debug) theo cấu hình mặc định:
```powershell
python train_logs.py --config configs/logs_stft.yaml
```

5) Đánh giá và xuất báo cáo/biểu đồ:
```powershell
python eval_logs.py --config configs/logs_stft.yaml --ckpt runs/logs_stft/best.pt
```

6) (Tùy chọn) Suy diễn 1 file cụ thể:
```powershell
python infer.py --config configs/logs_stft.yaml --ckpt runs/logs_stft/best.pt data/LogFile_2022-06-20-17-00-31.csv
```

7) Nâng lên chạy đầy đủ (thực tế): mở `configs/logs_stft.yaml` và chỉnh:
- `debug.limit_files_train/val/test: null`
- `debug.seconds_cap: null`
- Tăng `train.epochs` (ví dụ 80–150) nếu cần

Sau đó chạy lại huấn luyện và đánh giá:
```powershell
python train_logs.py --config configs/logs_stft.yaml
python eval_logs.py --config configs/logs_stft.yaml --ckpt runs/logs_stft/best.pt
```

## 2) Dữ liệu đầu vào

- Đặt các file CSV dữ liệu theo giờ vào thư mục `data/`, mỗi file có 4 cột: [rung_x, rung_y, temp_vong_bi, temp_moi_truong].
- Tạo file `data/manifest.csv` để ánh xạ từng file sang nhãn và tiến trình TTF:

```powershell
$files = Get-ChildItem -File "data/LogFile_*.csv" | Sort-Object Name
$n = $files.Count
$rows = for ($i=0; $i -lt $n; $i++) {
  $f = $files[$i]
  $p = if ($n -gt 1) { [math]::Round(($i * 100.0) / ($n - 1), 3) } else { 0.0 }
  $cls = if ($p -lt 60) { 'healthy' } elseif ($p -lt 90) { 'degrading' } else { 'fault' }
  [PSCustomObject]@{ file=$f.Name; run_id='run1'; ttf_percent=$p; fault_type=$cls }
}
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 "data/manifest.csv"
```

Ghi chú: Khi có nhãn thật theo loại lỗi/giai đoạn, thay `fault_type` tương ứng; nếu có nhiều run, đặt `run_id` theo run và tính `ttf_percent` riêng từng run.

## 3) Cấu hình

- File cấu hình: `configs/logs_stft.yaml` với các tham số:
  - Lấy mẫu: `sampling_rate=25600`, `window_seconds=1.0`, `hop_seconds=0.5`.
  - STFT: `n_fft=4096`, `hop_length=1024`, `window=hann`, `log_add=1.0`.
  - Kích thước đầu vào: `input_size: [224, 224]`.
  - Huấn luyện: `batch_size`, `epochs`, `lr`, `weight_decay`, `num_workers`, `seed`, `early_stop_patience`.
  - Lịch học: `optim.use_onecycle=true`.
  - Thư mục log: `log.out_dir: runs/logs_stft`.
  - Debug: `debug.limit_files_*` và `debug.seconds_cap` (giới hạn số file/số giây để chạy nhanh).

## 4) Chạy nhanh (Debug)

Mục tiêu: chạy nhanh vài epoch và ít dữ liệu để kiểm tra pipeline, sinh ra biểu đồ/ma trận nhầm lẫn thử.

- Đảm bảo trong `configs/logs_stft.yaml` có:
  - `debug.limit_files_train/val/test` là số nhỏ (ví dụ 8/4/4)
  - `debug.seconds_cap: 5.0` (chỉ đọc 5 giây đầu mỗi file)
  - `train.epochs: 5` (hoặc nhỏ để chạy nhanh)

- Huấn luyện nhanh:
```powershell
python train_logs.py --config configs/logs_stft.yaml
```
Kết quả sinh ra:
- Checkpoint tốt nhất: `runs/logs_stft/best.pt`
- Đường cong huấn luyện: `runs/logs_stft/train_curves.png`
- Lịch sử huấn luyện: `runs/logs_stft/train_log.csv`

- Đánh giá và xuất báo cáo/hình:
```powershell
python eval_logs.py --config configs/logs_stft.yaml --ckpt runs/logs_stft/best.pt
```
Kết quả sinh ra tại `runs/logs_stft/eval/`:
- `report.txt`: classification report tổng
- `confusion_matrix.csv` và `confusion_matrix.png`: ma trận nhầm lẫn
- `f1_per_class.png`: cột F1 theo lớp
- Phân tích theo TTF:
  - Early [70,90]: `report_early_70_90.txt`, `confusion_matrix_early.png`
  - Late (90,100]: `report_late_90_100.txt`, `confusion_matrix_late.png`

- Suy diễn một file mới (tùy chọn):
```powershell
python infer.py --config configs/logs_stft.yaml --ckpt runs/logs_stft/best.pt data/LogFile_2022-06-20-17-00-31.csv
```
Kết quả in ra nhãn dự đoán và độ tự tin.

## 5) Chạy đầy đủ (Thực tế)

Mục tiêu: dùng toàn bộ dữ liệu để huấn luyện, lấy số liệu/biểu đồ cuối cùng để viết bài.

- Chỉnh `configs/logs_stft.yaml`:
  - Tắt giới hạn debug: đặt `debug.limit_files_train/val/test: null`, `debug.seconds_cap: null`.
  - Tăng `train.epochs` (ví dụ 80–150), giữ `optim.use_onecycle: true` và `save_best_by: macro_f1`.
  - Tùy GPU/CPU, điều chỉnh `batch_size` và `num_workers`.

- Huấn luyện đầy đủ:
```powershell
python train_logs.py --config configs/logs_stft.yaml
```
- Đánh giá đầy đủ và xuất báo cáo/hình:
```powershell
python eval_logs.py --config configs/logs_stft.yaml --ckpt runs/logs_stft/best.pt
```

### Chọn file cấu hình qua biến môi trường (tiện đổi qua lại)

- Bạn có thể chọn file cấu hình bằng biến môi trường `BF_CONFIG` (PowerShell):
```powershell
$env:BF_CONFIG = "configs\logs_stft_train_strat.yaml"   # huấn luyện phân tầng ổn định
# Hoặc dùng temporal full:
$env:BF_CONFIG = "configs\logs_stft_full_temporal.yaml"

# Sau khi đặt, có thể chạy không cần --config:
python train_logs.py
python eval_logs.py --ckpt runs/logs_stft/best.pt

# Xoá biến môi trường khi không dùng nữa:
Remove-Item Env:BF_CONFIG
```

Hai cấu hình mẫu đi kèm:
- `configs/logs_stft_train_strat.yaml`: train/val/test phân tầng theo lớp; phù hợp để mô hình học đủ lớp trên 1 run.
- `configs/logs_stft_full_temporal.yaml`: chia theo thời gian (temporal) cho chạy đầy đủ sát thực tế.

Artefacts để dùng trong bài báo:
- Đường cong train/val: `runs/logs_stft/train_curves.png`
- Báo cáo tổng: `runs/logs_stft/eval/report.txt`
- Ma trận nhầm lẫn: `runs/logs_stft/eval/confusion_matrix.png`
- F1 theo lớp: `runs/logs_stft/eval/f1_per_class.png`
- Đánh giá theo TTF: `runs/logs_stft/eval/report_early_70_90.txt`, `report_late_90_100.txt` cùng hình CM tương ứng

## 6) Gợi ý và lưu ý

- Mô hình: ResNet-18 nhẹ (2 kênh rung) + fusion đặc trưng nhiệt độ (6 chiều) ở head.
- Chia tập theo TTF: train [0–70], val [70–80], test [80–100]; đảm bảo không rò rỉ theo thời gian.
- Nếu thiếu `matplotlib`, vẫn có CSV/báo cáo, nhưng không xuất hình. Cài thêm: `pip install matplotlib`.
- Lỗi thiếu RAM/GPU: giảm `train.batch_size` hoặc bật `debug.seconds_cap` khi kiểm thử.
- CSV quá ngắn (IndexError): file không đủ dài cho cửa sổ 1.0s; cân nhắc loại file đó khỏi manifest hoặc giảm `window_seconds` khi thử nghiệm.
- Tái lập kết quả: cấu hình có `seed`, nhưng với PyTorch GPU có thể vẫn có sai khác nhỏ.

## 7) Kiểm soát Early Stopping

- Mặc định dự án dùng Early Stopping để dừng sớm khi `macro_f1` (hoặc `val_acc` tùy `log.save_best_by`) không cải thiện sau một số epoch.
- Hai cách điều khiển:
  - Tắt hẳn Early Stopping: thêm vào dưới khối `train:` trong file cấu hình:
    - `early_stop: false`
  - Giữ Early Stopping nhưng nới lỏng: đặt `early_stop_patience` lớn hơn (ví dụ 50–100), hoặc đặt `0` để vô hiệu hóa.
- Lưu ý: nếu tập validation quá nhỏ, `val_acc` và `macro_f1` dao động mạnh (bước nhảy lớn), dễ kích hoạt dừng sớm. Hãy tăng tỉ lệ `val` hoặc dùng cấu hình phân tầng `configs/logs_stft_train_strat.yaml` để có phân phối lớp ổn định hơn.

Ví dụ chạy với cấu hình phân tầng và tắt Early Stopping:
```powershell
# Sửa configs/logs_stft_train_strat.yaml:
#   train:
#     early_stop: false
python train_logs.py --config configs/logs_stft_train_strat.yaml
```

---

Sau khi chạy đầy đủ, bạn có thể đưa trực tiếp các hình PNG/CSV trong `runs/logs_stft/` vào bài báo (ma trận nhầm lẫn, F1 theo lớp, đường cong huấn luyện) và trích số liệu từ `report.txt` (accuracy, macro-F1) cũng như các báo cáo early/late TTF.
