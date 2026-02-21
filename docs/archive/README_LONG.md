# Hướng Dẫn Pipeline Phân Loại Lỗi Vòng Bi (Bearing Fault Classification)

Dataset trong thư mục `data/` gồm các file CSV theo giờ (4 cột mỗi file):

- Thứ tự cột: rung (trục x), rung (trục y), nhiệt độ (vòng bi), nhiệt độ (môi trường)
- Tần số lấy mẫu: 25.600 Hz; thời lượng: 78,125 giây mỗi file (~2.000.000 dòng)
- Mục tiêu: Huấn luyện bộ phân loại (classifier) chẩn đoán/nhận dạng lỗi vòng bi theo chuỗi run-to-failure (TTF)

Tài liệu này liệt kê các bước thực dụng, có thể tái lập từ kiểm tra dữ liệu tới mô hình.

## Quy Trình Tổng Quan

1. Kiểm chứng dữ liệu (cấu trúc, sampling, tính toàn vẹn)
2. Tạo file nhãn manifest (loại lỗi hoặc giai đoạn sức khỏe) cho từng file
3. Chia tập theo thời gian (train/val/test) tránh rò rỉ
4. Cửa sổ hóa tín hiệu (1,0 s, bước nhảy 0,5 s)
5. Đặc trưng rung (STFT log-spectrogram)
6. Đặc trưng nhiệt độ (mean/std/slope theo mỗi cửa sổ)
7. Chọn mô hình (CNN 2D trên spectrogram + fusion nhiệt độ)
8. Huấn luyện với class weights + OneCycleLR; log chỉ số
9. Đánh giá (macro-F1, ma trận nhầm lẫn, early/late theo TTF)
10. Suy diễn (inference) cho file mới

Bên dưới là hướng dẫn chi tiết cho Bước (1) và (2). Các bước còn lại được phác thảo kèm hyperparameter đề xuất để triển khai tiếp.

---

## Bước 1 — Kiểm Chứng Dữ Liệu

Mục tiêu: xác nhận đủ 4 cột số, số dòng kỳ vọng, không có NaN, thống kê cơ bản.

Chọn một file đại diện (đổi tên nếu cần):

Kiểm tra nhanh bằng PowerShell

- Xem vài dòng đầu:
  ```powershell
  Get-Content -TotalCount 5 "data\LogFile_2022-06-20-17-00-31.csv"
  ```
- Kiểm tra có đúng 4 cột phân tách bởi dấu phẩy:
  ```powershell
  Get-Content -TotalCount 10 "data\LogFile_2022-06-20-17-00-31.csv" |
    ForEach-Object { ($_ -split ',').Length } | Sort-Object -Unique
  # Kỳ vọng: 4
  ```
- Đếm số dòng (~ 2.000.000; 25.600 Hz × 78,125 s):
  ```powershell
  $lines = (Get-Content -ReadCount 500000 "data\LogFile_2022-06-20-17-00-31.csv" | Measure-Object -Line).Lines
  $lines
  # (Tùy chọn) kiểm tra thời lượng ≈ 78,125 giây
  $duration = $lines / 25600.0; $duration
  ```
- Kiểm tra NaN/Inf (lấy mẫu 100k dòng đầu cho nhanh):
  ```powershell
  Get-Content -TotalCount 100000 "data\LogFile_2022-06-20-17-00-31.csv" |
    Select-String -SimpleMatch "NaN","nan","Inf","inf"
  ```

Thống kê nhanh bằng Python (tùy chọn)

- Cần `pandas`:
  ```powershell
  python -c "import pandas as pd,sys; f=sys.argv[1]; df=pd.read_csv(f,header=None); \
  print(df.shape); print(df.isna().sum().tolist()); print(df.describe().T)" \
  data/LogFile_2022-06-20-17-00-31.csv
  ```
  Diễn giải
- shape ≈ (2000000, 4)
- NaN = [0,0,0,0]
- Biên độ phù hợp dạng rung (x,y) và nhiệt độ (vòng bi, môi trường)

Nếu có bất thường

- Loại bỏ dòng NaN (nếu rất ít) hoặc tạm loại file khỏi huấn luyện để xử lý sau.
- Lập danh sách file loại trừ trong `data/exclude.txt` (tùy chọn) và xử lý trong data loader.

---

## Bước 2 — Tạo File Nhãn Manifest

Mục tiêu: tạo `data/manifest.csv` ánh xạ mỗi file → nhãn và tiến độ TTF.

Trường hợp A: Một run-to-failure không có loại lỗi chi tiết theo giờ

- Dùng tiến trình thời gian làm proxy cho giai đoạn sức khỏe (baseline 3 lớp):
  - healthy: 0–60% thời gian
  - degrading: 60–90%
  - fault: 90–100%
- Tính `ttf_percent` bằng cách sắp xếp file theo thời gian và ánh xạ chỉ số → [0,100].

Lệnh PowerShell tạo nhanh manifest

- Coi toàn bộ file là một run (`run1`). Có thể đổi ngưỡng/`fault_type` sau nếu có nhãn chi tiết.
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
- Kiểm tra nhanh:
  ```powershell
  Get-Content -TotalCount 10 "data\manifest.csv"
  ```

Trường hợp B: Có nhãn loại lỗi cụ thể theo giai đoạn/run

- Thay logic gán `fault_type` bằng nhãn thật của bạn.
- Nếu có nhiều run, gán `run_id` tương ứng (ví dụ `runA`, `runB`) và tính `ttf_percent` riêng trong từng run.

Schema manifest (CSV)

- Cột: `file,run_id,ttf_percent,fault_type`
- Ví dụ:
  ```csv
  file,run_id,ttf_percent,fault_type
  LogFile_2022-06-20-17-00-31.csv,run1,0.000,healthy
  LogFile_2022-06-22-08-00-31.csv,run1,98.457,fault
  ```

Xác nhận phân bố lớp

- Đếm theo lớp:
  ```powershell
  Import-Csv "data\manifest.csv" | Group-Object fault_type | Select-Object Name,Count
  ```

Ghi chú

- Bạn có thể tinh chỉnh ngưỡng giai đoạn (ví dụ 0–50/50–85/85–100) hoặc chuyển sang nhãn loại lỗi thật sau; các bước sau vẫn dùng `manifest.csv` như cũ.

---

## Bước 3 — Chia Tập Theo Thời Gian (tránh rò rỉ)

- Chia trong mỗi `run_id`: train = 0–70% TTF, val = 70–80%, test = 80–100% theo `ttf_percent`.
- Hoặc mạnh hơn: Leave-One-Run-Out nếu có nhiều run.

## Bước 4 — Cửa Sổ Hóa

- Độ dài cửa sổ: 1,0 s (25.600 mẫu), bước nhảy: 0,5 s (12.800 mẫu).
- Không trộn cửa sổ giữa các file hay giữa các ranh giới `fault_type` khác nhau.

## Bước 5 — Đặc Trưng Rung (STFT)

- Tiền xử lý: detrend, z-score theo từng file.
- STFT: `n_fft=4096`, `hop_length=1024`, Hann; log(1+|S|).
- Ghép kênh → tensor 2×F×T; chuẩn hoá theo băng tần; resize/crop 224×224.

## Bước 6 — Đặc Trưng Nhiệt Độ

- Mỗi cửa sổ tính cho bearing và atmospheric: mean, std, slope; z-score theo run.

## Bước 7 — Mô Hình

- CNN 2D (ResNet-18 nhẹ) với `in_channels=2` cho rung; late-fusion MLP cho đặc trưng nhiệt độ ở head.
- Loss: CrossEntropy + class weights; Optim: AdamW; Scheduler: OneCycleLR.

## Bước 8 — Huấn Luyện

- Batch 64–128; epochs 80–150; lr max ~3e-4; weight_decay 1e-2; early stopping theo macro-F1 trên val.
- Log accuracy, macro-F1, confusion matrix.

## Bước 9 — Đánh Giá

- Báo cáo macro-F1 và F1 theo lớp; confusion matrix.
- Early vs late TTF: đánh giá các subset `ttf_percent ∈ [70,90]` và `∈ (90,100]` để xem khả năng cảnh báo sớm.

## Bước 10 — Suy Diễn (Inference)

- Script đọc CSV mới, cửa sổ hóa, tạo đặc trưng, dự đoán theo cửa sổ, rồi tổng hợp theo file (majority vote/mean logit).

## Kế Hoạch Triển Khai (Gợi Ý Thư Mục Mã)

- Thêm module:
  - `datasets/logs_ttf.py` (đọc CSV, dùng `manifest.csv`, windowing)
  - `features/spectrogram.py` (STFT + chuẩn hoá)
  - `features/temp_features.py` (đặc trưng nhiệt độ theo cửa sổ)
  - `models/resnet2d.py` (CNN 2D) + `models/fusion_head.py` (fusion nhiệt độ)
  - `train_logs.py`, `eval_logs.py` (train, đánh giá)
  - `configs/logs_stft.yaml` (fs=25600, window=1.0, hop=0.5, n_fft=4096)

Khi hoàn tất Bước (1) và (2) và đã có `manifest.csv`, có thể bắt đầu triển khai các bước (3–10).
