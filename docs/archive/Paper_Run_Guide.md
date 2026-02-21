# Hướng Dẫn Chạy Thực Nghiệm — Phân Loại Lỗi Ổ Bi (Dùng Cho Bài Báo)

Tài liệu này mô tả từng bước để huấn luyện mô hình, đánh giá và thu thập kết quả phục vụ viết bài báo/hội thảo.

## 1) Chuẩn Bị Dữ Liệu

- Cấu trúc kỳ vọng
  - Thư mục dữ liệu: `data/`
  - Manifest CSV: `data/manifest.csv`
  - Các file CSV (rung/nhiệt) được tham chiếu trong manifest nằm trong `data/`.
- Định dạng manifest (bắt buộc có header):
  - `file,run_id,ttf_percent,fault_type`
  - Ví dụ: `bearing_001.csv,runA,83.5,fault`
- Tên nhãn phải thuộc một trong: `healthy`, `degrading`, `fault` (xem `datasets/logs_ttf.py`).

## 2) Môi Trường

- Khuyến nghị Python 3.9+
- Cài đặt thư viện (PyTorch, NumPy, scikit-learn, matplotlib — tùy chọn để vẽ biểu đồ):
  - Máy CPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
  - Chung: `pip install numpy scikit-learn matplotlib pyyaml`
- Tùy chọn: kiểm tra GPU với `python -c "import torch; print(torch.cuda.is_available())"`.

## 3) Chiến Lược Chia Tập

Nên báo cáo kết quả với chia theo thời gian (temporal) để sát thực tế; có thể bổ sung stratified để đối sánh/mổ xẻ mô hình.

- Temporal split (thực tế triển khai)
  - Train: 0–70% TTF, Val: 70–80% TTF, Test: 80–100% TTF
  - Cấu hình: `configs/logs_stft_full_temporal.yaml`
- Stratified split (cân bằng lớp, thuận tiện so sánh kiến trúc)
  - Cấu hình: `configs/logs_stft_train_strat.yaml`

Mẹo: để chọn mô hình “best” ổn định hơn, đặt `train.eval_every: 1` (đánh giá mỗi epoch).

## 4) Huấn Luyện (Train)

- Temporal (khuyến nghị):

```bash
python train_logs.py --config configs/logs_stft_full_temporal.yaml
```

- Stratified (tùy chọn so sánh):

```bash
python train_logs.py --config configs/logs_stft_train_strat.yaml
```

Kết quả sinh ra (trong `log.out_dir`, mặc định `runs/logs_stft/`):
- `best.pt` — checkpoint tốt nhất theo `log.save_best_by` (mặc định `macro_f1`)
- `train_log.csv` — log theo epoch (epoch, train_loss, val_acc, val_f1)
- `train_curves.png` — đồ thị huấn luyện (cần matplotlib)

Ghi chú
- Đánh giá diễn ra theo `train.eval_every` epoch; các epoch không đánh giá sẽ in `val_acc=NA | val_f1=NA` (có chủ đích).
- Lệch lớp được xử lý bằng CrossEntropy có trọng số lớp tính từ manifest.

## 5) Đánh Giá (Test)

Đánh giá checkpoint tốt nhất trên tập test và lưu báo cáo/biểu đồ.

- Temporal:

```bash
python eval_logs.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft/best.pt
```

- Stratified:

```bash
python eval_logs.py --config configs/logs_stft_train_strat.yaml --ckpt runs/logs_stft/best.pt
```

Đầu ra (lưu tại `runs/logs_stft/eval/`):
- `report.txt` — precision/recall/F1 theo lớp + accuracy, macro/weighted avg
- `confusion_matrix.csv` / `confusion_matrix.png` — ma trận nhầm lẫn theo file
- `f1_per_class.png` — biểu đồ F1 từng lớp
- `report_early_70_90.txt`, `confusion_matrix_early.*` — phân tích giai đoạn sớm (70–90% TTF)
- `report_late_90_100.txt`, `confusion_matrix_late.*` — phân tích giai đoạn muộn (90–100% TTF)

Hiển thị biểu đồ ra màn hình (vẫn lưu file) bằng cờ `--show`:

```bash
python eval_logs.py --config <config.yaml> --ckpt runs/logs_stft/best.pt --show
```

## 6) Suy Luận Trên File Lẻ (Inference)

Dự đoán nhãn cho một hoặc nhiều file CSV:

```bash
python infer.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft/best.pt path/to/file1.csv path/to/file2.csv
```

Script in nhãn dự đoán và độ tự tin; cách gộp theo file là trung bình xác suất theo các cửa sổ (windows).

## 7) Viết Báo Cáo/Bài Báo

- Kết quả chính (khuyến nghị): dùng split temporal — trích số liệu từ `report.txt` và hình `confusion_matrix.png`.
- Bổ sung: split stratified để minh họa năng lực mô hình trên dữ liệu cân bằng lớp.
- Phân tích theo giai đoạn: dùng `report_early_70_90.txt` và `report_late_90_100.txt` để bàn luận hiệu năng sớm/muộn.
- Phân lớp chi tiết: dùng `f1_per_class.png` và các ma trận nhầm lẫn.

Chạy nhiều seed
- Để tăng độ tin cậy, chạy nhiều seed (ví dụ 3–5). Sửa trong config:
  - `train.seed` và `random_seed` ở mức trên cùng
- Với mỗi seed, huấn luyện + đánh giá và lấy macro-F1/accuracy từ `report.txt`.
- Báo cáo trung bình ± độ lệch chuẩn.

Checklist tái lập (đính kèm phụ lục/supplementary)
- File cấu hình (đính kèm đúng YAML đã dùng)
- `runs/logs_stft/train_log.csv` và `eval/report.txt` cho từng lần chạy/seed
- Phiên bản phần mềm/phần cứng (Python, PyTorch, CUDA)
- Nếu dùng Git, chèn commit hash: `git rev-parse HEAD`

## 8) Mẹo Thực Tế

- Nếu validation dao động (tập val nhỏ), đặt `train.eval_every: 1` và cân nhắc tăng `train.val_max_windows` hoặc đặt `null` để dùng toàn bộ windows khi đánh giá.
- Chia temporal có thể khiến test thiếu một số lớp (ví dụ giai đoạn rất muộn hầu như không còn `healthy`). Cần giải thích điều này trong bài.
- Có thể đổi tiêu chí chọn best giữa `macro_f1` và `val_acc` qua `log.save_best_by`.

## 9) Gộp Dự Đoán Theo File

Cả khi đánh giá trong train và khi test đều gộp qua tất cả windows của một file. Mặc định là trung bình logits (mean). Bạn có thể thử "majority vote" bằng cách đổi tham số `agg` trong mã đánh giá.

---

Với quy trình này, bạn có thể tạo ra số liệu và biểu đồ sẵn sàng cho bài báo ở cả kịch bản temporal (sát triển khai) và stratified (cân bằng), đồng thời giữ khả năng tái lập nhờ cấu hình và seed cố định.
