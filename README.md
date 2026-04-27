# Phân Loại Trạng Thái Ổ Bi (Run-to-Failure, STFT + Nhiệt Độ)

Repository này chứa một pipeline đa phương thức gọn nhẹ cho bài toán phân loại 3 giai đoạn trạng thái ổ bi (healthy, degrading, fault) trên dữ liệu run-to-failure. Hệ thống sử dụng log-spectrogram STFT từ hai trục rung và đặc trưng nhiệt độ 6 chiều, kết hợp với quy trình đánh giá temporal có kiểm soát leakage theo trục time-to-failure (TTF).

## Điểm chính

- Huấn luyện hai pha: stratified (phát triển) -> fine-tune temporal (gần với bối cảnh triển khai).
- Đặc trưng rung: STFT log-magnitude với z-score theo từng cửa sổ và chuẩn hóa theo từng tần số; ảnh được resize theo `input_size`.
- Đặc trưng nhiệt độ: 6 chiều cho mỗi cửa sổ (mean/std/slope cho nhiệt độ ổ bi và nhiệt độ môi trường).
- Mô hình: CNN kiểu ResNet-18 nhỏ cho nhánh rung (`in_ch=2`) + phép chiếu tuyến tính cho nhánh nhiệt độ + bộ phân loại late fusion.
- Dự đoán theo file: lấy trung bình các logit trước softmax qua các cửa sổ trong cùng một file (mean-logit) để ra nhãn cuối.

## Cấu trúc repository

- `configs/`: các file YAML cho các run stratified, temporal và các biến thể khác.
- `datasets/`: bộ nạp dữ liệu có hỗ trợ chia tập theo TTF (`LogsTTFDataset`).
- `features/`: biến đổi STFT và trích xuất đặc trưng nhiệt độ.
- `models/`: mô hình ResNet2D nhỏ có ghép nhánh nhiệt độ.
- `runs/`: nơi lưu output, checkpoint và các kết quả đánh giá.
- `figures/`: các figure và report đã chọn để dùng cho paper.
- `docs/PAPER_RERUN_GUIDE.md`: hướng dẫn từng bước để chạy lại pipeline paper.
- `docs/CANONICAL_RUN_AND_CLASSIC_COMPARE.md`: chốt bộ run chuẩn và lệnh so sánh baseline classic.
- `docs/paper_sync_issues.md`: danh sách các điểm còn lệch giữa paper và pipeline hiện tại.

## Dữ liệu và manifest

- Mỗi file CSV đầu vào gồm các cột `[vib_x, vib_y, temp_bearing, temp_atm]`, lấy mẫu ở 25.6 kHz.
- `data/manifest.csv` gồm các cột: `file, run_id, ttf_percent, fault_type` (nhãn: healthy/degrading/fault).
- Cửa sổ trượt: `window_seconds=1.0`, `hop_seconds=0.5` cho cả rung và nhiệt độ.
- Ghi chú về dữ liệu thô: `data/` hiện chứa khoảng 130 file CSV (~17 GB). Nếu muốn lấy đầy đủ dữ liệu sau khi clone trên máy khác, hãy dùng Git LFS và chạy `git lfs pull`.

## Thiết lập

- Khuyến nghị Python 3.10+ và PyTorch >= 2.0.
- Cài dependency bằng môi trường bạn đang dùng; có thể bật CUDA để chạy GPU/AMP.

## Bắt đầu nhanh

1. Chạy lại pipeline paper theo cách chuẩn
   - `python scripts/run_paper_sync.py --python .venv/Scripts/python.exe --sync-figures`

2. Chạy lại paper theo từng bước thủ công
   - `python train_logs.py --config configs/best_temporal.yaml`
   - `python eval_logs.py --config configs/best_temporal.yaml --ckpt runs/paper_sync/temporal/best.pt`
   - `python eval_logs.py --config configs/best_fullrange_eval.yaml --ckpt runs/paper_sync/temporal/best.pt --agg vote`

3. Các config phát triển kiểu cũ
   - Chỉ dùng khi thật sự cần tra cứu các nhánh cũ hoặc ablation lịch sử
   - Nên đọc `configs/README.md` trước khi dùng các config không thuộc pipeline paper hiện tại

4. Bộ config chuẩn để viết paper
   - `configs/best_stratified_ref.yaml`
   - `configs/best_temporal.yaml`
   - `configs/best_fullrange_eval.yaml`

## Các thiết lập quan trọng

- STFT của pipeline paper-sync: `n_fft=2048`, `hop_length=512`, `window='hann'`, `log_add=1.0`.
- Kích thước ảnh của paper-sync: `input_size=[160,160]`.
- Optimizer: AdamW, label smoothing, AMP, balanced sampling, early stopping.
- Chia temporal: train `[0,60]`, val `[60,70]`, test `[70,100]` theo TTF.

## Clone sang máy khác

1. Cài Git LFS: `git lfs install`
2. Clone repository như bình thường.
3. Nếu remote có dữ liệu được track bằng LFS, lấy dữ liệu bằng `git lfs pull`.
4. Tạo môi trường Python cục bộ riêng; `.venv/` cố ý không được version.
5. Nếu muốn version dữ liệu thô trong repo này, hãy track bằng Git LFS và add tường minh, ví dụ `git add -f data/*.csv`.

## Tái lập figure cho paper

- Chạy pipeline `paper_sync` theo hướng dẫn trong `docs/PAPER_RERUN_GUIDE.md`.
- Artifact được sinh ra dưới `runs/paper_sync/...`.
- Bản đã chọn để dùng cho paper sẽ được sync vào `figures/stratified`, `figures/temporal` và `figures/fullrange`.

## Ghi nhận nguồn gốc

- Nguồn upstream: CNN-for-Paderborn-Bearing-Dataset (mdzalfirdausi) — https://github.com/mdzalfirdausi/CNN-for-Paderborn-Bearing-Dataset (truy cập ngày 2026-02-20).
- Project này kế thừa khung train/eval ở mức cao và mở rộng cho bài toán run-to-failure bằng: (i) chia temporal theo TTF để giảm leakage, (ii) log-spectrogram STFT hai trục với z-score theo cửa sổ và chuẩn hóa theo tần số, (iii) ghép đặc trưng nhiệt độ 6 chiều bằng một head gọn nhẹ, và (iv) gộp dự đoán theo file bằng mean của logit trước softmax.
- Xem `ATTRIBUTIONS.md` để biết chi tiết provenance và cách trích dẫn nguồn gốc.

## Giấy phép

- Xem giấy phép của các thành phần kế thừa trong `ATTRIBUTIONS.md`. Giấy phép tổng thể của project sẽ được căn chỉnh theo yêu cầu của nguồn upstream.
