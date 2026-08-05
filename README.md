# Phân Loại Trạng Thái Ổ Bi (Run-to-Failure, STFT + Nhiệt Độ)

Repository này chứa pipeline đa phương thức cho bài toán phân loại ba giai đoạn trạng thái ổ bi (healthy, degrading, fault) trên dữ liệu run-to-failure. Revision hiện hành dùng held-out multi-class split ở cấp file, validation-only model selection và five-seed confirmation. Temporal/full-range lịch sử chỉ còn là retrospective analysis.

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
- `paper/`: mã nguồn LaTeX, bibliography, Springer style và các figure/report của bài báo.
- `paper/run_revision.py`: runner từng bước cho audit, primary revision, baseline và build LaTeX.
- `paper/README.md`: tài liệu canonical về kết quả, artifact release, chạy lại và roadmap nâng cấp.
- `paper/REVISION_RUN_CHECKLIST.md`: checklist và nhật ký revision.
- `docs/CANONICAL_RUN_AND_CLASSIC_COMPARE.md`: chốt bộ run chuẩn và lệnh so sánh baseline classic.

## Dữ liệu và manifest

- Mỗi file CSV đầu vào gồm các cột `[vib_x, vib_y, temp_bearing, temp_atm]`, lấy mẫu ở 25.6 kHz.
- `data/manifest.csv` gồm các cột: `file, run_id, ttf_percent, fault_type` (nhãn: healthy/degrading/fault).
- Cửa sổ trượt: `window_seconds=1.0`, `hop_seconds=0.5` cho cả rung và nhiệt độ.
- Dữ liệu thô không nằm trong Git hoặc artifact release. Người dùng phải cung cấp CSV hợp lệ theo manifest.

## Thiết lập

- Khuyến nghị Python 3.10+ và PyTorch >= 2.0.
- Cài dependency bằng môi trường bạn đang dùng; có thể bật CUDA để chạy GPU/AMP.

## Bắt đầu nhanh

1. Đọc `paper/README.md`.
2. Cài dependency: `python -m pip install -r requirements-paper.txt`.
3. Tạo cache: `python scripts/cache_dataset_npy.py --workers 2`.
4. Kiểm tra: `python paper/run_revision.py preflight && python paper/run_revision.py smoke`.
5. Audit: `python paper/run_revision.py audit`.

## Các thiết lập quan trọng

- STFT của pipeline paper-sync: `n_fft=2048`, `hop_length=512`, `window='hann'`, `log_add=1.0`.
- Kích thước ảnh của paper-sync: `input_size=[160,160]`.
- Optimizer: AdamW, label smoothing, AMP, balanced sampling, early stopping.
- Chia temporal: train `[0,60]`, val `[60,70]`, test `[70,100]` theo TTF.

## Clone sang máy khác

1. Clone repository và tạo `.venv` cục bộ.
2. Cung cấp raw CSV trong `data/` theo `data/manifest.csv` nếu cần chạy lại.
3. Nếu chỉ inference/audit artifact, tải release theo `paper/README.md`; không cần đưa `runs/` vào Git.

## Tái lập figure cho paper

- Luồng revision hiện hành được mô tả trong `paper/README.md`.
- Artifact được sinh ra dưới `runs/paper_sync/...`.
- Bản đã chọn để dùng cho paper sẽ được sync vào `paper/figures/stratified`, `paper/figures/temporal` và `paper/figures/fullrange`.

## Ghi nhận nguồn gốc

- Nguồn upstream: CNN-for-Paderborn-Bearing-Dataset (mdzalfirdausi) — https://github.com/mdzalfirdausi/CNN-for-Paderborn-Bearing-Dataset (truy cập ngày 2026-02-20).
- Project này kế thừa khung train/eval ở mức cao và mở rộng cho bài toán run-to-failure bằng: (i) chia temporal theo TTF để giảm leakage, (ii) log-spectrogram STFT hai trục với z-score theo cửa sổ và chuẩn hóa theo tần số, (iii) ghép đặc trưng nhiệt độ 6 chiều bằng một head gọn nhẹ, và (iv) gộp dự đoán theo file bằng mean của logit trước softmax.
- Xem `ATTRIBUTIONS.md` để biết chi tiết provenance và cách trích dẫn nguồn gốc.

## Giấy phép

- Xem giấy phép của các thành phần kế thừa trong `ATTRIBUTIONS.md`. Giấy phép tổng thể của project sẽ được căn chỉnh theo yêu cầu của nguồn upstream.
