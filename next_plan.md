# Kế Hoạch Tập Trung: Stratified → Fine-tune Temporal (TTF)

## Mục tiêu
- Huấn luyện ổn định bằng split stratified để mô hình học đủ 3 lớp (healthy/degrading/fault).
- Fine-tune trên temporal TTF (0–85/85–90/90–100.1) nhằm khớp bối cảnh triển khai theo thời gian, tránh rò rỉ tương lai.
- Đảm bảo đánh giá luôn đủ 3 lớp và có biểu đồ theo TTF.

## Thay đổi chính (đã xong)
- Bật Early Stopping cho cả 2 cấu hình.
- Thêm cơ chế fine‑tune: `train.init_from` để nạp trọng số trước khi train (train_logs.py).
- Temporal YAML cập nhật để fine‑tune với LR nhỏ và patience phù hợp; đánh giá luôn ghi report 3 lớp.

## Các bước chạy (chi tiết từng bước)

1) Kiểm tra phân bố TTF trước khi fine‑tune
- Mục đích: bảo đảm dải `train/val/test` theo TTF có đủ lớp.
- Lệnh (tuỳ chọn):
  - `python scripts/check_manifest_ttf.py configs/logs_stft_full_temporal.yaml 0 85 85 90 90 100.1`
- Kỳ vọng: Val (85–90) và Test (90–100.1) có đủ degrading/fault; nếu thiếu, cân nhắc điều chỉnh mốc.

2) Train stratified (ổn định, học đủ 3 lớp)
- Cấu hình: `configs/logs_stft_train_strat.yaml` (train/val/test = 0.6/0.2/0.2, min_per_class_*=3, early_stop: true, use_class_weights: true)
- Lệnh: `python train_logs.py --config configs/logs_stft_train_strat.yaml`
- Kết quả cần kiểm tra:
  - Checkpoint: `runs/logs_stft_strat/best.pt`
  - Báo cáo: `python eval_logs.py --config configs/logs_stft_train_strat.yaml --ckpt runs/logs_stft_strat/best.pt --show`
  - File: `runs/logs_stft_strat/eval/report.txt`, `confusion_matrix.png`

3) Fine‑tune temporal (TTF)
- Cấu hình: `configs/logs_stft_full_temporal.yaml`
  - `temporal_ttf: train [0,85], val [85,90], test [90,100.1]`
  - `train.init_from: runs/logs_stft_strat/best.pt`
  - `train.lr: 3e-5`, `epochs: 60`, `early_stop_patience: 12`
- Lệnh: `python train_logs.py --config configs/logs_stft_full_temporal.yaml`
- Kết quả cần kiểm tra:
  - Checkpoint: `runs/logs_stft_temporal/best.pt`
  - Đồ thị huấn luyện: `runs/logs_stft_temporal/train_loss_curve.png`, `val_metrics_curve.png`

4) Đánh giá temporal (đủ 3 lớp)
- Lệnh: `python eval_logs.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --show`
- Kết quả:
  - `runs/logs_stft_temporal/eval/report.txt` (luôn đủ 3 lớp)
  - `runs/logs_stft_temporal/eval/confusion_matrix.png`, `f1_per_class.png`
  - Theo mốc 70–90 và 90–100.1: `report_early_70_90.txt`, `report_late_90_100.txt`

5) Phân tích theo thời gian (time metrics)
- Lệnh: `python scripts/plot_time_metrics.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --max_windows 50 --out_subdir eval/time_metrics --show`
- Kết quả:
  - `runs/logs_stft_temporal/eval/time_metrics/time_metrics.csv|.png`
  - Phân bố lớp theo TTF: `class_dist_over_ttf.png`

## Lưu ý & Tối ưu thêm
- Nếu val vẫn khó hội tụ ở temporal, có thể:
  - Tăng `early_stop_patience` (15–20) hoặc tăng `epochs` nhẹ (80).
  - Điều chỉnh `temporal_ttf.val` để đảm bảo có đủ degrading/fault trong val tùy dữ liệu.
- Có thể bật `use_class_weights` ở stratified nếu lệch lớp mạnh.

## Tiêu chí hoàn thành
- Stratified: Macro‑F1 tốt và ổn (không sụp về 1 lớp).
- Temporal FT: cải thiện điểm ở các bin 70–95% so với trước; báo cáo/CM/F1 theo lớp đầy đủ.
