# Tổng Kết Quy Trình Và Kết Quả (final)

## Mục tiêu
- Huấn luyện mô hình chẩn đoán 3 lớp (healthy/degrading/fault) theo file, đánh giá không rò rỉ theo thời gian (TTF).
- Ổn định học bằng split stratified, sau đó fine‑tune theo temporal (TTF).
- Tự động hoá train→eval→điều chỉnh cấu hình cho đến khi đạt ngưỡng tốt.

## Các thay đổi chính trong mã nguồn
- Early stop + cấu hình temporal
  - Bật/tinh chỉnh early stop (`early_stop: true`, `early_stop_patience` hợp lý) và cập nhật `temporal_ttf: train [0,85], val [85,90], test [90,100.1]`.
  - Đảm bảo eval xuất đủ 3 lớp: `eval_logs.py` luôn ghi `report.txt` với full 3 lớp, thêm `report_present.txt` cho lớp xuất hiện.
- Ổn định/nhanh hoá huấn luyện
  - AMP (mixed precision) với API mới (`torch.amp.*`), tránh FutureWarning; chỉ `scheduler.step()` khi `optimizer.step()` đã thực hiện.
  - Giới hạn số cửa sổ khi val bằng `train.val_max_windows`; bản sửa để `0` = dùng tất cả cửa sổ.
  - Thêm `label_smoothing` cho CrossEntropy; chuẩn hoá class weights (nếu dùng) về mean=1 và kẹp [0.5,5].
  - Balanced sampling (WeightedRandomSampler) thay cho class weights khi cần, tránh bù kép.
- Tự động hoá train→eval→điều chỉnh
  - Thêm `scripts/auto_train_eval.py`:
    - Chạy train→eval theo vòng (round), đọc `eval/report.txt` để lấy Macro‑F1 và F1 từng lớp.
    - Điều chỉnh cấu hình nhẹ theo vòng (lr/patience/val_max_windows/độ phức tạp đầu vào).
    - Mỗi vòng có thư mục riêng `.../auto_rN` (config.yaml, best.pt, train_log.csv, ảnh eval/train).
    - Hỗ trợ `--continue` (tiếp tục từ round hiện có) và `--resume_from_prev` (fine‑tune từ best.pt vòng trước).
- Hướng mở rộng
  - Viết `docs/survival_plan.md` mô tả lộ trình chuyển sang Survival (DeepSurv/DeepHit) nếu cần dự báo theo thời gian sống còn.

## Pipeline vận hành
1) Train stratified để ổn định 3 lớp
- Cấu hình cơ sở: `configs/logs_stft_train_strat.yaml` (đã bật AMP, label_smoothing, balanced_sampling, OneCycle tắt để ổn định).
- Tự động hoá (khuyến nghị):
  - `python scripts/auto_train_eval.py --config configs/logs_stft_train_strat.yaml --min_macro_f1 0.4 --min_class_f1 0.2 --max_rounds 50 --continue --resume_from_prev`
  - Script tạo các vòng `runs/logs_stft_strat/auto_rN` cho đến khi đạt ngưỡng.

2) Fine‑tune temporal (TTF)
- Cấu hình: `configs/logs_stft_full_temporal.yaml` (đã `train.init_from` trỏ tới checkpoint tốt nhất từ stratified).
- Lệnh:
  - Train: `python train_logs.py --config configs/logs_stft_full_temporal.yaml`
  - Eval 3 lớp: `python eval_logs.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --show`
  - Time metrics: `python scripts/plot_time_metrics.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --max_windows 50 --out_subdir eval/time_metrics --show`

## Cấu hình & mẹo đã chứng minh hiệu quả
- Stratified
  - Balanced sampling: `train.balanced_sampling: true`, tắt `use_class_weights` để tránh bù kép.
  - Giảm độ phức tạp đầu vào: `stft.n_fft: 2048`, `stft.hop_length: 512`, `input_size: [160,160]` giúp hội tụ tốt.
  - LR ổn định: `lr ≈ 2e-4`, `weight_decay ≈ 1e-3` (Cosine, OneCycle tắt), `label_smoothing: 0.05`.
  - `val_max_windows: 0` giai đoạn ổn định đo lường; về sau có thể đặt `50` để tăng tốc.
- Temporal (FT)
  - Dùng `init_from` checkpoint tốt nhất của stratified (đã set R22), LR nhỏ hơn (ví dụ `3e-5`) và patience hợp lý.

## Kết quả nổi bật
- Sau các vòng auto ở stratified:
  - Vòng 22 (auto_r22) đạt Macro‑F1 ≈ 0.7762; F1 từng lớp: healthy ≈ 0.69, degrading ≈ 0.64, fault = 1.00.
  - Ma trận nhầm lẫn cho thấy đường chéo rõ ở cả 3 lớp (không còn sụp về 1 lớp).
- Đã cập nhật `configs/logs_stft_full_temporal.yaml` để FT từ `runs/logs_stft_strat/auto_r22/best.pt`.

## Bài học kinh nghiệm
- Không nên bật đồng thời balanced sampling và class weights mạnh → dễ “bù kép” lệch một lớp.
- `val_max_windows: 0` giúp phép đo val ổn định khi còn nhiễu; khi tin cậy có thể tăng lại để tiết kiệm thời gian.
- Giảm độ phức tạp đặc trưng (n_fft, input_size) giúp mô hình học ranh giới sớm hơn; có thể tăng lại khi đã ổn định.
- Fine‑tune nối tiếp (init_from) theo từng vòng cho tiến triển đều và tiết kiệm thời gian.

## Tái lập nhanh
- Train stratified tự động:
  - `python scripts/auto_train_eval.py --config configs/logs_stft_train_strat.yaml --min_macro_f1 0.4 --min_class_f1 0.2 --max_rounds 50 --continue --resume_from_prev`
- Fine‑tune temporal + eval:
  - `python train_logs.py --config configs/logs_stft_full_temporal.yaml`
  - `python eval_logs.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --show`
  - `python scripts/plot_time_metrics.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --max_windows 50 --out_subdir eval/time_metrics --show`
