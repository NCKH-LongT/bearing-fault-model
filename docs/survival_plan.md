# Kế hoạch triển khai Survival (DeepSurv/DeepHit)

## Mục tiêu
- Mô hình hóa nguy cơ hỏng theo thời gian (không rò rỉ tương lai), phù hợp dữ liệu run‑to‑failure.
- Bắt đầu bằng DeepSurv (Cox) vì gọn, ổn định; sau đó mở rộng DeepHit (thời gian rời rạc) nếu cần.

## Thiết kế nhãn (survival labeling)
- Mỗi file (một quan sát) có cặp nhãn `(time, event)`:
  - `time`: dùng `ttf_percent` (0–100) hoặc đổi ra giờ nếu có.
  - `event`: 1 nếu là file cuối của `run_id` (thời điểm hỏng), 0 nếu trước đó (censored).
- Gom theo `run_id` để xác định file cuối (max `ttf_percent`).

## Thay đổi code (tối thiểu, tuần tự)
1) Dataset: `datasets/logs_ttf.py`
- Thêm tham số `label_source: "fault"|"survival"` (mặc định `fault`).
- Khi build index: gán thêm `time` và `event` theo `run_id`.
- `__getitem__`: nếu `label_source=="survival"` trả về `x, tfeat, y_surv=[time, event]`.

2) Model: `models/resnet2d.py`
- Thêm head survival: `self.risk = nn.Linear(256+32, 1)`.
- Trong `forward`: nếu `task=="survival"` trả về `risk` (B×1); nếu `classify` giữ logits cũ.

3) Train: `train_logs.py`
- Đọc `cfg["task"]`. Nếu `survival`:
  - Loss: Cox partial negative log‑likelihood (`cox_loss(risk, time, event)`).
  - Metric val: Concordance index (C‑index) file‑wise (gộp mean risk trên các cửa sổ của 1 file).
  - Lưu `best.pt` theo `c_index` (đặt `log.save_best_by: c_index`).
- AMP, early‑stop, balanced sampling vẫn dùng như hiện tại.

4) Eval: tạo script `eval_surv.py`
- Tải model + dataset (survival), xuất:
  - `c_index.txt` (C‑index tổng và theo split).
  - `scatter_time_vs_risk.png`, `scatter.csv` (cột: file, run_id, time, event, risk).
  - (Tùy chọn) C‑index theo các bin TTF: `time_metrics_surv.csv|.png`.

## Cấu hình YAML mẫu
- Stratified (dev) — `configs/surv_strat.yaml`
```
task: survival
survival:
  mode: cox
  time_unit: percent
split_mode: stratified
stratified: { train: 0.6, val: 0.2, test: 0.2 }
stft: { n_fft: 4096, hop_length: 1024, window: hann, log_add: 1.0 }
input_size: [224, 224]
train:
  batch_size: 24
  lr: 3.0e-4
  epochs: 120
  early_stop: true
  early_stop_patience: 15
  use_amp: true
  balanced_sampling: true
optim: { use_onecycle: false }
log: { out_dir: runs/surv_strat, save_best_by: c_index }
```

- Temporal FT — `configs/surv_temporal.yaml`
```
task: survival
survival: { mode: cox, time_unit: percent }
split_mode: temporal
temporal_ttf:
  train: [0.0, 85.0]
  val:   [85.0, 90.0]
  test:  [90.0, 100.1]
train:
  init_from: runs/surv_strat/best.pt
  lr: 1.0e-4
  epochs: 60
  early_stop: true
  early_stop_patience: 12
  use_amp: true
log: { out_dir: runs/surv_temporal, save_best_by: c_index }
```

## Quy trình chạy
1) Train stratified (DeepSurv)
- `python train_logs.py --config configs/surv_strat.yaml`
- Kiểm tra: `runs/surv_strat/train_log.csv` (cột `val_c_index`), `best.pt`.

2) Fine‑tune temporal
- `python train_logs.py --config configs/surv_temporal.yaml`

3) Đánh giá
- `python eval_surv.py --config configs/surv_temporal.yaml --ckpt runs/surv_temporal/best.pt --show`
- Xem `c_index.txt`, `scatter_time_vs_risk.png`.

## Pros/Cons của survival (DeepSurv → DeepHit)
Pros
- Xử lý đúng bản chất dữ liệu run‑to‑failure có censoring (file chưa hỏng).
- C‑index phản ánh đúng thứ hạng rủi ro theo thời gian, tránh rò rỉ tương lai.
- Dễ FT temporal, giữ nguyên quy trình gộp theo cửa sổ (file‑wise).

Cons
- Không đưa ra thời gian còn lại tuyệt đối; cho thứ hạng/nguy cơ (DeepSurv). Muốn thời gian cần DeepHit/hồi quy.
- Cần xây dựng loss Cox chính xác (risk set, logsumexp), batch đủ lớn để ổn định.
- Đánh giá phức tạp hơn (C‑index, time‑dependent metrics).

## Tiêu chí hoàn thành
- Stratified: `val_c_index` > 0.6 và tăng ổn định theo epoch; scatter risk tăng theo `time`.
- Temporal: `c_index` giữ ổn định; xu hướng risk tăng theo TTF; không rò rỉ thời gian.

## Gợi ý mở rộng (giai đoạn 2: DeepHit)
- Thêm `survival.mode: deephit` và `ttf_bins: [...]` (ví dụ các mốc trong time_metrics).
- Head K chiều (softmax theo thời gian), loss DeepHit (likelihood + ranking), metric C‑index rời rạc, Brier/IBS theo thời gian.

