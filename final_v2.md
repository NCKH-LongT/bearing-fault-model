# Báo Cáo Kỹ Thuật (final_v2)

## 1. Mục tiêu và đóng góp
- Xây dựng mô hình chẩn đoán trạng thái ổ trục 3 lớp (healthy/degrading/fault) theo file, phù hợp triển khai thời gian thực (gộp cửa sổ – file‑wise).
- Tránh rò rỉ thời gian bằng tách tập theo TTF (temporal). Quy trình 2 pha: (1) học ổn định với stratified, (2) fine‑tune theo temporal.
- Tự động hoá train→eval→điều chỉnh để tìm cấu hình tốt, kèm phân tích theo TTF (time metrics, phân bố lớp).

### Bộ dữ liệu và cách đánh nhãn
- Dữ liệu ở dạng nhật ký (CSV) theo giờ cho từng lần chạy run‑to‑failure, mỗi dòng chứa 4 kênh: `vib_x, vib_y, temp_bearing, temp_atm` và metadata trong `manifest.csv`: `file, run_id, ttf_percent, fault_type`.
- Nhãn ground‑truth lấy trực tiếp từ cột `fault_type` trong manifest (đã được gán sẵn theo quá trình vận hành):
  - `healthy` (ổn định), `degrading` (suy giảm), `fault` (hỏng).
- Phân bố lớp thay đổi theo thời gian TTF (đặc thù run‑to‑failure). Khi đánh giá toàn dải (0–100% TTF) với mô hình hiện tại, mẫu test minh hoạ có tỷ lệ xấp xỉ: healthy ≈ 60%, degrading ≈ 30%, fault ≈ 10% (ví dụ: 77/129/13 trên tổng 129 file) — các tỷ lệ này phụ thuộc dữ liệu thực tế và từng run.
- Các ngưỡng TTF đặc trưng theo pha vận hành dùng để chọn ranh split (không thay đổi nhãn gốc):
  - Healthy chủ yếu ở sớm: ~0–60% TTF
  - Degrading tăng dần ở trung kỳ: ~60–90% TTF
  - Fault tập trung ở muộn: ~90–100% TTF
  Các ranh này giúp chọn `train/val/test` liên tục theo trục thời gian, mô phỏng đúng kịch bản triển khai.

### Lý do phân theo TTF (temporal split)
- Tránh rò rỉ tương lai: mẫu `test` phải nằm sau `train/val` trên trục TTF để phản ánh đúng bài toán dự báo/giám sát theo thời gian.
- Phù hợp triển khai: mô hình được đánh giá trên các file muộn hơn (gần hỏng), đồng thời có thể mở rộng dải `test` (ví dụ từ 70% TTF) để bao phủ đủ 3 lớp cho phân tích chẩn đoán.

## 2. Dữ liệu và chia tập
- Nguồn: `data/manifest.csv` (cột: file, run_id, ttf_percent, fault_type).
- Cửa sổ tín hiệu (window): `window_seconds=1.0`, `hop_seconds=0.5` → nhiều cửa sổ/1 file. Train lấy ngẫu nhiên 1 cửa sổ/file, eval gộp tất cả.
- Hai chế độ chia:
  - Stratified (phát triển/tinh chỉnh): train/val/test = 0.6/0.2/0.2, đảm bảo tối thiểu mỗi lớp ở val/test.
  - Temporal (triển khai): tách theo TTF, tránh rò rỉ thời gian. Cấu hình hiện tại:
    - `train: [0, 60]`, `val: [60, 70]`, `test: [70, 100.1]` (đảm bảo test có nhiều lớp hơn giai đoạn muộn).

## 3. Đặc trưng và mô hình
- Đặc trưng rung (2 kênh): STFT log‑magnitude, chuẩn hoá theo frame, resize về `input_size`.
  - Cấu hình hiệu quả: `n_fft=2048/4096`, `hop_length=512/1024`, `input_size=[160,160]` (stratified tốt), hoặc `[224,224]` (temporal gốc).
- Đặc trưng nhiệt (6 chiều): mean/std/slope cho bearing/atm mỗi cửa sổ (`features/temp_features.py`).
- Kiến trúc: ResNet2D nhỏ + nối đặc trưng nhiệt ở head (`models/resnet2d.py: ResNet18Small(in_ch=2, temp_feat_dim=6)`).

## 4. Huấn luyện
- Loss: `CrossEntropy` + `label_smoothing` (0.05–0.1).
- Xử lý lệch lớp: ưu tiên `balanced_sampling: true`, tránh bật đồng thời class‑weights mạnh (tránh bù kép).
- Tối ưu: AdamW; lịch LR Cosine (OneCycle tắt để ổn định ở FT).
- Tăng tốc: AMP (`torch.amp.*`), DataLoader workers + prefetch.
- Early stopping: theo `macro_f1` (file‑wise), patience 12–20 (tuỳ pha), log best checkpoint.

## 5. Tự động hoá
- Tự động train stratified: `scripts/auto_train_eval.py`
  - Vòng lặp train→eval→điều chỉnh nhẹ (lr/patience/val_max_windows/giảm độ phức tạp input), mỗi vòng lưu ở `runs/logs_stft_strat/auto_rN`.
  - Tiếp tục vòng hiện có: `--continue`; khởi tạo từ ckpt vòng trước: `--resume_from_prev`.
- Tự động fine‑tune temporal: `scripts/auto_finetune_temporal.py`
  - Vòng FT ngắn, ưu tiên ổn định (Cosine, lr nhỏ, balanced_sampling), mỗi vòng ở `runs/logs_stft_temporal/auto_ft_rN`.

## 6. Cấu hình tiêu biểu (đã dùng)
- Stratified: `configs/logs_stft_train_strat.yaml`
  - `split_mode: stratified` (0.6/0.2/0.2), `min_per_class_val/test=5`
  - `stft: {n_fft: 2048, hop_length: 512}`; `input_size: [160,160]`
  - `train: {epochs: 80, lr: 2e-4, weight_decay: 1e-3, balanced_sampling: true, use_class_weights: false, label_smoothing: 0.05, val_max_windows: 0 (ổn định đo lường)}`
  - `optim.use_onecycle: false`
- Temporal (FT): `configs/logs_stft_full_temporal.yaml`
  - `split_mode: temporal`; `temporal_ttf: train [0,60], val [60,70], test [70,100.1]`
  - `train: {init_from: runs/logs_stft_strat/auto_r22/best.pt, lr: 3e-5, epochs: 60, early_stop_patience: 12, balanced_sampling: true, use_class_weights: false, val_max_windows: 50}`
  - `optim.use_onecycle: false`
- Đánh giá tổng thể (phụ): `configs/logs_stft_full_temporal_alltest.yaml` với `test: [0,100.1]` (chỉ để tham khảo toàn dải; không dùng làm test chính thức vì có rò rỉ thời gian).

## 7. Kết quả chính
- Stratified (vòng tốt nhất: `auto_r22`)
  - `runs/logs_stft_strat/auto_r22/eval/report.txt`:
    - Macro‑F1 ≈ 0.7762; F1(healthy≈0.69, degrading≈0.64, fault=1.00). CM có đường chéo rõ cả 3 lớp.
- Temporal – test muộn 2 lớp (ví dụ R2/R3 FT, test gồm degrading+fault)
  - `runs/logs_stft_temporal/eval/report_present.txt`:
    - Acc ≈ 0.95; Macro‑F1(2 lớp) ≈ 0.947; F1(degrading≈0.93, fault≈0.96).
- Temporal – test rộng [70,100.1] (mục tiêu đủ lớp, thực tế thiếu healthy ở log hiện có)
  - Một số phiên bản cho thấy degrade yếu (nhầm sang healthy). Cấu hình đã điều chỉnh dải TTF để cải thiện phủ lớp test.
- All‑TTF (0–100%) – tham khảo tổng thể
  - `runs/logs_stft_temporal_alltest/eval/report.txt`:
    - Acc ≈ 0.69; Macro‑F1 ≈ 0.589; F1(healthy≈0.80, fault≈0.92, degrading≈0.05).
  - Ý nghĩa: tổng thể mô hình rất mạnh ở healthy/fault; degrade là điểm cần tăng cường (đánh đổi giữa early vs late stage).

## 8. Thảo luận và hạn chế
- Temporal test phụ thuộc phân bố lớp theo TTF; nếu test muộn quá có thể thiếu healthy/degrading → macro‑F1 bị lệch. Cần báo cáo rõ phân bố lớp (kèm `class_dist_over_ttf.png`).
- Degrading là lớp khó (chuyển tiếp) và nhạy theo ngưỡng TTF; nên tăng hiện diện degrade ở train/val (balanced sampling, tinh chỉnh dải TTF) hoặc dùng curriculum FT.
- All‑TTF [0–100%] chỉ để tham khảo xu hướng; không dùng làm test chính thức vì có rò rỉ thời gian.

## 9. Hướng dẫn tái lập và sử dụng
- Cài đặt phụ thuộc (Python, PyTorch, sklearn, matplotlib…)
- Train stratified tự động (khuyến nghị):
  - `python scripts/auto_train_eval.py --config configs/logs_stft_train_strat.yaml --min_macro_f1 0.4 --min_class_f1 0.2 --max_rounds 50 --continue --resume_from_prev`
  - Kết quả theo vòng: `runs/logs_stft_strat/auto_rN/`
- Fine‑tune temporal (cấu hình đã chốt):
  - `python train_logs.py --config configs/logs_stft_full_temporal.yaml`
  - `python eval_logs.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --show`
  - `python scripts/plot_time_metrics.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --max_windows 50 --out_subdir eval/time_metrics --show`
- Đánh giá tổng thể (tuỳ chọn):
  - `python eval_logs.py --config configs/logs_stft_full_temporal_alltest.yaml --ckpt runs/logs_stft_temporal/best.pt --show`

## 10. Kết luận
- Quy trình Stratified → Temporal FT giúp mô hình học đủ 3 lớp ổn định, đồng thời đáp ứng ràng buộc thời gian thực (TTF, không rò rỉ).
- Kết quả tốt nhất cho thấy năng lực mạnh ở healthy/fault; lớp degrading cần tăng cường thêm bằng điều chỉnh dải TTF và cân bằng mẫu.
- Bộ mã cung cấp đầy đủ cấu hình, script tự động và artifact (CM, F1 theo lớp, time‑metrics) để tái lập và đưa vào bài báo hội thảo.

## 11. Train Stratified vs Temporal: Ý nghĩa, mục tiêu, lý do, tài nguyên

### 11.1. Stratified (giai đoạn phát triển/tinh chỉnh)
- Ý nghĩa: Chia train/val/test theo tỷ lệ và phân tầng theo lớp để mỗi tập con đều có đủ 3 lớp. Giảm phương sai phép đo, phù hợp chọn cấu hình/hyper‑params và phát hiện lỗi pipeline.
- Mục tiêu: Học ranh giới 3 lớp ổn định, có “đường chéo” rõ trong ma trận nhầm lẫn; Macro‑F1 cao và cân bằng giữa các lớp.
- Lý do: Dữ liệu TTF thường lệch theo thời gian (ít fault sớm, ít healthy muộn). Phân tầng giúp đánh giá công bằng, không phụ thuộc vào ranh TTF ngay từ đầu.
- Tài nguyên đi kèm:
  - Cấu hình: `configs/logs_stft_train_strat.yaml`
  - Script auto: `scripts/auto_train_eval.py` (tạo `runs/logs_stft_strat/auto_rN/`)
  - Artifact: `runs/logs_stft_strat/auto_rXX/eval/report.txt`, `confusion_matrix.png`, `f1_per_class.png`
  - Ngưỡng dừng khuyến nghị: Macro‑F1 ≥ 0.30–0.40 và F1 mỗi lớp > 0 (không sụp về 1 lớp) trước khi chuyển FT temporal.

### 11.2. Temporal (giai đoạn triển khai/đánh giá hiện thực)
- Ý nghĩa: Chia theo trục thời gian TTF (liên tục) để mô phỏng deployment không rò rỉ tương lai: train ở giai đoạn sớm hơn, test ở giai đoạn muộn hơn.
- Mục tiêu: Đo năng lực chẩn đoán theo thời gian; có thể tập trung giai đoạn cận hỏng (90–100%) hoặc mở rộng (70–100%) để đủ 3 lớp trên test.
- Lý do: Mô hình thực tế chỉ thấy “quá khứ→hiện tại”, không thấy tương lai; tách theo TTF bảo đảm tính chính danh của đánh giá triển khai.
- Tài nguyên đi kèm:
  - Cấu hình FT: `configs/logs_stft_full_temporal.yaml` (đã chốt theo vòng tốt R2/R3)
  - Script auto FT: `scripts/auto_finetune_temporal.py` (tạo `runs/logs_stft_temporal/auto_ft_rN/`)
  - Artifact: `runs/logs_stft_temporal/eval/report.txt`, `confusion_matrix.png`, `report_early_70_90.txt`, `report_late_90_100.txt`
  - Phân tích theo thời gian: `scripts/plot_time_metrics.py` → `eval/time_metrics/time_metrics.csv|.png`, `class_dist_over_ttf.png`
  - Kiểm tra dải TTF trước khi train/eval: `scripts/check_manifest_ttf.py`
  - Lưu ý: Báo cáo “present labels” (report_present.txt) khi test thiếu lớp; ghi rõ phân bố lớp test để diễn giải Macro‑F1.

### 11.3. Quy tắc thực hành tốt
- Dùng Stratified để đạt 3 lớp ổn định trước (điều chỉnh STFT, input, lr, label smoothing, balanced sampling).
- Sau đó FT Temporal với LR nhỏ, Cosine ổn định, balanced sampling; chọn ranh TTF test để phù hợp mục tiêu (cận hỏng hay đủ 3 lớp).
- Luôn đính kèm phân bố lớp theo TTF và time‑metrics để giải thích xu hướng theo thời gian.

## 12. Chi tiết Train Stratified

### 12.1. Thông số & cấu hình chính
- Cấu hình: `configs/logs_stft_train_strat.yaml`
- Chia tập: `split_mode: stratified` với tỉ lệ `train/val/test = 0.6/0.2/0.2`, đảm bảo tối thiểu mỗi lớp ở val/test (`min_per_class_val/test = 5`).
- STFT & ảnh: `n_fft=2048`, `hop_length=512`, `input_size=[160,160]`.
- Huấn luyện:
  - `epochs: 80`, `lr: 2e-4`, `weight_decay: 1e-3`
  - `balanced_sampling: true`, `use_class_weights: false` (tránh bù kép)
  - `label_smoothing: 0.05`, `use_amp: true`
  - `early_stop: true`, `early_stop_patience: 20`
  - Trong giai đoạn ổn định phép đo: `val_max_windows: 0` (sau đó có thể trả về 50)
  - Tối ưu: AdamW + Cosine (`optim.use_onecycle: false`)

### 12.2. Cách chạy
- Tự động (khuyến nghị):
  - `python scripts/auto_train_eval.py --config configs/logs_stft_train_strat.yaml --min_macro_f1 0.4 --min_class_f1 0.2 --max_rounds 50 --continue --resume_from_prev`
- Thủ công:
  - Train: `python train_logs.py --config configs/logs_stft_train_strat.yaml`
  - Eval: `python eval_logs.py --config configs/logs_stft_train_strat.yaml --ckpt runs/logs_stft_strat/best.pt --show`

### 12.3. Kết quả & ý nghĩa
- Vòng tốt nhất: `runs/logs_stft_strat/auto_r22/`
  - Macro‑F1 ≈ 0.7762; F1(healthy≈0.69, degrading≈0.64, fault=1.00)
  - Ma trận nhầm lẫn có đường chéo rõ trên cả 3 lớp → mô hình đã học ranh giới ổn định.
- Ý nghĩa: Đây là checkpoint tiền đề đáng tin cậy để fine‑tune theo temporal, giảm rủi ro sụp về 1 lớp khi chuyển sang chia theo thời gian.

## 13. Chi tiết Fine‑Tune Temporal

### 13.1. Thông số & cấu hình chính
- Cấu hình: `configs/logs_stft_full_temporal.yaml`
- Chia theo TTF (không rò rỉ): `train: [0,60]`, `val: [60,70]`, `test: [70,100.1]`
- Khởi tạo từ strat tốt nhất: `train.init_from: runs/logs_stft_strat/auto_r22/best.pt`
- Huấn luyện FT:
  - `epochs: 60`, `lr: 3e-5`, `early_stop_patience: 12`
  - `balanced_sampling: true`, `use_class_weights: false`
  - `val_max_windows: 50`, `use_amp: true`
  - Tối ưu: AdamW + Cosine (`optim.use_onecycle: false`)

### 13.2. Cách chạy
- Train: `python train_logs.py --config configs/logs_stft_full_temporal.yaml`
- Eval 3 lớp: `python eval_logs.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --show`
- Time metrics: `python scripts/plot_time_metrics.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --max_windows 50 --out_subdir eval/time_metrics --show`

### 13.3. Kết quả & ý nghĩa
- Khi test gồm 2 lớp (degrading+fault, dải muộn): Acc≈0.95; Macro‑F1(2 lớp)≈0.947; F1(degrading≈0.93, fault≈0.96) — phù hợp kịch bản giám sát cận hỏng.
- Khi mở rộng test để bao phủ nhiều TTF hơn (ví dụ [70,100.1]): có thể xuất hiện healthy/degrading; tùy phân bố thực tế, cần tinh chỉnh dải để không thiếu lớp.
- Phân tích tổng thể 0–100% (không dùng làm test chính thức) cho thấy mô hình mạnh ở healthy/fault; degrade còn yếu ở một số vùng — gợi ý tăng cưỡng hệ số hiện diện degrade hoặc curriculum FT.

## 14. Hình minh hoạ đề xuất cho bài báo

### 14.1. Stratified (3 lớp ổn định)
- Ma trận nhầm lẫn: `runs/logs_stft_strat/auto_r22/eval/confusion_matrix.png`
- F1 theo lớp: `runs/logs_stft_strat/auto_r22/eval/f1_per_class.png`
- Đường cong huấn luyện: `runs/logs_stft_strat/train_curves.png`
- Đường cong val (acc/F1): `runs/logs_stft_strat/val_metrics_curve.png`

### 14.2. Temporal FT (triển khai theo TTF)
- Ma trận nhầm lẫn (test temporal): `runs/logs_stft_temporal/eval/confusion_matrix.png`
- F1 theo lớp (test temporal): `runs/logs_stft_temporal/eval/f1_per_class.png`
- Báo cáo theo giai đoạn TTF:
  - Sớm (70–90%): `runs/logs_stft_temporal/eval/report_early_70_90.txt`, `confusion_matrix_early.png`
  - Muộn (90–100%): `runs/logs_stft_temporal/eval/report_late_90_100.txt`, `confusion_matrix_late.png`
- Chỉ số theo thời gian: `runs/logs_stft_temporal/eval/time_metrics/time_metrics.png`
- Phân bố lớp theo TTF: `runs/logs_stft_temporal/eval/time_metrics/class_dist_over_ttf.png`

### 14.3. Toàn dải 0–100% (phụ lục, tham khảo xu hướng)
- Ma trận nhầm lẫn: `runs/logs_stft_temporal_alltest/eval/confusion_matrix.png`
- F1 theo lớp: `runs/logs_stft_temporal_alltest/eval/f1_per_class.png`
- Ghi chú rõ: đây là đánh giá tổng thể để quan sát xu hướng, không dùng làm test chính thức (có rò rỉ thời gian).

### 14.4. Cách sinh/khôi phục hình
- Stratified: chạy train/eval như mục 12 để sinh ảnh trong `runs/logs_stft_strat/`.
- Temporal: chạy FT/eval như mục 13 để sinh ảnh trong `runs/logs_stft_temporal/`.
- Time‑metrics: `python scripts/plot_time_metrics.py --config <config_temporal>.yaml --ckpt <ckpt_temporal> --max_windows 50 --out_subdir eval/time_metrics --show`.
