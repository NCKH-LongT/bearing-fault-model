# Hướng Dẫn Ablation (Tiếng Việt)

Tài liệu này gom các thí nghiệm ablation vào các cấu hình riêng và lệnh chạy đơn giản để bạn tái lập số liệu cho bài báo mà không đụng đến các config/code gốc.

## Mục Tiêu Thử Nghiệm

- Lợi ích quy trình 2 pha: fine‑tune theo thời gian (temporal FT) có vs. không khởi tạo từ mô hình stratified (`init_from`).
- Ảnh hưởng lát cắt thời gian (temporal slice) của tập test: [70,100.1]% vs. [90,100.1]% TTF.
- (Tuỳ chọn) Cách gộp file‑wise: trung bình logit (mean pre‑softmax) vs. bỏ phiếu đa số (majority vote).

## Bố Cục Thư Mục

- Config dành cho ablation nằm tại `configs/ablation/` và ghi kết quả vào `runs/ablations/...`.
- Các config/run gốc giữ nguyên, không bị ghi đè.

## A. FT Từ Đầu (Không `init_from`) so với Có `init_from`

Config temporal FT gốc (có `init_from`): `configs/logs_stft_full_temporal.yaml`.

Thiết lập ablation (từ đầu): dùng các file YAML đã chuẩn bị sẵn (bỏ `train.init_from`) và ghi ra thư mục riêng.

Ví dụ các seed đã tạo sẵn:

- `configs/ablation/temporal_ft_from_scratch_s42.yaml`
- `configs/ablation/temporal_ft_from_scratch_s43.yaml` 
- `configs/ablation/temporal_ft_from_scratch_s44.yaml`

Chạy train + eval cho từng seed:

- Train: `python train_logs.py --config configs/ablation/temporal_ft_from_scratch_s42.yaml`
- Eval: `python eval_logs.py  --config configs/ablation/temporal_ft_from_scratch_s42.yaml --ckpt runs/ablations/temporal_scratch_s42/best.pt`

Thu thập `runs/ablations/temporal_scratch_sXX/eval/report.txt`, tính Macro‑F1 trung bình ± độ lệch chuẩn (mean±std). So sánh với thiết lập temporal FT có `init_from` trong `configs/logs_stft_full_temporal.yaml`.

Gợi ý: lặp lại 3–5 seed (nhân bản YAML, chỉnh `train.seed` và `log.out_dir`).

## B. Lát Cắt Test Theo Thời Gian: [70,100.1]% vs. [90,100.1]%

Mục đích: cho thấy protocol (sự hiện diện lớp trong lát cắt) ảnh hưởng mạnh đến kết quả.

Configs:

- Late rộng (giữ test `[70.0,100.1]`, tách thư mục): `configs/ablation/temporal_ft_broad_test.yaml`.
- Late đời cuối (2 lớp): `configs/ablation/temporal_ft_late_test.yaml` (test `[90.0,100.1]`).

Dùng cùng một checkpoint temporal FT (đã khởi tạo từ stratified) để tránh phải train lại:

- Broad: `python eval_logs.py --config configs/ablation/temporal_ft_broad_test.yaml --ckpt runs/logs_stft_temporal/best.pt`
- Late: `python eval_logs.py --config configs/ablation/temporal_ft_late_test.yaml  --ckpt runs/logs_stft_temporal/best.pt`

So sánh `eval/report.txt`, `eval/report_present.txt` và các hình ma trận nhầm lẫn. Ghi chú số lớp hiện diện và khác biệt Macro‑F1.

## C. (Tuỳ Chọn) Gộp File‑wise: mean‑logit vs. majority‑vote

Đã bổ sung tham số cho `eval_logs.py` để chọn cách gộp:

- `--agg mean` (mặc định): trung bình logit trước softmax (đúng với bài báo)
- `--agg vote`: bỏ phiếu đa số theo dự đoán từng cửa sổ

Ví dụ:

- `python eval_logs.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --agg mean`
- `python eval_logs.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --agg vote`

Ghi nhận thay đổi Macro‑F1 trên cùng checkpoint để cô lập ảnh hưởng do cách gộp.

## Checklist Báo Cáo (đưa vào main.tex)

- Với A: bảng Macro‑F1 (mean±std) theo seed, Δ so với cấu hình có `init_from`, và p‑value (paired t‑test). Nhận xét độ ổn định.
- Với B: bảng so lát cắt test, nêu rõ số lớp hiện diện; đính kèm hình ma trận nhầm lẫn để đưa vào vị trí figure trong `main.tex`.
- Với C: một hàng ngắn thể hiện Δ Macro‑F1 giữa hai cách gộp trên cùng lát cắt.

## Ghi Chú

- Tất cả kết quả ablation nằm dưới `runs/ablations/...`, tách biệt với run gốc.
- Muốn thêm seed: sao chép một YAML “scratch”, chỉnh `train.seed` và `log.out_dir`.
- Không thay đổi config gốc; chỉ thêm tham số `--agg` vào `eval_logs.py` (tương thích ngược, mặc định vẫn là `mean`).
