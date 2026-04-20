# Baseline Run Plan

Tổng hợp số liệu hiện có, lệnh cần chạy, và số liệu cần cập nhật vào paper.

---

## Trạng thái hiện tại (2026-04-21)

### Số liệu đã có — dùng được cho paper

| Model | Protocol | Acc | Macro-F1 | Degrading F1 | Fault F1 | Present Macro-F1† | Nguồn |
|---|---|---|---|---|---|---|---|
| SVM 8D | Stratified | 0.7241 | 0.6190 | 0.6667 | 0.3333 | — | `runs/classical/svm_vib8_stratified/report_test.txt` |
| SVM 8D | Temporal | 0.6923 | — | 0.8000 | **0.3750** | 0.5875 | `runs/classical/svm_vib8_strat_train_temporal_eval/report_test.txt` |
| **Multi-modal** | Stratified | 0.7241 | **0.7762** | 0.6364 | 1.0000 | — | `runs/logs_stft_strat/auto_r22/eval/report.txt` |
| **Multi-modal** | Temporal | **0.8974** | — | **0.9200** | **0.8889** | **0.9045** | `runs/logs_stft_temporal_gating_final/eval/report.txt` |

† Present-class Macro-F1 = Macro-F1 chỉ tính trên Degrading + Fault (Healthy vắng mặt trong temporal test)

### Số liệu CẦN CHẠY LẠI — vib-only CNN chưa có kết quả tin cậy

| Model | Protocol | Trạng thái | Vấn đề |
|---|---|---|---|
| Vib-only CNN | Stratified | ❌ Collapsed (27.6%) | `use_class_weights: false`, LR quá cao, train từ scratch |
| Vib-only CNN | Temporal | ⚠️ Phụ thuộc phase 1 | Phase 2 init từ collapsed checkpoint → kết quả không tin cậy |

---

## Lệnh chạy — thứ tự bắt buộc

### Bước 1 — Retrain vib-only stratified (đã sửa config)

Config đã fix: `use_class_weights: true`, `lr: 1e-4`, `epochs: 100`, `patience: 25`

```powershell
python train_logs.py --config configs/ablation/logs_stft_train_strat_vib.yaml
```

Output: `runs/ablations/strat_vib_only/best.pt`

### Bước 2 — Eval vib-only stratified

```powershell
python eval_logs.py --config configs/ablation/logs_stft_train_strat_vib.yaml --ckpt runs/ablations/strat_vib_only/best.pt
```

Kết quả tại: `runs/ablations/strat_vib_only/eval/report.txt`

### Bước 3 — Retrain vib-only temporal (fair config, sau khi Bước 1 xong)

Config: `logs_stft_temporal_vib_ft_fair.yaml` — cùng STFT params với multimodal temporal (2048/512/160×160), `init_from: strat_vib_only/best.pt`

```powershell
python train_logs.py --config configs/ablation/logs_stft_temporal_vib_ft_fair.yaml
```

Output: `runs/ablations/temporal_vib_only_fair/best.pt`

### Bước 4 — Eval vib-only temporal

```powershell
python eval_logs.py --config configs/ablation/logs_stft_temporal_vib_ft_fair.yaml --ckpt runs/ablations/temporal_vib_only_fair/best.pt
```

Kết quả tại: `runs/ablations/temporal_vib_only_fair/eval/report.txt`

---

## Bảng so sánh trong paper — cần cập nhật sau khi chạy xong

File: `main.tex` — `tab:baseline_comparison` (~line 314)

```
Cần thay hai dòng:
  Vib-only stratified: 0.2759 / 0.1441  →  <số mới từ bước 2>
  Vib-only temporal:   0.6667 / 0.4000  →  <số mới từ bước 4>
```

---

## Kiểm tra nhanh sau khi chạy xong

```powershell
# Đọc kết quả stratified
cat runs/ablations/strat_vib_only/eval/report.txt

# Đọc kết quả temporal (present-class)
cat runs/ablations/temporal_vib_only_fair/eval/report_present.txt
```

Kỳ vọng hợp lý:
- Vib-only stratified: Macro-F1 > 0.5 (tất cả 3 class đều được predict)
- Vib-only temporal Fault F1: vẫn thấp hơn multimodal (0.888) vì thiếu temperature — đây là điều bình thường và là điểm chứng minh đóng góp của temperature branch

---

## Ghi chú tính công bằng của so sánh

| Tiêu chí | SVM | Vib-only CNN | Multi-modal |
|---|---|---|---|
| Cùng `manifest.csv` | ✅ | ✅ | ✅ |
| Cùng temporal test `[70,100.1]%` | ✅ | ✅ | ✅ |
| Thấy đủ 3 class khi train | ✅ (stratified) | ✅ (phase 1) | ✅ (phase 1) |
| Cùng STFT params temporal | N/A | ✅ (2048/512/160) | ✅ (2048/512/160) |
| Cùng window/hop 1s/0.5s | ✅ | ✅ | ✅ |
| File-wise aggregation | ✅ | ✅ | ✅ |

**Lưu ý quan trọng cho paper**: SVM temporal được train trên stratified split (thấy đủ 3 class), KHÔNG phải train trên temporal train [0-70]% (vì temporal train data không có Fault). Điều này mirror với deep models cũng được pretrain trên stratified trước. Cần ghi chú rõ trong paper.
