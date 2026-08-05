# Bearing paper: trạng thái, tái lập và hướng nâng cấp

Đây là tài liệu canonical cho revision. Các guide cũ đã được xóa để tránh chạy nhầm protocol lịch sử. Checklist chi tiết và nhật ký nằm trong `REVISION_RUN_CHECKLIST.md`; audit split nằm trong `revision_artifacts/protocol_audit.md`.

## 1. Kết quả đã khóa

Protocol revision dùng file-wise stratified split cố định bằng seed `20260803`: 76 train, 26 validation và 27 test file. Test gồm 16 Healthy, 8 Degrading và 3 Fault; không giao file giữa các partition. Hyperparameter chỉ được chọn bằng validation, sau đó test được mở một lần cho five-seed confirmation.

| Model | Accuracy | Macro-F1 | Healthy F1 | Degrading F1 | Fault F1 |
|---|---:|---:|---:|---:|---:|
| SVM vibration 8-D | 0.8148 | 0.7681 | 0.8889 | 0.6154 | 0.8000 |
| Vibration-only CNN | 0.4593 ± 0.2437 | 0.3314 ± 0.2643 | 0.4118 ± 0.3858 | 0.3425 ± 0.3204 | 0.2400 ± 0.4336 |
| Multimodal late fusion | 0.5556 ± 0.1889 | 0.4544 ± 0.1138 | 0.4932 ± 0.4512 | 0.3500 ± 0.2049 | 0.5200 ± 0.3271 |

SVM là model mạnh nhất trên test hiện tại. Hai deep model không ổn định qua seed và đôi khi bỏ hẳn một lớp. Không tune thêm dựa trên 27 test file này.

Bootstrap 10.000 lần ở cấp file cho kết quả:

| Model | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---:|---:|
| SVM vibration 8-D | 0.8148 [0.6667, 0.9630] | 0.7681 [0.4372, 0.9375] |
| Multimodal late fusion | 0.5556 [0.4889, 0.6074] | 0.4544 [0.2617, 0.5717] |
| Vibration-only CNN | 0.4593 [0.4000, 0.5111] | 0.3314 [0.2611, 0.3713] |

Paired bootstrap cho multimodal trừ vibration-only: Accuracy `+0.0969` [0.0148, 0.1852], nhưng Macro-F1 `+0.1019` [-0.0677, 0.2495] chưa loại trừ 0. So với SVM, multimodal thấp hơn cả Accuracy `-0.2591` [-0.3852, -0.1259] và Macro-F1 `-0.3063` [-0.5947, -0.0406]. Chi tiết McNemar theo seed nằm trong `revision_artifacts/locked_test_statistics/statistics.json`.

## 2. Lấy artifact để không phải train lại

Artifact winner không nằm trong Git. Tải GitHub Release `revision-v2-artifacts-20260805.1`:

```bash
gh release download revision-v2-artifacts-20260805.1 \
  --repo NCKH-LongT/bearing-fault-model \
  --pattern 'bearing-revision-v2-artifacts*'

sha256sum -c bearing-revision-v2-artifacts.tar.gz.sha256
tar -xzf bearing-revision-v2-artifacts.tar.gz
```

Gói chứa 10 winner checkpoint, config/report của từng seed, hai aggregate summary, SVM model/report và validation leaderboard. Gói không chứa raw dataset.

Tạo lại gói từ máy đã chạy thí nghiệm:

```bash
python scripts/package_revision_release.py
```

## 3. Chạy lại từ đầu

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-paper.txt
python scripts/cache_dataset_npy.py --workers 2
python paper/run_revision.py preflight
python paper/run_revision.py smoke
python paper/run_revision.py audit
```

Raw CSV phải được đặt trong `data/` theo `data/manifest.csv`. Dataset, cache, runs và checkpoint đều bị Git ignore.

## 4. Roadmap nâng cấp số liệu đúng khoa học

### P0 — Không làm test overfitting

- Đóng băng test hiện tại và các số đã báo cáo.
- Mọi thay đổi model tiếp theo chỉ dùng train/validation hoặc group cross-validation.
- Kết quả confirmatory mới cần một run/bearing/test set chưa từng xem.
- Không chọn seed tốt nhất; luôn báo cáo toàn bộ seed định trước.

### P1 — Chẩn đoán deep model collapse

1. [x] Lưu prediction và logits cấp file cho mọi seed.
2. [x] Kiểm tra class collapse, confidence và độ nhất trí giữa seed.
3. [x] So sánh best-validation với test theo từng seed.
4. [x] Kiểm tra thang đo và distribution của temperature feature theo split.
5. [ ] Kiểm tra sampler/gradient theo lớp trên một validation experiment mới.
6. [ ] Chạy temperature-only baseline với scaler chỉ fit trên train.

Báo cáo chẩn đoán hiện tại: `revision_artifacts/deep_collapse_diagnostics/diagnostics.md`. Có 9/10 seed bỏ hẳn ít nhất một lớp, không có test file nào được cả năm seed dự đoán nhất trí, và khoảng cách best-validation đến test Macro-F1 là `0.2797–0.9333`. Sáu temperature descriptor đang đi vào nhánh tuyến tính theo đơn vị thô; tỷ lệ giữa độ lệch chuẩn lớn nhất và nhỏ nhất trên train là `2768.14`. Phân tích test này chỉ dùng để giải thích artifact đã khóa, không được dùng để chọn model tiếp theo.

Tạo lại báo cáo:

```bash
python scripts/diagnose_deep_collapse.py \
  --multimodal-root runs/revision_confirm_5060_v2 \
  --vibration-root runs/revision_confirm_vibration_5060_v2 \
  --output-dir paper/revision_artifacts/deep_collapse_diagnostics \
  --windows-per-file 32
```

Tiêu chí qua P1: temperature-only và multimodal đã được đánh giá bằng validation/group-CV, không lớp nào có F1 bằng 0 ở phần lớn seed và độ lệch chuẩn Macro-F1 giảm rõ rệt. Chưa đạt tiêu chí này.

### P2 — Baseline và feature engineering trên validation mới

- SVM grid: `C ∈ {0.1,1,10,100}`, `gamma ∈ {scale,0.001,0.01,0.1}`.
- So sánh mean decision score với probability aggregation.
- Nếu cần probability, dùng calibration group-aware theo file; tránh calibration CV ở cấp window.
- Thử Logistic Regression, Random Forest và XGBoost/LightGBM nếu dependency được khóa.
- Mở rộng handcrafted feature bằng kurtosis, skewness, impulse/shape/clearance factor và band energy; ablation phải chỉ ra feature nào hữu ích.
- Thử vibration + temperature handcrafted baseline để kiểm tra giá trị bổ sung của modality trước khi tăng độ phức tạp deep model.

Không được dùng test hiện tại để chọn thuật toán hoặc hyperparameter.

### P3 — Nâng cấp multimodal model

Chỉ triển khai sau P1/P2:

1. Temperature branch có train-only normalization.
2. Temperature-only, vibration-only và late-concat làm ba mốc bắt buộc.
3. Gated fusion kèm modality mask.
4. Modality dropout để giảm phụ thuộc sensor.
5. Class-balanced loss hoặc focal loss chỉ khi validation/group-CV chứng minh cải thiện ổn định.
6. Pretrained vibration encoder chỉ dùng dữ liệu không giao với evaluation run.

Không tăng kiến trúc chỉ để vượt SVM trên test đã xem.

### P4 — Protocol mạnh hơn

Ưu tiên cao nhất là bổ sung nhiều run/bearing:

- leave-one-run-out;
- leave-one-bearing-out;
- cross-load/cross-speed;
- nested group cross-validation cho model selection;
- một external test khóa hoàn toàn.

Với một run duy nhất, claim phải giới hạn ở within-run file-level classification.

### P5 — Evidence để sửa paper

- Five-seed mean ± standard deviation.
- Bootstrap 95% CI ở cấp file, không bootstrap window.
- Paired bootstrap hoặc McNemar cho cùng test file.
- Robustness: vibration noise, missing temperature, drift và mất một vibration axis.
- Efficiency: parameter count, model size, CPU/GPU latency, STFT time, throughput và peak memory.
- Error analysis quanh Healthy–Degrading và Degrading–Fault boundaries.

Chỉ cập nhật `main.tex` sau khi bảng artifact mới có nguồn truy vết và protocol audit đi kèm.

## 5. Kết quả lịch sử

Temporal `[70,100]%` và whole-trajectory `[0,100]%` trong bản thảo cũ chỉ được giữ làm secondary retrospective analysis. Stratified pretraining lịch sử giao temporal test 25 file; full-range chứa vùng train/validation; các lát 70–90% và 90–100% là single-class. Không dùng chúng làm primary generalization evidence.

## 6. Tài liệu còn hiệu lực

- `README.md`: tài liệu canonical này.
- `REVISION_RUN_CHECKLIST.md`: checklist, lệnh resume và nhật ký.
- `revision_artifacts/protocol_audit.md`: bằng chứng split/overlap.
- `main.tex` và `refs.bib`: bản thảo và bibliography.
