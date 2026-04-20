# Tóm tắt dự án: Three-Stage Bearing Health Classification (STFT + Temperature, Run-to-Failure)

Tài liệu này tóm tắt **quá trình thực hiện** và **kết quả chính** của dự án trong repo `Bearing Fault`, dựa trên nội dung trong file `Three_Stage_Bearing_Health_Classification_Using_Vibration_STFT_and_Temperature_Trends_in_a_Run_to_Failure_Setting.pdf`. Mục tiêu là chuẩn bị khung nội dung để viết một bài báo mới (paper mới) dựa trên hệ thống/ý tưởng đã làm.

---

## 1) Bài toán & mục tiêu

- Bài toán: phân loại sức khỏe vòng bi **3 giai đoạn**: **Healthy / Degrading / Fault** trong bối cảnh **run-to-failure**.
- Điểm nhấn triển khai thực tế: tránh đánh giá “ảo” do **temporal correlation** và **overlapped windows leakage** khi chia train/test theo kiểu random ở mức cửa sổ.
- Hướng tiếp cận: kết hợp **vibration** (biểu diễn time–frequency bằng STFT) + **temperature** (trend/statistics) theo mô hình **late fusion** nhẹ.

---

## 2) Dữ liệu & định nghĩa nhãn

- Dữ liệu run-to-failure (một run):
  - `RUNS = 1`, `FILES = 129`
  - Vibration 2 trục, sampling **25.6 kHz**
  - Nhiệt độ 2 kênh (bearing + ambient/atm)
  - Mỗi sample có 4 kênh: `[vib_x, vib_y, temp_bearing, temp_atm]`
- Cửa sổ:
  - `window_seconds = 1.0s` → 25,600 mẫu
  - `hop_seconds = 0.5s` (overlap 50%)
- Chuẩn hoá timeline:
  - `TTF% ∈ [0,1]` (tỷ lệ thời gian tới hỏng theo vòng đời đã chuẩn hoá)
- Nhãn 3 lớp theo ngưỡng TTF:
  - Healthy: `[0, 0.60)`
  - Degrading: `[0.60, 0.90)`
  - Fault: `[0.90, 1.00]`

---

## 3) Pipeline đặc trưng

### 3.1 Vibration → STFT log-spectrogram (2 kênh)

Mỗi window rung (1s) được xử lý:
- Sanitization: thay NaN/Inf bằng 0.
- Z-score theo **mỗi window, mỗi trục** trên waveform.
- STFT:
  - `n_fft = 4096`, `hop_length = 1024`, Hann window
  - Lấy magnitude, log-compress: `log(1 + α|X|)` (thực tế là `log1p(mag * log_add)` với `log_add = 1.0`)
- Chuẩn hoá “độ sáng” phổ:
  - **per-frequency normalization**: z-score theo **trục thời gian** cho từng bin tần số.
- Resize phổ về kích thước cố định:
  - Dev: `160×160`
  - Temporal FT: `224×224`
- Stack 2 trục → tensor `2×H×W`.

### 3.2 Temperature → descriptor 6-D (mỗi window)

Cho 2 kênh nhiệt độ (bearing, ambient), tính 3 đặc trưng/kênh:
- mean
- std
- slope (least-squares theo index trong window)

→ tổng cộng **6-D / window**. Nếu không có giá trị hữu hạn thì set 0.

---

## 4) Mô hình

- Nhánh rung: CNN 2D nhẹ kiểu **ResNet-18 variant** cho input 2 kênh.
  - Stem 7×7 stride 2, channel nhỏ (32→256)
  - Global average pooling → embedding rung **256-D**
- Nhánh nhiệt: Linear `6 → 32` + ReLU → embedding nhiệt **32-D**
- Late fusion: concat `[256;32] = 288-D` → Linear classifier (3 lớp)
- Inference theo file (file-wise):
  - forward các windows của cùng 1 file
  - lấy **mean của logits (pre-softmax)** qua windows
  - dự đoán file bằng `argmax(mean_logits)`

---

## 5) Protocol đánh giá (chống leakage) & workflow 2 pha

### 5.1 Dev (stratified, file-wise)

- Split theo file: train/val/test = **0.6/0.2/0.2**
- Mục đích: chọn cấu hình/siêu tham số trong giai đoạn phát triển.

### 5.2 Deployment-oriented (temporal, leakage-aware theo TTF)

- Split theo TTF (contiguous theo thời gian):
  - train: **[0, 60]%**
  - val: **[60, 70]%**
  - test: **[70, 100.1]%**
- Fine-tune từ checkpoint tốt nhất của stratified (`init_from = best stratified`)
- Lưu ý quan trọng về metric:
  - Ở test slice late-life, **Healthy có thể không xuất hiện** → báo cáo metric theo **present classes** (thường là Degrading + Fault).

---

## 6) Huấn luyện (tóm tắt)

### 6.1 Stratified training (dev)

- Optimizer: AdamW
- LR: `2e-4`, weight decay: `1e-3`
- Epochs: đến 80, early stopping patience 20
- Balanced sampling, label smoothing 0.05, AMP bật
- Chọn best checkpoint theo macro-F1

### 6.2 Temporal fine-tuning

- Init: từ best stratified checkpoint
- AdamW: LR `3e-5`, weight decay `1e-2`
- Epochs đến 60, early stopping patience 12
- Cosine schedule (OneCycle off), AMP bật, balanced sampling
- `val_max_windows = 50` để giảm chi phí validation

---

## 7) Kết quả chính

### 7.1 Dev (stratified, 3-class)

- **Macro-F1 ≈ 0.7762**
- F1 theo lớp (xấp xỉ):
  - Healthy ≈ 0.69
  - Degrading ≈ 0.64
  - Fault = 1.00

### 7.2 Temporal (late-life, present classes)

Trên test slice **[70, 100.1]% TTF** (Healthy không xuất hiện):
- **Accuracy ≈ 0.95**
- **Macro-F1 (present classes) ≈ 0.9467**
- F1 (present classes, xấp xỉ):
  - Degrading ≈ 0.9333
  - Fault ≈ 0.9600

### 7.3 Failure case (broader temporal coverage)

Khi mở rộng slice test để chứa nhiều vùng “mơ hồ” hơn (gần ranh Healthy↔Degrading), performance giảm mạnh:
- **Accuracy ≈ 0.3333**
- **Macro-F1 ≈ 0.3324**

---

## 8) Ablation & quan sát

### 8.1 Stratified init vs. từ scratch (temporal FT)

- Fine-tune có init từ stratified tốt hơn rõ rệt so với train/fine-tune từ scratch (đặc biệt ở late-life slice).

### 8.2 Multi-modal (vib+temp) vs vib-only

- Late-life slice **[85, 100.1]% TTF** (present-labels):
  - vib-only: Macro-F1 ≈ 0.2593 (Degrading ≈ 0.5185, Fault = 0.0000)
  - vib+temp: Macro-F1 ≈ 0.9467 (Degrading ≈ 0.9333, Fault ≈ 0.9600)
- Broad late slice **[70, 100.1]% TTF** (present-labels):
  - vib-only: Macro-F1 ≈ 0.4000 (Degrading ≈ 0.8000, Fault = 0.0000)
  - vib+temp: Macro-F1 ≈ 0.3324 (Degrading ≈ 0.0741, Fault ≈ 0.9231)

Diễn giải nhanh: nhiệt độ giúp mạnh ở vùng rất-late (Fault rõ), nhưng ở broad slice có trade-off (Degrading sớm dễ bị khó).

---

## 9) Hạn chế & hướng phát triển (gợi ý cho paper mới)

- External validity: mới thử trên **1 run** → cần thêm runs/bearings, đánh giá bearing-wise để kiểm tra domain shift.
- Construct validity: nhãn stage dựa trên ngưỡng TTF (0.60/0.90) → cần:
  - sensitivity analysis theo `(τ1, τ2)`
  - hoặc biên giới dữ liệu/physics-guided: change-point detection dựa RMS/kurtosis/envelope spectrum/temperature slope.
- Internal validity: overlap 50% tăng nguy cơ leakage → cân nhắc:
  - thêm “temporal gap” giữa train/val/test
  - hoặc dùng non-overlapping windows cho evaluation.
- Mô hình: Degrading sớm khó → có thể thử:
  - sequence-aware (TCN/Transformer/LSTM) với context dài hơn
  - temperature modeling giàu hơn (trend dài hạn thay vì 1s slope).

---

## 10) Liên hệ với mã nguồn trong repo (để tái hiện)

Theo `README.md`:
- Train stratified: `python train_logs.py --config configs/logs_stft_train_strat.yaml`
- Eval stratified: `python eval_logs.py --config configs/logs_stft_train_strat.yaml --ckpt runs/logs_stft_strat/best.pt --show`
- Temporal FT: `python train_logs.py --config configs/logs_stft_full_temporal.yaml`
- Eval temporal: `python eval_logs.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --show`

---

## 11) Gợi ý cấu trúc paper mới (khung mục lục)

1. Introduction (vấn đề + leakage + nhu cầu 3-stage)
2. Related Work (bearing TF-CNN, run-to-failure stage labeling, leakage-aware evaluation, multimodal fusion)
3. Dataset & Labeling (TTF, thresholds, windowing)
4. Method (STFT + normalization, temp 6-D, model, file-wise aggregation)
5. Experimental Setup (stratified vs temporal FT, metrics present-class)
6. Results + Ablations + Failure-case
7. Discussion & Threats to Validity
8. Conclusion & Future Work

