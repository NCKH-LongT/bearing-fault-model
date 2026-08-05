# Revision Guide for the Bearing Health Classification Paper

## 1. Mục tiêu của tài liệu

Tài liệu này hướng dẫn chỉnh sửa bài báo:

**Three-Stage Bearing Health Classification Using Vibration Short-Time Fourier Transform and Temperature Trends in a Run-to-Failure Setting**

Mục tiêu là xử lý trực tiếp các nhóm góp ý chính của reviewer:

1. Chưa làm rõ novelty của phương pháp
2. Thiếu baseline hiện đại và so sánh với phương pháp liên quan
3. Chưa chứng minh generalization
4. Chưa báo cáo computational cost và deployment feasibility
5. Chưa đánh giá robustness với noise, missing hoặc unreliable sensors
6. Một số điểm trong temporal evaluation protocol cần làm rõ hơn

---

# 2. Tóm tắt tình trạng hiện tại của bài báo

Bài báo hiện sử dụng:

- Vibration hai trục
- STFT spectrogram
- ResNet-style 2D CNN
- Temperature descriptor 6 chiều:
  - mean
  - standard deviation
  - slope
  - cho hai kênh nhiệt độ
- Late fusion bằng cách concat:
  - vibration embedding 256-D
  - temperature embedding 32-D
- File-wise mean-logit aggregation
- Hai phase huấn luyện:
  - stratified pretraining
  - temporal fine-tuning
- Temporal split:
  - train: 0–60% TTF
  - validation: 60–70% TTF
  - test: 70–100.1% TTF

Kết quả hiện tại tập trung vào:

- Stratified Macro-F1
- Early-life Degrading F1
- Late-life Fault F1
- Full-lifecycle Accuracy và Macro-F1
- Vibration-only ablation
- SVM 8D baseline

---

# 3. Các vấn đề cần sửa theo mức độ ưu tiên

## Priority 1 — Kiểm tra lại temporal evaluation protocol

### Vấn đề

Nhãn hiện được định nghĩa theo TTF:

- Healthy: 0–60%
- Degrading: 60–90%
- Fault: 90–100%

Trong khi temporal split là:

- Train: 0–60%
- Validation: 60–70%
- Test: 70–100.1%

Điều này có thể dẫn đến:

- temporal train gần như chỉ chứa Healthy
- validation gần như chỉ chứa Degrading
- test chứa Degrading và Fault

Reviewer có thể đặt câu hỏi:

- Temporal fine-tuning học ba lớp bằng cách nào?
- Fault đã xuất hiện trong stratified pretraining hay chưa?
- Có leakage từ phase pretraining sang temporal test hay không?
- Mô hình có thật sự học forward degradation hay chỉ reuse checkpoint đã thấy đủ lớp?

### Việc cần làm

- [ ] Xuất bảng số lượng file và window theo từng split
- [ ] Xuất số lượng mẫu theo từng lớp trong train/val/test
- [ ] Ghi rõ phase nào đã nhìn thấy Healthy, Degrading, Fault
- [ ] Ghi rõ checkpoint stratified có được train trên các file xuất hiện trong temporal test hay không
- [ ] Ghi rõ layer nào được freeze hoặc fine-tune
- [ ] Ghi rõ optimizer, learning rate và class weights cho temporal phase
- [ ] Kiểm tra có overlap window giữa hai split hay không
- [ ] Kiểm tra có file nào xuất hiện ở cả train và test hay không

### Bảng cần thêm vào paper

| Protocol | Partition | TTF range | Number of files | Number of windows | Healthy | Degrading | Fault |
|---|---|---:|---:|---:|---:|---:|---:|
| Temporal | Train | 0–60% | | | | | |
| Temporal | Validation | 60–70% | | | | | |
| Temporal | Test | 70–100.1% | | | | | |

### Hướng sửa nội dung

Nếu chỉ dùng một run, không nên gọi đây là bằng chứng mạnh về generalization.

Nên đổi cách diễn đạt thành:

> within-run forward-transfer evaluation

hoặc:

> chronology-preserving late-life evaluation within a single run

Không nên viết:

> The model generalizes well to unseen bearing conditions.

---

## Priority 2 — Kiểm tra lại full-lifecycle result

### Vấn đề

Bài báo cáo full-lifecycle result trên 0–100.1% TTF.

Cần xác minh:

- Có dùng lại các file thuộc train không?
- Có dùng lại validation không?
- Có phải toàn bộ trajectory của cùng run không?

Nếu có chứa train data, kết quả này không phải held-out test result.

### Việc cần làm

- [ ] Kiểm tra code tạo full-lifecycle prediction
- [ ] Xác minh danh sách file được đưa vào evaluation
- [ ] Tách rõ:
  - held-out test
  - retrospective whole-trajectory analysis
- [ ] Không đưa full-lifecycle result vào Abstract nếu nó chứa train data
- [ ] Đổi tên kết quả nếu cần

### Cách đặt tên an toàn

Thay vì:

> Full-lifecycle test performance

Dùng:

> Whole-trajectory retrospective evaluation

---

## Priority 3 — Không dùng single-class slices làm kết quả chính

### Vấn đề

Các slice:

- 70–90% gần như chỉ có Degrading
- 90–100.1% gần như chỉ có Fault

Nếu slice chỉ có một lớp, Accuracy hoặc F1 cao không đủ chứng minh khả năng phân biệt ba lớp.

### Việc cần làm

- [ ] Ghi số lượng lớp hiện diện trong từng slice
- [ ] Ghi support từng lớp
- [ ] Đưa single-class evaluation xuống secondary analysis
- [ ] Không dùng Fault F1 trên single-class slice làm bằng chứng chính
- [ ] Bổ sung evaluation trên multi-class hoặc transition-region test set

### Metric nên thêm

- Macro-F1 trên tập có đủ lớp
- Balanced Accuracy
- Per-class Recall
- False alarm rate trên Healthy
- Detection delay
- Transition detection performance
- Confidence interval

---

# 4. Làm rõ novelty của phương pháp

## Vấn đề hiện tại

Kiến trúc hiện tại chủ yếu là:

```text
STFT spectrogram
→ ResNet-style encoder
→ 256-D vibration embedding

Temperature 6-D
→ Linear
→ 32-D temperature embedding

Concatenation
→ Linear classifier
```

Reviewer có thể xem đây là standard late fusion.

## Hướng sửa 1 — Giữ mô hình hiện tại, sửa cách claim

Không nên gọi đây là novel fusion architecture.

Nên đóng góp theo hướng:

- compute-efficient multimodal pipeline
- compact temperature trend representation
- leakage-aware deployment-oriented evaluation
- systematic ablation

### Câu đóng góp gợi ý

> We introduce a compact vibration–temperature monitoring pipeline that combines two-axis STFT features with a six-dimensional temperature trend descriptor under a chronology-preserving evaluation protocol.

## Hướng sửa 2 — Nâng cấp fusion module

Có thể thay simple concatenation bằng gated fusion.

### Kiến trúc gợi ý

```text
hv = vibration embedding
ht = temperature embedding

g = sigmoid(Wg [hv ; ht] + bg)

hf = g ⊙ hv + (1 - g) ⊙ Wt ht

logits = classifier(hf)
```

### Lợi ích

- Mô hình học khi nào nên ưu tiên vibration
- Mô hình học khi nào temperature đáng tin cậy hơn
- Có thể mở rộng sang missing-modality robustness
- Novelty rõ hơn concatenation thông thường

### Ablation bắt buộc

- [ ] Vibration only
- [ ] Temperature only
- [ ] Simple concatenation
- [ ] Score-level fusion
- [ ] Gated fusion
- [ ] Without temperature slope
- [ ] Mean and std only
- [ ] Without per-frequency normalization

---

# 5. Bổ sung baseline

## Baseline hiện tại

- Vibration-only deep model
- Classical SVM với 8D vibration features

Các baseline này chưa đủ để reviewer đánh giá competitiveness.

## Baseline tối thiểu nên chạy

### Classical

- [ ] SVM vibration-only
- [ ] SVM vibration + temperature
- [ ] Random Forest
- [ ] XGBoost hoặc LightGBM nếu phù hợp

### Deep learning

- [ ] 1D CNN trên raw vibration
- [ ] 2D CNN trên STFT
- [ ] ResNet18
- [ ] CNN-LSTM hoặc CNN-GRU
- [ ] TCN

### Multimodal

- [ ] Late concatenation
- [ ] Score-level fusion
- [ ] Gated fusion

## Điều kiện so sánh công bằng

Tất cả model phải dùng:

- cùng split
- cùng file list
- cùng window size
- cùng overlap
- cùng seed set
- cùng class definition
- cùng aggregation
- cùng evaluation metrics

## Bảng kết quả gợi ý

| Model | Input | Params | Accuracy | Macro-F1 | F1 Healthy | F1 Degrading | F1 Fault |
|---|---|---:|---:|---:|---:|---:|---:|
| SVM | Vib handcrafted | | | | | | |
| 1D CNN | Raw vibration | | | | | | |
| ResNet18 | STFT vibration | | | | | | |
| CNN-GRU | STFT sequence | | | | | | |
| Proposed | Vib + Temp | | | | | | |

---

# 6. Generalization experiments

## Vấn đề hiện tại

Bài chỉ sử dụng một run với 129 files.

Điều này chưa chứng minh:

- cross-run generalization
- cross-bearing generalization
- cross-load generalization
- cross-speed generalization

## Hướng tốt nhất

- [ ] Sử dụng nhiều run nếu dataset có
- [ ] Leave-one-run-out
- [ ] Leave-one-bearing-out
- [ ] Train trên operating condition A, test trên B
- [ ] Cross-load test
- [ ] Cross-speed test

## Nếu không có dataset multimodal khác

Có thể tách thành hai thí nghiệm:

### Experiment A

Full multimodal model trên dataset hiện tại

### Experiment B

Vibration encoder generalization trên dataset bearing khác

Cần ghi rõ Experiment B chỉ kiểm tra vibration branch, không chứng minh full multimodal generalization.

## Nếu chưa thể thêm dữ liệu

Phải giảm mức claim:

> The results demonstrate within-run forward-transfer performance, while cross-run and cross-bearing generalization remain unverified.

---

# 7. Computational cost và deployment feasibility

## Vấn đề hiện tại

Việc nói model chạy trên RTX 2060 chưa đủ để chứng minh model lightweight.

## Số liệu cần đo

- [ ] Number of trainable parameters
- [ ] Model size in MB
- [ ] FLOPs per window
- [ ] Inference latency, batch size 1
- [ ] Throughput
- [ ] Peak GPU memory
- [ ] CPU inference latency
- [ ] STFT preprocessing time
- [ ] Temperature feature extraction time
- [ ] End-to-end latency
- [ ] Training time per epoch

## Bảng cần thêm

| Model | Params | FLOPs/window | Size | GPU latency | CPU latency | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| Vibration-only | | | | | | |
| Proposed fusion | | | | | | |

## Cách sửa câu trong paper

Không viết:

> The model is lightweight.

Nên viết:

> The model contains X million parameters and requires Y ms per one-second window on an RTX 2060.

---

# 8. Robustness experiments

## 8.1 Vibration noise

Thêm Gaussian noise với các mức:

- [ ] 20 dB
- [ ] 10 dB
- [ ] 5 dB
- [ ] 0 dB

Đo:

- Accuracy
- Macro-F1
- F1 từng lớp
- mức giảm so với clean data

## 8.2 Temperature corruption

Thử:

- [ ] constant offset
- [ ] linear drift
- [ ] random spikes
- [ ] clipping
- [ ] stuck-at-constant
- [ ] missing segments

## 8.3 Missing modality

Thử:

- [ ] mất temperature 10%
- [ ] mất temperature 30%
- [ ] mất temperature 50%
- [ ] mất một vibration axis
- [ ] mất liên tục một đoạn signal
- [ ] mất ngẫu nhiên từng window

## 8.4 Cách xử lý missing

So sánh:

- zero fill
- mean imputation
- last observation carried forward
- modality mask
- modality dropout
- fallback to vibration-only branch

## Bảng robustness gợi ý

| Corruption | Level | Vib-only | Concatenation | Proposed |
|---|---:|---:|---:|---:|
| Gaussian noise | 10 dB | | | |
| Missing temperature | 30% | | | |
| Temperature drift | medium | | | |
| Missing vibration axis | 1 axis | | | |

---

# 9. Statistical reliability

## Việc cần làm

- [ ] Chạy ít nhất 5 random seeds
- [ ] Báo cáo mean ± standard deviation
- [ ] Bootstrap 95% confidence interval ở file level
- [ ] Không bootstrap theo window nếu output cuối là file-level
- [ ] So sánh multimodal và vibration-only bằng paired bootstrap hoặc McNemar test

## Bảng gợi ý

| Model | Accuracy | Macro-F1 | F1 Healthy | F1 Degrading | F1 Fault |
|---|---:|---:|---:|---:|---:|
| Vib-only | mean ± std | | | | |
| Proposed | mean ± std | | | | |

---

# 10. Sửa Abstract

## Abstract hiện tại cần bổ sung

- kiến trúc fusion rõ hơn
- baseline chính
- held-out multi-class result
- computational cost
- robustness
- limitation về single-run

## Không nên

- nhấn quá mạnh Fault F1 trên single-class slice
- gọi full-lifecycle result là test nếu chứa train data
- gọi mô hình lightweight mà không có số đo

## Template Abstract mới

```text
This study addresses three-stage bearing health classification under a
run-to-failure setting using two-axis vibration and temperature measurements.
The proposed method employs a ResNet-based encoder for STFT spectrograms and
a compact six-dimensional temperature trend descriptor integrated through a
[gated/late] fusion module. Evaluation is conducted at file level under a
chronology-preserving protocol with no overlapping windows across partitions.

Compared with vibration-only, classical feature-based, and temporal deep-learning
baselines, the proposed model improves Macro-F1 by X percentage points on the
held-out multi-class evaluation. The model contains X million parameters and
requires Y ms per one-second window. Robustness tests under vibration noise and
missing temperature measurements show a performance degradation of Z.

The current evaluation is limited to [one run / the evaluated conditions], and
cross-run generalization remains future work.
```

---

# 11. Sửa Introduction

## Cần thêm

- practical importance của Degrading stage
- vấn đề temporal leakage
- limitation của random window split
- gap trong multimodal fusion
- gap về efficiency và robustness
- gap về missing sensors

## Đoạn research gap gợi ý

```text
Existing bearing diagnosis studies often focus on fault-type classification under
controlled conditions, while fewer studies address health-stage recognition over a
run-to-failure trajectory. In addition, window-level random splitting may lead to
optimistic performance because adjacent windows are highly correlated. Existing
multimodal approaches also rarely report computational cost, missing-sensor robustness,
and chronology-preserving evaluation together.
```

---

# 12. Sửa Contributions

Nên viết còn ba contribution chính.

## Contribution 1 — Method

```text
A compact vibration–temperature classification framework combining two-axis STFT
representations with a six-dimensional temperature trend descriptor through a
[gated/late] fusion module.
```

## Contribution 2 — Evaluation

```text
A chronology-preserving, file-level evaluation protocol designed to reduce overlap
leakage and analyze transition regions across the run-to-failure trajectory.
```

## Contribution 3 — Empirical analysis

```text
A systematic comparison against classical, vibration-only, temporal, and multimodal
baselines, together with efficiency and robustness evaluations.
```

---

# 13. Sửa Related Work

Tổ chức thành bốn nhóm:

## 13.1 Time-frequency bearing diagnosis

- STFT
- CWT
- spectrogram CNN
- ResNet-based diagnosis

## 13.2 Run-to-failure and degradation-stage classification

- Healthy/Degrading/Fault
- degradation onset
- RUL-related stage modeling

## 13.3 Multimodal fusion

- vibration + temperature
- sensor fusion
- early/late fusion
- reliability-aware fusion

## 13.4 Leakage-aware and generalization evaluation

- random split leakage
- group split
- temporal split
- cross-domain evaluation

Cuối phần Related Work phải có gap rõ ràng.

---

# 14. Sửa Methodology

## Cần mô tả rõ hơn

- exact tensor shape
- CNN backbone chi tiết
- số residual blocks
- embedding dimension
- temperature normalization
- slope calculation
- fusion equation
- classifier architecture
- class weights
- aggregation rule
- phase 1 và phase 2
- freeze/unfreeze strategy

## Nếu thêm gated fusion

Viết rõ:

```text
hv ∈ R^256
ht ∈ R^32

g = sigmoid(Wg [hv ; ht] + bg)

hf = g ⊙ hv + (1 - g) ⊙ Wt ht

logits = Wc hf + bc
```

---

# 15. Sửa Experimental Setup

Cần thêm:

- [ ] hardware
- [ ] software version
- [ ] seed list
- [ ] file IDs theo split
- [ ] class distribution
- [ ] hyperparameter search space
- [ ] early stopping criterion
- [ ] best checkpoint rule
- [ ] class weights
- [ ] model selection metric
- [ ] aggregation method
- [ ] number of repetitions

---

# 16. Sửa Results

Nên chia lại thành:

## 16.1 Primary held-out evaluation

- multi-class
- file-level
- không chứa train files

## 16.2 Baseline comparison

- classical
- deep learning
- multimodal

## 16.3 Transition-region analysis

- Healthy → Degrading
- Degrading → Fault

## 16.4 Robustness

- noise
- missing temperature
- missing vibration axis

## 16.5 Efficiency

- parameters
- latency
- FLOPs
- memory

## 16.6 Retrospective trajectory analysis

- probability vs TTF
- prediction transition
- confidence
- temperature trend

---

# 17. Hình và bảng nên thêm

## Hình

- [ ] Updated architecture diagram
- [ ] Class distribution over TTF
- [ ] Predicted class probabilities over TTF
- [ ] Temperature trend over TTF
- [ ] Error locations near stage boundaries
- [ ] Robustness curve vs SNR
- [ ] Inference latency comparison

## Bảng

- [ ] Split and class distribution
- [ ] Baseline comparison
- [ ] Fusion ablation
- [ ] Efficiency comparison
- [ ] Robustness comparison
- [ ] Multi-seed result
- [ ] Generalization result

---

# 18. Sửa Discussion

Discussion cần trả lời:

- Vì sao temperature giúp late-life Fault?
- Vì sao Degrading thường bị nhầm thành Healthy?
- Fusion nào có lợi nhất?
- Kết quả có ổn định qua seed không?
- Model suy giảm thế nào khi thiếu sensor?
- Model có đủ nhanh để deployment không?
- Single-run limitation ảnh hưởng mức nào?
- Stage threshold có phụ thuộc dataset không?

Không nên chỉ lặp lại con số trong Results.

---

# 19. Sửa Limitations

Nên ghi rõ:

- single-run evaluation
- empirical TTF thresholds
- possible residual temporal correlation
- synchronous sensing assumption
- no cross-bearing validation
- limited operating conditions
- dependence on temperature sensor availability
- no real-time edge deployment test nếu chưa thực hiện

---

# 20. Sửa Conclusion

Conclusion chỉ nên claim đúng với evidence.

## Cách viết an toàn

```text
The proposed multimodal pipeline improves within-run file-level classification over
the vibration-only baseline under the evaluated protocol. Temperature trends provide
complementary information, particularly near late-life degradation. However, the
current evidence is limited to the evaluated run and does not establish cross-bearing
or cross-condition generalization.
```

---

# 21. Response to Reviewer template

## Comment 1 — Novelty

```text
Reviewer comment:
The methodological novelty is not sufficiently clear.

Response:
We thank the reviewer for this observation. We agree that the previous manuscript
did not sufficiently distinguish the proposed fusion mechanism from standard late
concatenation.

Revision:
We have [clarified the contribution as a compact deployment-oriented pipeline /
replaced the original concatenation head with a gated fusion module]. We also added
ablations against vibration-only, temperature-only, score-level fusion, and simple
concatenation.

Location:
Section X, Figure X, Table X.
```

## Comment 2 — Missing SOTA comparison

```text
Response:
We added classical, vibration-only deep-learning, temporal, and multimodal baselines
under the same file-level split and preprocessing pipeline. All methods are evaluated
using the same seeds, metrics, and aggregation rule.
```

## Comment 3 — Generalization

```text
Response:
We agree that the original single-run evaluation did not establish cross-bearing
generalization. We have [added leave-one-run-out experiments / revised the claims to
within-run forward-transfer evaluation]. The limitation is now stated explicitly in
the Abstract, Discussion, and Conclusion.
```

## Comment 4 — Computational cost

```text
Response:
We now report parameter count, FLOPs, model size, GPU and CPU latency, throughput,
peak memory, and preprocessing time.
```

## Comment 5 — Robustness

```text
Response:
We added controlled experiments involving vibration noise, temperature drift,
missing temperature windows, and missing vibration channels. The revised manuscript
also compares zero filling, imputation, and modality-aware training.
```

---

# 22. Kế hoạch thực hiện theo thứ tự

## Phase 1 — Audit lại code và data

- [ ] Kiểm tra file split
- [ ] Kiểm tra overlap
- [ ] Kiểm tra class distribution
- [ ] Kiểm tra full-lifecycle evaluation
- [ ] Kiểm tra stratified checkpoint leakage
- [ ] Lưu manifest chính thức

## Phase 2 — Chạy lại baseline

- [ ] Vib-only
- [ ] Temp-only
- [ ] SVM
- [ ] 1D CNN
- [ ] ResNet STFT
- [ ] CNN-GRU hoặc TCN
- [ ] Fusion variants

## Phase 3 — Thêm robustness và efficiency

- [ ] Noise experiments
- [ ] Missing temperature
- [ ] Missing vibration axis
- [ ] Params
- [ ] FLOPs
- [ ] Latency
- [ ] Memory

## Phase 4 — Statistical testing

- [ ] 5 seeds
- [ ] mean ± std
- [ ] bootstrap confidence interval
- [ ] significance test

## Phase 5 — Viết lại paper

- [ ] Abstract
- [ ] Introduction
- [ ] Contributions
- [ ] Related Work
- [ ] Methodology
- [ ] Experimental Setup
- [ ] Results
- [ ] Discussion
- [ ] Limitations
- [ ] Conclusion

## Phase 6 — Chuẩn bị response letter

- [ ] Copy từng reviewer comment
- [ ] Trả lời từng comment
- [ ] Nêu thay đổi cụ thể
- [ ] Ghi section, page, table, figure
- [ ] Không trả lời chung chung
- [ ] Không claim vượt quá evidence

---

# 23. Checklist trước khi nộp lại

## Protocol

- [ ] Không có overlap window giữa split
- [ ] Không có file leakage
- [ ] Class distribution rõ ràng
- [ ] Full-lifecycle result được đặt tên đúng
- [ ] Single-class slice không phải kết quả chính

## Method

- [ ] Fusion architecture rõ
- [ ] Tensor shape rõ
- [ ] Training phases rõ
- [ ] Class weights rõ
- [ ] Aggregation rõ

## Experiments

- [ ] Baseline đủ mạnh
- [ ] Multiple seeds
- [ ] Efficiency metrics
- [ ] Robustness metrics
- [ ] Generalization hoặc claim được hạ phù hợp

## Writing

- [ ] Abstract không overclaim
- [ ] Contributions cụ thể
- [ ] Related Work có gap
- [ ] Discussion có error analysis
- [ ] Limitations đầy đủ
- [ ] Conclusion đúng với evidence

## Reviewer response

- [ ] Từng comment có response riêng
- [ ] Có evidence
- [ ] Có vị trí thay đổi
- [ ] Có bảng/hình mới nếu cần
- [ ] Giọng văn lịch sự và trực tiếp

---

# 24. Tiêu chí hoàn thành revision

Bản revision được xem là đủ tốt khi:

1. Reviewer hiểu chính xác novelty nằm ở đâu
2. Split và evaluation không còn mơ hồ
3. Kết quả chính đến từ held-out multi-class evaluation
4. Có baseline hiện đại và so sánh công bằng
5. Có số liệu chứng minh lightweight
6. Có robustness test
7. Claim generalization phù hợp với dữ liệu
8. Có multiple seeds và confidence interval
9. Tất cả reviewer comments đều được trả lời bằng thay đổi cụ thể
