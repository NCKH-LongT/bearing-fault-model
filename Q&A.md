Q&A — Reviewer Responses and Clarifications
=========================================

This document collects concise answers to likely reviewer questions and outlines planned strengthening experiments.

1) External Validity: RUNS=1 and risk of overfitting
- Question: With a single run (RUNS=1, 129 files), how do you ensure the ResNet‑18 + late‑fusion model does not overfit to one bearing’s idiosyncrasies? Why not cross‑validate on CWRU/Paderborn?
- Answer (current scope): We do not claim absolute generalization with RUNS=1. Instead, we reduce temporal over‑optimism via a leakage‑aware TTF split (train earlier, test later), control model capacity (small ResNet‑18, weight decay, label smoothing, early stopping, balanced sampling), and make file‑wise decisions via mean pre‑softmax logits. The two‑phase workflow (stratified→temporal) demonstrates transfer from early‑life to late‑life without temporal leakage.
- Why not cross‑dataset immediately: CWRU/Paderborn are fault‑seeded, short, and differ in sensors/sampling/labels (no TTF Degrading); a naïve transfer would conflate domain shift with task mismatch. We will report vibration‑only, leakage‑aware baselines on these datasets with harmonized window/STFT settings as a sanity check, while keeping the primary focus on TTF‑oriented evaluation.
- Strengthening plan (actionable):
  - Multi‑run holdout when available: leave‑one‑run‑out (LOTO) / bearing‑wise splits.
  - Blocked temporal CV within the run: rotate early/mid/late slices; report mean±std.
  - Cross‑condition holdout: reserve unseen load/speed conditions for test (if metadata available).
  - Cross‑dataset sanity (vibration‑only): CWRU, Paderborn with consistent window/STFT and anti‑leakage protocol.
  - Ablations to rule out overfitting: capacity sweep (width 16–128), +temp vs vib‑only, and robustness on TTF bins.

2) Auto‑train rounds vs. number of runs
- Question: Auto‑train many rounds (auto_r*) means RUNS>1?
- Answer: No. Each round is a fresh train→eval for hyperparameter search on the same dataset; RUNS remains 1. We pick the best stratified checkpoint, then fine‑tune temporally.

3) Novelty vs. upstream repository (mdzalfirdausi)
- Question: What is new beyond the upstream CNN?
- Answer: Our pipeline targets deployment realism on run‑to‑failure data by combining: (i) leakage‑aware, TTF‑aligned evaluation (with present‑class reporting on late‑life slices), (ii) multi‑modal fusion with 6‑D temperature trends, (iii) file‑wise mean pre‑softmax logit decisions, and (iv) STFT normalization tailored to windowed RTF signals (per‑window waveform z‑score + per‑frequency frame‑wise z‑score). This shifts the focus from single‑sensor, window‑level accuracy to a deployment‑oriented, file‑level assessment without temporal leakage.

4) Why the Degrading stage is hard, and how to improve it
- Question: Why is Degrading difficult to classify and how can it be improved?
- Answer: Degrading overlaps with Healthy near the early/mid boundary (TTF≈0.6), exhibits condition shifts, and labels are thresholded. Improvements:
  - Sequence‑aware heads: add TCN/LSTM/GRU or a light Transformer over short window sequences on top of the 2D CNN to use local temporal context.
  - Multi‑scale temperature trends: short/medium slopes, EMA trends, or STL residuals beyond single‑window slope.
  - Curriculum FT around TTF∈[0.5,0.8]: upweight/debias ambiguous mid‑life; optionally focal loss (moderate γ).
  - Robust targets: soft labels or neighborhood‑consistent pseudo‑labels within short time neighborhoods.
  - Uncertainty handling: calibration and selective abstention near stage boundaries.

5) Leakage‑aware evaluation — justification and practice
- Question: Why emphasize leakage‑aware temporal evaluation?
- Answer: Overlapped windows make adjacent frames highly similar; random/window‑level splits leak near‑duplicates and inflate accuracy. We enforce TTF‑aligned splits (train earlier, test later), report present‑class metrics on late‑life slices (no Healthy), and provide time‑binned metrics (Acc/Macro‑F1 vs. TTF). We also cap validation windows for speed only; test uses full file‑wise aggregation. A temporal gap between split boundaries can further reduce adjacency effects (planned sensitivity study).

6) File‑wise aggregation choice
- Question: Why average logits (pre‑softmax) instead of probabilities?
- Answer: Averaging logits is less biased by over‑confident per‑window predictions and aligns with linear decision surfaces in logit space; empirically it yields more stable file‑level labels than averaging probabilities or majority vote.

7) Temperature sampling and alignment
- Question: Are temperature windows aligned with vibration windows?
- Answer: Yes. All four channels are sampled synchronously; we slice the same indices [s:e] for vibration and temperature. If future deployments use different sampling rates, we will add explicit resampling/timestamp alignment.

8) Planned reporting to strengthen the paper
- Add ablations: vib‑only vs +temp; capacity sweep; OneCycle vs Cosine; input size sensitivity; temporal gap sensitivity.
- Add cross‑dataset sanity (vibration‑only) on CWRU/Paderborn with anti‑leakage splits and harmonized STFT.
- Add time‑metrics figure (Acc/Macro‑F1 vs TTF bins) with confidence intervals.
- Add bearing/run‑wise holdout experiments when more runs are available.

---

Prepared to accompany the main manuscript (main.tex) for reviewer discussion and camera‑ready planning.

9) Ablation of temperature features, boundary leakage handling, and late‑life vs full‑life deployment

- Question (VI): Kết quả ở Hình 1 cho thấy Degrading có F1 thấp (~0.64) so với Fault (1.00). Có ablation chứng minh thêm 6 đặc trưng nhiệt giúp cải thiện so với chỉ STFT rung không? 6 chỉ số này có đủ mô tả “xu hướng nhiệt độ” chưa? Ở mốc 60% TTF (train/val), các cửa sổ chồng lấn sát ranh có thể rò rỉ — nhóm xử lý thế nào? Có cần “temporal gap”? Việc chỉ đánh giá Degrading vs Fault ở cuối đời có phản ánh đúng triển khai khi Healthy chiếm đa số? FAR sẽ ra sao nếu hay nhầm Healthy↔Degrading?

  Answer (VI):
  - Ablation: Hiện tại bài nộp chưa kèm số liệu ablation; chúng tôi sẽ bổ sung (i) vib‑only, (ii) +temp(6‑D), (iii) +temp đa tỉ lệ (multi‑scale slopes/EMA). Kỳ vọng cải thiện chủ yếu ở Degrading (TTF 60–90%). Tất cả cấu hình, seed, lịch train giữ nguyên để so sánh công bằng, báo cáo ở mức file‑wise và theo bin TTF.
  - Đủ hay chưa: 6‑D (mean/std/slope) là đặc trưng xu hướng mức cửa sổ; để mô tả xu hướng tốt hơn, chúng tôi sẽ thêm slope đa thang, EMA/ngưỡng trôi (drift), hoặc phân rã STL (trend/residual). Với mô hình chuỗi (TCN/LSTM/Transformer) trên dãy cửa sổ ngắn, xu hướng nhiệt sẽ được khai thác tốt hơn.
  - Rò rỉ ở ranh TTF: Hiện tại không có cửa sổ nào vượt qua ranh; tuy nhiên, cửa sổ sát ranh vẫn tương quan cao. Chúng tôi sẽ bổ sung phân tích độ nhạy với “temporal gap” (loại bỏ một dải ±ΔTTF quanh ranh, ví dụ 0.5–2.0%) và báo cáo sự thay đổi của Acc/F1 để lượng hoá mức rò rỉ còn lại. Đồng thời, có tùy chọn dùng cửa sổ không chồng lấn trong đánh giá để giảm tương quan.
  - Degrading vs Fault ở cuối đời: Lát cuối đời (70–100.1% TTF) phù hợp kịch bản cảnh báo cận hỏng. Tuy nhiên, để phản ánh vận hành toàn dải, chúng tôi sẽ (i) báo cáo thêm Healthy vs Non‑Healthy ở giai đoạn sớm/trung, (ii) ước lượng FAR ở mức file theo luật cảnh báo bền vững (K‑of‑N, hoặc ngưỡng xác suất + yêu cầu liên tiếp), (iii) hiệu chỉnh/calibration để kiểm soát false alarm. Các số liệu này sẽ được đưa vào phụ lục/time‑metrics theo bin TTF.

  Question (EN): Figure 1 shows Degrading lags (F1≈0.64) vs Fault (1.00). Do you have ablations proving that adding the 6‑D temperature features improves over vibration‑only STFT? Are six indices sufficient to capture “temperature trends”? Around the 60% TTF boundary, overlapping windows may remain highly similar — how do you avoid leakage there; is a temporal gap necessary? Finally, does strong performance on late‑life Degrading vs Fault reflect deployment where Healthy dominates? What is the expected false‑alarm rate if Healthy↔Degrading confusion persists?

  Answer (EN):
  - Ablation: Not included in the current submission; we will add (i) vib‑only, (ii) +temp(6‑D), (iii) +multi‑scale temperature (multi‑horizon slopes/EMA). We expect gains primarily for Degrading (TTF 60–90%). All settings/seeds kept identical; we report file‑wise and TTF‑binned metrics.
  - Sufficiency: The 6‑D set (mean/std/slope) captures coarse per‑window trends; to better encode temperature dynamics we will add multi‑scale slopes, EMA/long‑horizon drift, or STL trend/residuals, and optionally a short‑sequence head (TCN/LSTM/Transformer) on top of the 2D CNN.
  - Boundary leakage: No window spans the boundary, but near‑boundary windows are still correlated. We will include a sensitivity study with a temporal gap (exclude ±ΔTTF around split, e.g., 0.5–2.0%) and report the metric deltas, and optionally evaluate with non‑overlapping windows to further reduce correlation.
  - Late‑life realism vs full‑life: The late‑life slice matches end‑of‑life warning scenarios. For whole‑life operation, we will (i) report Healthy vs Non‑Healthy early/mid‑life performance, (ii) estimate file‑level FAR under persistent alarm logic (K‑of‑N, calibrated thresholds), and (iii) include calibrated confidence to curb false alarms. These will be added as supplemental time‑metrics across TTF bins.
