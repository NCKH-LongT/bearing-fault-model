Q3 Submission Readiness — Action Plan (TODO)

Goal: Elevate the manuscript and artifacts to a solid Q3-level submission with stronger generalization evidence, rigorous statistics, clearer metrics definition, and full reproducibility.

High Priority (pre-submission)
- External generalization
  - [ ] Add cross-run/cross-bearing evaluation on the same dataset if available.
  - [ ] Add at least one external dataset (e.g., Paderborn/CWRU/XJTU-SY) for transfer test (train on Jung et al., test on external; or fine-tune few-shot).
  - [ ] Report ≥5 seeds with mean ± std for all key metrics; include 95% CI.
  - [ ] Add paired statistical significance (e.g., McNemar’s test) vs. main baselines on file-level predictions.

- Strong baselines and variants
  - [ ] Implement stronger baselines: 1D CNN and/or TCN/LSTM on raw vibration; ResNet18 on STFT; a simple temporal transformer; SSL pretrain (e.g., SimCLR/CPC) + linear probe.
  - [ ] Compare fusion strategies: early, mid, late, and gating/attention; keep params similar where possible.
  - [ ] Ensure vib-only strong baseline is included alongside multi-modal.

- Present-class metrics — formalization and clarity
  - [ ] Add a formal definition for “present-class” metrics and handling absent classes.
  - [ ] Provide pseudo-code for per-file mean-logit aggregation, present-class precision/recall/F1.
  - [ ] Include a small worked example in the appendix to avoid ambiguity.

- Stage thresholding and boundary analysis
  - [ ] Add sensitivity analysis for stage thresholds (TTF%) around Healthy–Degrading boundary (e.g., sweep 0.55–0.70).
  - [ ] Consider an ordinal formulation or a regression-to-TTF auxiliary loss for robustness; discuss pros/cons.
  - [ ] Tie thresholds to operational decisions (alarms near late life) with justification.

- Efficiency and resource profile
  - [ ] Report parameter count, FLOPs (e.g., ptflops/torchprofile), and inference latency (ms/window) on RTX 2060.
  - [ ] Provide a compact table comparing efficiency vs. accuracy across baselines.

- Reproducibility and artifacts
  - [ ] Replace “Code will be released upon acceptance” with an anonymized repository or supplemental zip for review.
  - [ ] Include exact configs, preprocessing scripts, and a minimal downloadable checkpoint.
  - [ ] Pin package versions (requirements.txt/conda env); include CUDA/PyTorch versions.
  - [ ] Provide a one-command run script for dev split and temporal split (train/eval) with seeds.
  - [ ] Document deterministic flags (cudnn.benchmark = False, cudnn.deterministic = True where applicable) and any caveats.

- Figures and tables
  - [ ] Complete ablation table with missing cells or mark N/A explicitly; clarify when metrics are single-class.
  - [ ] Add PR/ROC curves for Fault class (temporal slices) and calibration (reliability/ECE) plots.
  - [ ] Add representative spectrograms (Healthy/Degrading/Fault) and temperature trend exemplars.
  - [ ] Ensure all referenced figures exist and paths match LaTeX.

- Writing, formatting, and references
  - [ ] Expand Related Work (fusion for PHM; temporal leakage-aware evaluation) with 6–10 targeted citations.
  - [ ] Rename “Related Works” → “Related Work”; remove Vietnamese comments in TikZ blocks.
  - [ ] Clarify limitations (single-run) and how new experiments mitigate them.
  - [ ] Audit refs.bib for consistency (year/volume/pages/doi), remove future-dated urldate; ensure capitalization where needed.

Medium Priority
- Error analysis and interpretability
  - [ ] Analyze misclassifications near 60–70% TTF; plot TTF vs. class prob; confusion over time.
  - [ ] Add saliency/Grad-CAM on spectrograms or SHAP on temperature features.

- Ablation completeness
  - [ ] Summarize effects of: class-weights on/off, spectrogram resolution, per-frequency normalization on/off, window length/overlap variations.
  - [ ] Brief hyperparameter sweep section with top 3 sensitivities.

- Robustness tests
  - [ ] Noise/operating-condition shifts (SNR, speed/load if available) to show multi-modal resilience.

Low Priority / Nice to have
- [ ] Minimal demo notebook (file-wise inference, plots) and small UI screenshot.
- [ ] Simple uncertainty thresholding for deployment (reject option) and ablation.
- [ ] Ablation of temperature descriptor dimensionality and smoothing.

File-level Edits (manuscript)
- main.tex
  - [ ] Related Work title fix and expansion: main.tex:56.
  - [ ] Define present-class metrics with formula and pseudo-code near file-wise inference: main.tex:125–129 or before Results main.tex:243.
  - [ ] Complete Table (Ablation): main.tex:299–323 — fill missing cells or mark “—”; note single-class cases.
  - [ ] Add efficiency table and short subsection (Complexity): near Experimental Setup main.tex:167 or create a new subsection.
  - [ ] Remove VN comments in TikZ and unify English annotations: e.g., main.tex:201–212, 222–231.
  - [ ] Replace “will be released upon acceptance” with anonymized link: main.tex:365.

- refs.bib
  - [ ] Verify entries with in-press years (e.g., Information Fusion 2025) and DOIs; adjust urldate (avoid future dates), e.g., refs.bib:204–212.

- Code/configs
  - [ ] Ensure referenced config exists and is final: configs/logs_stft_full_temporal_gating.yaml (mentioned in main.tex:167).
  - [ ] Parameterize stage thresholds in datasets/logs_ttf.py; expose CLI flags and log into run metadata.
  - [ ] Add scripts: train_dev.sh/.ps1, train_temporal.sh/.ps1, eval_temporal.sh/.ps1 with seed loops.

- Figures
  - [ ] Verify all figure paths exist (checked: figures/temporal/*, figures/fullrange/*). Add missing PR/ROC/calibration figures.

Submission Package Checklist
- [ ] Conform to Springer Nature sn-jnl template page limits; microtype/xurl already set.
- [ ] PDF compiles without overfull boxes/warnings; clean auxiliary files.
- [ ] Cover letter: highlight leakage-aware evaluation, multi-modal gains, and practical deployment framing.
- [ ] Anonymized code/supplemental materials uploaded; README with exact commands.
- [ ] Data availability statement points to stable DOI/URL; include exact subset, RUNS/FILES details.

Suggested Timeline
- Week 1: Baselines (1D CNN, ResNet18), present-class formalization, ablation table cleanup.
- Week 2: Cross-run/external dataset, multi-seed runs, efficiency profiling.
- Week 3: Figures (PR/ROC, calibration), Related Work expansion, refs audit.
- Week 4: Reproducible repo (anon), final polish, compile and submit.

Owners (fill in)
- Experiments: [...]
- Writing: [...]
- Code/release: [...]
- Figures: [...]

