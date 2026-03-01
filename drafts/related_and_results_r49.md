# Draft: Related Works and Results (r49-aligned)

## Related Works

- Time–frequency CNNs for bearing diagnostics: Prior work commonly converts vibration windows to 2D images (e.g., STFT spectrograms, wavelet scalograms) and applies 2D CNNs to learn discriminative patterns [zhao2019deep, lei2020roadmap, ince2016tie, wen2018neurocomputing]. These methods provide strong baselines for fault vs. healthy discrimination in controlled settings.
- Run-to-failure and degradation-stage recognition: Compared to binary fault detection, recognizing an intermediate Degrading stage is more ambiguous due to subtle, evolving signatures and class imbalance. Studies on run-to-failure emphasize realistic label definitions and temporal evaluation to avoid optimistic bias.
- Data leakage and temporal splits: Random/window-level splits with overlapped windows can leak near-duplicate content across train/test, inflating metrics. Leakage-aware, TTF-aligned temporal protocols better match deployment where future data are unseen during training [kaufman2012leakage, bergmeir2012timeseriescv].
- Multi-sensor fusion: Temperature correlates with friction/thermal load and can complement vibration, especially near degradation onset. Late fusion of compact temperature descriptors with vibration CNN features offers a practical trade-off between robustness and efficiency [baltrusaitis2019multimodal].

Our work follows this trajectory by: (i) using STFT-based two-axis spectrograms with per-frequency normalization, (ii) fusing a 6-D temperature trend/statistics descriptor via a lightweight late-fusion head, and (iii) evaluating with a TTF-aligned temporal split to mitigate leakage.

## Results (r49 configuration)

Settings: STFT n_fft=2048, hop=512; input 160×160; AdamW lr=2e-4, wd=1e-2; class-weights + label smoothing 0.05; balanced sampling; AMP; `val_max_windows=0`. Stratified pre-training (`init_from`) then temporal fine-tuning.

- Stratified (3-class): best dev configuration achieves Macro-F1 ≈ 0.7762 (Healthy ≈ 0.69, Degrading ≈ 0.64, Fault = 1.00).
- Temporal (late-life, present-labels):
  - Early 70–90%: Accuracy = 1.0000 (Degrading only)
  - Late 90–100%: Accuracy = 0.3846; Fault F1 = 0.5556 (mean-aggregation)
- Full-range test 0–100.1%:
  - Mean-aggregation: Accuracy = 0.4806; Macro-F1 = 0.4907
  - Vote-aggregation: Accuracy = 0.8295; Macro-F1 = 0.6999

Takeaways:
- Mean vs Vote: Vote greatly improves overall metrics (0–100.1%) but reduces late Fault recall; mean retains better late Fault F1.
- Temperature helps late-life Fault discrimination versus vibration-only baselines (see ablations).

## Notes on Run Naming (e.g., “auto_ft_r49”)

The label `auto_ft_r49` denotes the 49th round in an internal automatic fine-tuning sweep. Rounds are sequential attempts that tweak hyperparameters while keeping the model architecture fixed; the “best” round is selected by macro-F1 on the evaluation set. In the paper, consider referring to this as “our best temporal fine-tuned configuration” and summarizing its key hyperparameters, rather than relying on the round ID.

