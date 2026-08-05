# Temperature proxy audit

All analyses use train-file folds only; validation and the locked test split were not instantiated.

Baseline multimodal CV Macro-F1: `0.9634 ± 0.0337`.

## File-grouped modality permutation

| Permuted modality | Macro-F1 after permutation | Repeat std | Macro-F1 drop |
|---|---:|---:|---:|
| vibration_26d | 0.6533 | 0.0426 | 0.3101 |
| temperature_6d | 0.6144 | 0.0507 | 0.3489 |

## Temperature association with TTF

| Descriptor | Spearman rho | p-value |
|---|---:|---:|
| bearing_mean | 0.8714 | 1.362e-24 |
| bearing_std | 0.7988 | 5.359e-18 |
| bearing_slope | 0.7390 | 2.516e-14 |
| ambient_mean | 0.5823 | 3.445e-08 |
| ambient_std | -0.2192 | 5.705e-02 |
| ambient_slope | 0.0592 | 6.117e-01 |

## Interpretation

- A large drop after file-grouped temperature permutation means the model materially relies on temperature beyond vibration within this run.
- Strong temperature–TTF correlation also means temperature may encode trajectory position; this cannot be separated from transferable degradation signal with only run1.
- Do not claim cross-run sensor-fusion generalization until an independent run/bearing is evaluated.
