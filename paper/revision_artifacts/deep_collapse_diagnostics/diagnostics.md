# Deep model collapse diagnostics

## multimodal

| Seed | Pred H/D/F | Missing class | Confidence | Entropy | Best val F1 | Test F1 | Gap |
|---:|---:|---|---:|---:|---:|---:|---:|
| 42 | 0/25/2 | healthy | 0.4660 | 0.9565 | 0.9670 | 0.4283 | 0.5387 |
| 43 | 25/0/2 | degrading | 0.3942 | 0.9903 | 1.0000 | 0.5268 | 0.4732 |
| 44 | 0/26/1 | healthy | 0.3743 | 0.9942 | 1.0000 | 0.3235 | 0.6765 |
| 45 | 21/5/1 | - | 0.4245 | 0.9523 | 1.0000 | 0.6088 | 0.3912 |
| 46 | 23/4/0 | fault | 0.4764 | 0.9468 | 1.0000 | 0.3846 | 0.6154 |

Unanimous files across five seeds: 0/27.

## vibration_only

| Seed | Pred H/D/F | Missing class | Confidence | Entropy | Best val F1 | Test F1 | Gap |
|---:|---:|---|---:|---:|---:|---:|---:|
| 42 | 10/17/0 | fault | 0.3845 | 0.9877 | 1.0000 | 0.4697 | 0.5303 |
| 43 | 6/18/3 | - | 0.3505 | 0.9984 | 1.0000 | 0.7203 | 0.2797 |
| 44 | 27/0/0 | degrading, fault | 0.3776 | 0.9954 | 1.0000 | 0.2481 | 0.7519 |
| 45 | 0/27/0 | healthy, fault | 0.3760 | 0.9956 | 1.0000 | 0.1524 | 0.8476 |
| 46 | 0/0/27 | healthy, degrading | 0.3943 | 0.9924 | 1.0000 | 0.0667 | 0.9333 |

Unanimous files across five seeds: 0/27.

## Temperature feature diagnostics

Raw train feature std scale ratio (max/min): 2768.14.

Standardized split-mean shift relative to train:

- val: bearing_mean=0.01, bearing_std=0.24, bearing_slope=0.21, ambient_mean=0.14, ambient_std=0.02, ambient_slope=-0.05
- test: bearing_mean=-0.23, bearing_std=-0.09, bearing_slope=-0.12, ambient_mean=0.08, ambient_std=-0.03, ambient_slope=0.02

## Evidence-based interpretation

- Validation Macro-F1 is saturated while held-out test Macro-F1 drops sharply, indicating split sensitivity and model-selection overfitting to a 26-file validation set.
- Several seeds never predict one or more classes, confirming genuine class collapse rather than a small metric fluctuation.
- Temperature descriptors enter the linear branch in raw physical units without a train-fitted scaler; feature magnitudes and split shifts can destabilize fusion.
- One-second temperature slope/std features are small relative to raw means and may be dominated without normalization or longer causal context.
- The next model experiment must be validation/group-CV only: train-fitted temperature normalization plus a temperature-only baseline before gated fusion.
