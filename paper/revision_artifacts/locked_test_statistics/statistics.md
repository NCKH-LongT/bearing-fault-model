# Locked test file-level statistical analysis

Evaluation unit: file; n=27; bootstrap replicates=10000.

| Model | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---:|---:|
| svm_vib8 | 0.8148 [0.6667, 0.9630] | 0.7681 [0.4372, 0.9375] |
| multimodal | 0.5556 [0.4889, 0.6074] | 0.4544 [0.2617, 0.5717] |
| vibration_only | 0.4593 [0.4000, 0.5111] | 0.3314 [0.2611, 0.3713] |

## Paired bootstrap differences (A - B)

### multimodal_minus_vibration_only

- accuracy: 0.0969 [95% CI 0.0148, 0.1852]; P(A>B)=0.9851.
- macro_f1: 0.1019 [95% CI -0.0677, 0.2495]; P(A>B)=0.8848.

### multimodal_minus_svm

- accuracy: -0.2591 [95% CI -0.3852, -0.1259]; P(A>B)=0.0002.
- macro_f1: -0.3063 [95% CI -0.5947, -0.0406]; P(A>B)=0.0078.

McNemar exact results for every seed are stored in `statistics.json`.
