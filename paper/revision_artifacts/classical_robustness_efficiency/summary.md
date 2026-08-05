# Classical robustness and efficiency

Validation only; missing-temperature imputation was fitted on train features and the locked test split was not instantiated.

| Condition | Accuracy | Macro-F1 | Drop | Healthy F1 | Degrading F1 | Fault F1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| noise_20db | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| noise_10db | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| missing_vib_x | 0.4231 | 0.5054 | 0.4946 | 0.0000 | 0.5161 | 1.0000 |
| missing_vib_y | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| missing_temperature | 0.8077 | 0.5530 | 0.4470 | 0.9091 | 0.7500 | 0.0000 |
| temperature_drift_plus_2c | 0.7308 | 0.5633 | 0.4367 | 0.9677 | 0.2222 | 0.5000 |

## CPU efficiency

- Feature extraction: `2.458 ms/window` (`406.8` windows/s).
- Batched RF inference: `0.0363 ms/window` (`27575.9` windows/s).
- Estimated end-to-end for a 32-window file: `79.82 ms/file` (`12.53` files/s), excluding CSV/NPY I/O.
- Model: `2253.4 KiB`, `200` trees, `25328` total nodes, max depth `12`.

Timing is machine-dependent and was measured on the current CPU; use the JSON artifact for exact protocol fields.
