# Deep multimodal robustness and efficiency

Five-seed validation only; raw-signal perturbations were applied before STFT and the locked test split was not instantiated.

| Condition | Accuracy mean ± std | Macro-F1 mean ± std | Drop | H F1 | D F1 | F F1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| noise_20db | 0.1154 ± 0.0000 | 0.0690 ± 0.0000 | 0.9310 | 0.0000 | 0.0000 | 0.2069 |
| noise_10db | 0.1154 ± 0.0000 | 0.0690 ± 0.0000 | 0.9310 | 0.0000 | 0.0000 | 0.2069 |
| missing_vib_x | 0.2846 ± 0.1003 | 0.1946 ± 0.1388 | 0.8054 | 0.0000 | 0.3824 | 0.2014 |
| missing_vib_y | 0.4538 ± 0.1663 | 0.3506 ± 0.1348 | 0.6494 | 0.3038 | 0.2881 | 0.4600 |
| missing_temperature | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| temperature_drift_plus_2c | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

## Efficiency (seed 42)

- CPU STFT + temperature preprocessing: `7.177 ms/window`.
- cuda model forward: `0.034 ms/window` (`29481.5` windows/s), batch 32.
- Estimated 32-window end-to-end: `230.74 ms/file` (`4.33` files/s), excluding I/O.
- Parameters: `2798403`; checkpoint: `10.73 MiB`.
- Peak CUDA allocated memory: `97.96 MiB`.
