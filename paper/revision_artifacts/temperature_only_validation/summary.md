# Temperature-only validation

Exploratory validation-only experiment; the locked test split was not evaluated.

| Metric | Mean | Std |
|---|---:|---:|
| val_accuracy | 0.7385 | 0.0322 |
| val_macro_f1 | 0.7776 | 0.0264 |
| f1_healthy | 0.7704 | 0.0257 |
| f1_degrading | 0.5910 | 0.0760 |
| f1_fault | 0.9714 | 0.0639 |

| Seed | Best epoch | Accuracy | Macro-F1 | H F1 | D F1 | F F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 20 | 0.7692 | 0.8175 | 0.7857 | 0.6667 | 1.0000 |
| 43 | 19 | 0.7308 | 0.7692 | 0.7742 | 0.5333 | 1.0000 |
| 44 | 18 | 0.7308 | 0.7823 | 0.7586 | 0.5882 | 1.0000 |
| 45 | 28 | 0.6923 | 0.7444 | 0.7333 | 0.5000 | 1.0000 |
| 46 | 7 | 0.7692 | 0.7746 | 0.8000 | 0.6667 | 0.8571 |
