# Train sampler and gradient audit

Train split only; the locked test split was not instantiated.

| Class | Train files | Empirical balanced draws | Draw share |
|---|---:|---:|---:|
| healthy | 46 | 3411 | 0.3411 |
| degrading | 23 | 3254 | 0.3254 |
| fault | 7 | 3335 | 0.3335 |

| Class | Files | Loss | Vibration grad RMS | Temperature grad RMS | Classifier grad RMS |
|---|---:|---:|---:|---:|---:|
| healthy | 8 | 1.5331 | 0.010792 | 0.008796 | 0.357315 |
| degrading | 8 | 1.7142 | 0.007604 | 0.012756 | 0.382338 |
| fault | 7 | 0.9711 | 0.009163 | 0.030088 | 0.349172 |
