# SVM file-grouped CV and validation

Five-fold stratified file-grouped CV was performed on train files only. The selected model was evaluated once on validation; the locked test split was not instantiated.

- Winner: `C=1.0`, `gamma=0.1`, mean decision-score aggregation.
- Train CV Macro-F1: `0.8157 ± 0.1576`.
- Validation Accuracy: `0.8462`.
- Validation Macro-F1: `0.8631`.
- Validation class F1: Healthy `0.8750`, Degrading `0.7143`, Fault `1.0000`.
- Limitation: all files belong to `run1`; this is within-run file-grouped CV, not cross-run/cross-bearing validation.

| Rank | C | Gamma | CV Macro-F1 mean | CV std | CV Accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 0.1 | 0.8157 | 0.1576 | 0.8958 |
| 2 | 10.0 | 0.1 | 0.8151 | 0.1948 | 0.8833 |
| 3 | 1.0 | scale | 0.8077 | 0.1740 | 0.8833 |
| 4 | 10.0 | scale | 0.8031 | 0.2288 | 0.8708 |
| 5 | 100.0 | 0.01 | 0.7543 | 0.1700 | 0.8692 |
| 6 | 100.0 | 0.1 | 0.7466 | 0.2933 | 0.8317 |
| 7 | 100.0 | scale | 0.7159 | 0.3071 | 0.8175 |
| 8 | 0.1 | 0.1 | 0.6930 | 0.2496 | 0.7792 |
| 9 | 0.1 | scale | 0.6834 | 0.2595 | 0.7658 |
| 10 | 100.0 | 0.001 | 0.6714 | 0.1748 | 0.8300 |
| 11 | 10.0 | 0.01 | 0.6677 | 0.1575 | 0.8300 |
| 12 | 1.0 | 0.01 | 0.6090 | 0.1961 | 0.7667 |
| 13 | 1.0 | 0.001 | 0.5993 | 0.2602 | 0.6475 |
| 14 | 10.0 | 0.001 | 0.5992 | 0.1981 | 0.7533 |
| 15 | 0.1 | 0.01 | 0.5919 | 0.2711 | 0.6350 |
| 16 | 0.1 | 0.001 | 0.4356 | 0.1726 | 0.6450 |
