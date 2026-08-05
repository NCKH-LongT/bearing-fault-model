# Classical feature and algorithm file-CV ablation

All candidates used the same five train-file folds. The global CV winner alone was evaluated on validation; the locked test split was not instantiated.

- Winner: `svm` with `vib_stats_26d` and `C=0.1, gamma=0.1`.
- Winner train-CV Macro-F1: `0.8416 ± 0.0914`.
- Winner validation Accuracy: `1.0000`; Macro-F1: `1.0000`.
- Limitation: the manifest contains only `run1`; this remains within-run file-CV.

| Rank | Feature | Model | Settings | Aggregation | CV Macro-F1 | Std | CV Accuracy |
|---:|---|---|---|---|---:|---:|---:|
| 1 | vib_stats_26d | svm | C=0.1, gamma=0.1 | mean_decision | 0.8416 | 0.0914 | 0.8950 |
| 2 | vib_stats_26d | logreg | C=10.0 | mean_decision | 0.8295 | 0.1597 | 0.9208 |
| 3 | vib_stats_26d | logreg | C=1.0 | mean_decision | 0.8170 | 0.1599 | 0.9083 |
| 4 | vib_stats_26d | svm | C=100.0, gamma=0.001 | mean_decision | 0.8170 | 0.1599 | 0.9083 |
| 5 | vib_stats_8d | svm | C=1.0, gamma=0.1 | mean_decision | 0.8157 | 0.1576 | 0.8958 |
| 6 | vib_stats_8d | svm | C=10.0, gamma=0.1 | mean_decision | 0.8151 | 0.1948 | 0.8833 |
| 7 | vib_stats_8d | svm | C=1.0, gamma=scale | mean_decision | 0.8077 | 0.1740 | 0.8833 |
| 8 | vib_stats_8d | svm | C=10.0, gamma=scale | mean_decision | 0.8031 | 0.2288 | 0.8708 |
| 9 | vib_stats_26d | svm | C=10.0, gamma=0.01 | mean_decision | 0.7995 | 0.1896 | 0.9217 |
| 10 | vib_stats_26d | svm | C=100.0, gamma=0.01 | mean_decision | 0.7995 | 0.1896 | 0.9217 |
| 11 | vib_stats_26d | svm | C=10.0, gamma=0.001 | mean_decision | 0.7987 | 0.1718 | 0.8958 |
| 12 | vib_stats_26d | svm | C=1.0, gamma=0.01 | mean_decision | 0.7928 | 0.2035 | 0.8833 |
| 13 | vib_stats_8d | rf | depth=12, leaf=2, trees=200 | mean_probability | 0.7904 | 0.1740 | 0.8700 |
| 14 | vib_stats_8d | rf | depth=None, leaf=1, trees=200 | mean_probability | 0.7904 | 0.1740 | 0.8700 |
| 15 | vib_stats_8d | rf | depth=None, leaf=2, trees=200 | mean_probability | 0.7904 | 0.1740 | 0.8700 |
| 16 | vib_stats_26d | logreg | C=0.1 | mean_decision | 0.7898 | 0.1863 | 0.8825 |
| 17 | vib_stats_26d | svm | C=1.0, gamma=scale | mean_decision | 0.7871 | 0.2071 | 0.9083 |
| 18 | vib_stats_26d | svm | C=0.1, gamma=scale | mean_decision | 0.7797 | 0.1885 | 0.8700 |
| 19 | vib_stats_8d | rf | depth=12, leaf=1, trees=200 | mean_probability | 0.7791 | 0.1961 | 0.8575 |
| 20 | vib_stats_26d | logreg | C=100.0 | mean_decision | 0.7772 | 0.1804 | 0.9208 |
| 21 | vib_stats_26d | svm | C=1.0, gamma=0.1 | mean_decision | 0.7766 | 0.1647 | 0.9083 |
| 22 | vib_stats_26d | svm | C=10.0, gamma=scale | mean_decision | 0.7766 | 0.1647 | 0.9083 |
| 23 | vib_stats_26d | rf | depth=12, leaf=1, trees=200 | mean_probability | 0.7744 | 0.1914 | 0.8958 |
| 24 | vib_stats_26d | rf | depth=12, leaf=2, trees=200 | mean_probability | 0.7744 | 0.1914 | 0.8958 |
| 25 | vib_stats_26d | rf | depth=None, leaf=1, trees=200 | mean_probability | 0.7744 | 0.1914 | 0.8958 |
| 26 | vib_stats_26d | rf | depth=None, leaf=2, trees=200 | mean_probability | 0.7744 | 0.1914 | 0.8958 |
| 27 | vib_stats_8d | logreg | C=1.0 | mean_decision | 0.7742 | 0.1480 | 0.8683 |
| 28 | vib_stats_8d | logreg | C=10.0 | mean_decision | 0.7742 | 0.1480 | 0.8683 |
| 29 | vib_stats_8d | logreg | C=100.0 | mean_decision | 0.7742 | 0.1480 | 0.8683 |
| 30 | vib_stats_26d | svm | C=100.0, gamma=scale | mean_decision | 0.7740 | 0.1919 | 0.8950 |
| 31 | vib_stats_8d | svm | C=100.0, gamma=0.01 | mean_decision | 0.7543 | 0.1700 | 0.8692 |
| 32 | vib_stats_26d | svm | C=10.0, gamma=0.1 | mean_decision | 0.7508 | 0.2228 | 0.8692 |
| 33 | vib_stats_26d | svm | C=100.0, gamma=0.1 | mean_decision | 0.7508 | 0.2228 | 0.8692 |
| 34 | vib_stats_8d | svm | C=100.0, gamma=0.1 | mean_decision | 0.7466 | 0.2933 | 0.8317 |
| 35 | vib_stats_26d | svm | C=0.1, gamma=0.001 | mean_decision | 0.7455 | 0.2543 | 0.8717 |
| 36 | vib_stats_26d | svm | C=0.1, gamma=0.01 | mean_decision | 0.7213 | 0.2640 | 0.8575 |
| 37 | vib_stats_26d | svm | C=1.0, gamma=0.001 | mean_decision | 0.7213 | 0.2640 | 0.8575 |
| 38 | vib_stats_8d | svm | C=100.0, gamma=scale | mean_decision | 0.7159 | 0.3071 | 0.8175 |
| 39 | vib_stats_8d | logreg | C=0.1 | mean_decision | 0.6979 | 0.1424 | 0.8433 |
| 40 | vib_stats_8d | svm | C=0.1, gamma=0.1 | mean_decision | 0.6930 | 0.2496 | 0.7792 |
| 41 | vib_stats_8d | svm | C=0.1, gamma=scale | mean_decision | 0.6834 | 0.2595 | 0.7658 |
| 42 | vib_stats_8d | svm | C=100.0, gamma=0.001 | mean_decision | 0.6714 | 0.1748 | 0.8300 |
| 43 | vib_stats_8d | svm | C=10.0, gamma=0.01 | mean_decision | 0.6677 | 0.1575 | 0.8300 |
| 44 | vib_stats_8d | svm | C=1.0, gamma=0.01 | mean_decision | 0.6090 | 0.1961 | 0.7667 |
| 45 | vib_stats_8d | svm | C=1.0, gamma=0.001 | mean_decision | 0.5993 | 0.2602 | 0.6475 |
| 46 | vib_stats_8d | svm | C=10.0, gamma=0.001 | mean_decision | 0.5992 | 0.1981 | 0.7533 |
| 47 | vib_stats_8d | svm | C=0.1, gamma=0.01 | mean_decision | 0.5919 | 0.2711 | 0.6350 |
| 48 | vib_stats_8d | svm | C=0.1, gamma=0.001 | mean_decision | 0.4356 | 0.1726 | 0.6450 |
