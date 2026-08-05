# Classical feature and algorithm file-CV ablation

All candidates used the same five train-file folds. The global CV winner alone was evaluated on validation; the locked test split was not instantiated.

- Winner: `rf` with `vib_temp_stats_32d` and `depth=12, leaf=1, trees=200`.
- Winner train-CV Macro-F1: `0.9634 ± 0.0337`.
- Winner validation Accuracy: `1.0000`; Macro-F1: `1.0000`.
- Limitation: the manifest contains only `run1`; this remains within-run file-CV.

| Rank | Feature | Model | Settings | Aggregation | CV Macro-F1 | Std | CV Accuracy |
|---:|---|---|---|---|---:|---:|---:|
| 1 | vib_temp_stats_32d | rf | depth=12, leaf=1, trees=200 | mean_probability | 0.9634 | 0.0337 | 0.9608 |
| 2 | vib_temp_stats_32d | rf | depth=12, leaf=2, trees=200 | mean_probability | 0.9337 | 0.0530 | 0.9475 |
| 3 | vib_temp_stats_32d | rf | depth=None, leaf=1, trees=200 | mean_probability | 0.9337 | 0.0530 | 0.9475 |
| 4 | vib_temp_stats_32d | rf | depth=None, leaf=2, trees=200 | mean_probability | 0.9337 | 0.0530 | 0.9475 |
| 5 | vib_temp_stats_32d | logreg | C=1.0 | mean_decision | 0.9298 | 0.0746 | 0.9608 |
| 6 | vib_temp_stats_32d | logreg | C=10.0 | mean_decision | 0.9207 | 0.1194 | 0.9342 |
| 7 | vib_temp_stats_32d | logreg | C=100.0 | mean_decision | 0.9207 | 0.1194 | 0.9342 |
| 8 | vib_temp_stats_32d | svm | C=1.0, gamma=scale | mean_decision | 0.9171 | 0.0942 | 0.9475 |
| 9 | vib_temp_stats_32d | svm | C=10.0, gamma=0.01 | mean_decision | 0.9171 | 0.0942 | 0.9475 |
| 10 | vib_temp_stats_32d | svm | C=100.0, gamma=0.001 | mean_decision | 0.9171 | 0.0942 | 0.9475 |
| 11 | vib_temp_stats_32d | svm | C=100.0, gamma=0.01 | mean_decision | 0.9171 | 0.0942 | 0.9475 |
| 12 | vib_temp_stats_32d | logreg | C=0.1 | mean_decision | 0.9046 | 0.1080 | 0.9350 |
| 13 | vib_temp_stats_32d | svm | C=10.0, gamma=scale | mean_decision | 0.9046 | 0.1080 | 0.9350 |
| 14 | temp_stats_6d | rf | depth=12, leaf=1, trees=200 | mean_probability | 0.8954 | 0.0694 | 0.8825 |
| 15 | temp_stats_6d | svm | C=100.0, gamma=scale | mean_decision | 0.8928 | 0.0330 | 0.8817 |
| 16 | vib_temp_stats_32d | svm | C=100.0, gamma=scale | mean_decision | 0.8916 | 0.0970 | 0.9217 |
| 17 | vib_temp_stats_32d | svm | C=10.0, gamma=0.001 | mean_decision | 0.8896 | 0.1311 | 0.9217 |
| 18 | temp_stats_6d | rf | depth=12, leaf=2, trees=200 | mean_probability | 0.8864 | 0.0853 | 0.8692 |
| 19 | temp_stats_6d | rf | depth=None, leaf=1, trees=200 | mean_probability | 0.8864 | 0.0853 | 0.8692 |
| 20 | temp_stats_6d | rf | depth=None, leaf=2, trees=200 | mean_probability | 0.8864 | 0.0853 | 0.8692 |
| 21 | vib_temp_stats_32d | svm | C=1.0, gamma=0.01 | mean_decision | 0.8765 | 0.1202 | 0.9083 |
| 22 | vib_temp_stats_32d | svm | C=1.0, gamma=0.1 | mean_decision | 0.8750 | 0.0948 | 0.9217 |
| 23 | temp_stats_6d | svm | C=100.0, gamma=0.1 | mean_decision | 0.8735 | 0.0688 | 0.8550 |
| 24 | vib_temp_stats_32d | svm | C=10.0, gamma=0.1 | mean_decision | 0.8620 | 0.0759 | 0.9083 |
| 25 | vib_temp_stats_32d | svm | C=100.0, gamma=0.1 | mean_decision | 0.8620 | 0.0759 | 0.9083 |
| 26 | temp_stats_6d | svm | C=10.0, gamma=scale | mean_decision | 0.8554 | 0.0770 | 0.8300 |
| 27 | vib_stats_26d | svm | C=0.1, gamma=0.1 | mean_decision | 0.8416 | 0.0914 | 0.8950 |
| 28 | vib_temp_stats_32d | svm | C=0.1, gamma=0.1 | mean_decision | 0.8390 | 0.1055 | 0.8942 |
| 29 | vib_temp_stats_32d | svm | C=0.1, gamma=scale | mean_decision | 0.8321 | 0.1136 | 0.8825 |
| 30 | vib_stats_26d | logreg | C=10.0 | mean_decision | 0.8295 | 0.1597 | 0.9208 |
| 31 | vib_temp_stats_32d | svm | C=1.0, gamma=0.001 | mean_decision | 0.8266 | 0.1575 | 0.8833 |
| 32 | temp_stats_6d | svm | C=1.0, gamma=0.1 | mean_decision | 0.8207 | 0.1092 | 0.7775 |
| 33 | temp_stats_6d | svm | C=100.0, gamma=0.01 | mean_decision | 0.8207 | 0.1092 | 0.7775 |
| 34 | temp_stats_6d | svm | C=1.0, gamma=scale | mean_decision | 0.8198 | 0.0900 | 0.7775 |
| 35 | temp_stats_6d | svm | C=10.0, gamma=0.1 | mean_decision | 0.8198 | 0.0900 | 0.7775 |
| 36 | vib_stats_26d | logreg | C=1.0 | mean_decision | 0.8170 | 0.1599 | 0.9083 |
| 37 | vib_stats_26d | svm | C=100.0, gamma=0.001 | mean_decision | 0.8170 | 0.1599 | 0.9083 |
| 38 | vib_stats_8d | svm | C=1.0, gamma=0.1 | mean_decision | 0.8157 | 0.1576 | 0.8958 |
| 39 | vib_stats_8d | svm | C=10.0, gamma=0.1 | mean_decision | 0.8151 | 0.1948 | 0.8833 |
| 40 | vib_temp_stats_32d | svm | C=0.1, gamma=0.01 | mean_decision | 0.8136 | 0.1414 | 0.8700 |
| 41 | vib_stats_8d | svm | C=1.0, gamma=scale | mean_decision | 0.8077 | 0.1740 | 0.8833 |
| 42 | vib_stats_8d | svm | C=10.0, gamma=scale | mean_decision | 0.8031 | 0.2288 | 0.8708 |
| 43 | temp_stats_6d | svm | C=0.1, gamma=scale | mean_decision | 0.8011 | 0.1123 | 0.7775 |
| 44 | vib_stats_26d | svm | C=10.0, gamma=0.01 | mean_decision | 0.7995 | 0.1896 | 0.9217 |
| 45 | vib_stats_26d | svm | C=100.0, gamma=0.01 | mean_decision | 0.7995 | 0.1896 | 0.9217 |
| 46 | vib_stats_26d | svm | C=10.0, gamma=0.001 | mean_decision | 0.7987 | 0.1718 | 0.8958 |
| 47 | temp_stats_6d | logreg | C=10.0 | mean_decision | 0.7963 | 0.1181 | 0.7375 |
| 48 | temp_stats_6d | logreg | C=100.0 | mean_decision | 0.7963 | 0.1181 | 0.7375 |
| 49 | temp_stats_6d | svm | C=0.1, gamma=0.01 | mean_decision | 0.7938 | 0.1420 | 0.7525 |
| 50 | temp_stats_6d | svm | C=1.0, gamma=0.001 | mean_decision | 0.7938 | 0.1420 | 0.7525 |
| 51 | vib_stats_26d | svm | C=1.0, gamma=0.01 | mean_decision | 0.7928 | 0.2035 | 0.8833 |
| 52 | vib_stats_8d | rf | depth=12, leaf=2, trees=200 | mean_probability | 0.7904 | 0.1740 | 0.8700 |
| 53 | vib_stats_8d | rf | depth=None, leaf=1, trees=200 | mean_probability | 0.7904 | 0.1740 | 0.8700 |
| 54 | vib_stats_8d | rf | depth=None, leaf=2, trees=200 | mean_probability | 0.7904 | 0.1740 | 0.8700 |
| 55 | vib_stats_26d | logreg | C=0.1 | mean_decision | 0.7898 | 0.1863 | 0.8825 |
| 56 | vib_stats_26d | svm | C=1.0, gamma=scale | mean_decision | 0.7871 | 0.2071 | 0.9083 |
| 57 | temp_stats_6d | svm | C=10.0, gamma=0.01 | mean_decision | 0.7865 | 0.1038 | 0.7242 |
| 58 | temp_stats_6d | svm | C=100.0, gamma=0.001 | mean_decision | 0.7865 | 0.1038 | 0.7242 |
| 59 | vib_stats_26d | svm | C=0.1, gamma=scale | mean_decision | 0.7797 | 0.1885 | 0.8700 |
| 60 | vib_stats_8d | rf | depth=12, leaf=1, trees=200 | mean_probability | 0.7791 | 0.1961 | 0.8575 |
| 61 | vib_stats_26d | logreg | C=100.0 | mean_decision | 0.7772 | 0.1804 | 0.9208 |
| 62 | vib_stats_26d | svm | C=1.0, gamma=0.1 | mean_decision | 0.7766 | 0.1647 | 0.9083 |
| 63 | vib_stats_26d | svm | C=10.0, gamma=scale | mean_decision | 0.7766 | 0.1647 | 0.9083 |
| 64 | temp_stats_6d | svm | C=0.1, gamma=0.1 | mean_decision | 0.7751 | 0.1510 | 0.7375 |
| 65 | vib_stats_26d | rf | depth=12, leaf=1, trees=200 | mean_probability | 0.7744 | 0.1914 | 0.8958 |
| 66 | vib_stats_26d | rf | depth=12, leaf=2, trees=200 | mean_probability | 0.7744 | 0.1914 | 0.8958 |
| 67 | vib_stats_26d | rf | depth=None, leaf=1, trees=200 | mean_probability | 0.7744 | 0.1914 | 0.8958 |
| 68 | vib_stats_26d | rf | depth=None, leaf=2, trees=200 | mean_probability | 0.7744 | 0.1914 | 0.8958 |
| 69 | vib_stats_8d | logreg | C=1.0 | mean_decision | 0.7742 | 0.1480 | 0.8683 |
| 70 | vib_stats_8d | logreg | C=10.0 | mean_decision | 0.7742 | 0.1480 | 0.8683 |
| 71 | vib_stats_8d | logreg | C=100.0 | mean_decision | 0.7742 | 0.1480 | 0.8683 |
| 72 | vib_stats_26d | svm | C=100.0, gamma=scale | mean_decision | 0.7740 | 0.1919 | 0.8950 |
| 73 | temp_stats_6d | logreg | C=0.1 | mean_decision | 0.7667 | 0.1561 | 0.7242 |
| 74 | temp_stats_6d | logreg | C=1.0 | mean_decision | 0.7667 | 0.1561 | 0.7242 |
| 75 | temp_stats_6d | svm | C=1.0, gamma=0.01 | mean_decision | 0.7653 | 0.1382 | 0.7242 |
| 76 | temp_stats_6d | svm | C=10.0, gamma=0.001 | mean_decision | 0.7653 | 0.1382 | 0.7242 |
| 77 | vib_stats_8d | svm | C=100.0, gamma=0.01 | mean_decision | 0.7543 | 0.1700 | 0.8692 |
| 78 | vib_stats_26d | svm | C=10.0, gamma=0.1 | mean_decision | 0.7508 | 0.2228 | 0.8692 |
| 79 | vib_stats_26d | svm | C=100.0, gamma=0.1 | mean_decision | 0.7508 | 0.2228 | 0.8692 |
| 80 | vib_stats_8d | svm | C=100.0, gamma=0.1 | mean_decision | 0.7466 | 0.2933 | 0.8317 |
| 81 | vib_stats_26d | svm | C=0.1, gamma=0.001 | mean_decision | 0.7455 | 0.2543 | 0.8717 |
| 82 | vib_stats_26d | svm | C=0.1, gamma=0.01 | mean_decision | 0.7213 | 0.2640 | 0.8575 |
| 83 | vib_stats_26d | svm | C=1.0, gamma=0.001 | mean_decision | 0.7213 | 0.2640 | 0.8575 |
| 84 | vib_stats_8d | svm | C=100.0, gamma=scale | mean_decision | 0.7159 | 0.3071 | 0.8175 |
| 85 | vib_temp_stats_32d | svm | C=0.1, gamma=0.001 | mean_decision | 0.7060 | 0.2176 | 0.8450 |
| 86 | vib_stats_8d | logreg | C=0.1 | mean_decision | 0.6979 | 0.1424 | 0.8433 |
| 87 | vib_stats_8d | svm | C=0.1, gamma=0.1 | mean_decision | 0.6930 | 0.2496 | 0.7792 |
| 88 | vib_stats_8d | svm | C=0.1, gamma=scale | mean_decision | 0.6834 | 0.2595 | 0.7658 |
| 89 | vib_stats_8d | svm | C=100.0, gamma=0.001 | mean_decision | 0.6714 | 0.1748 | 0.8300 |
| 90 | vib_stats_8d | svm | C=10.0, gamma=0.01 | mean_decision | 0.6677 | 0.1575 | 0.8300 |
| 91 | vib_stats_8d | svm | C=1.0, gamma=0.01 | mean_decision | 0.6090 | 0.1961 | 0.7667 |
| 92 | vib_stats_8d | svm | C=1.0, gamma=0.001 | mean_decision | 0.5993 | 0.2602 | 0.6475 |
| 93 | vib_stats_8d | svm | C=10.0, gamma=0.001 | mean_decision | 0.5992 | 0.1981 | 0.7533 |
| 94 | vib_stats_8d | svm | C=0.1, gamma=0.01 | mean_decision | 0.5919 | 0.2711 | 0.6350 |
| 95 | temp_stats_6d | svm | C=0.1, gamma=0.001 | mean_decision | 0.4356 | 0.1726 | 0.6450 |
| 96 | vib_stats_8d | svm | C=0.1, gamma=0.001 | mean_decision | 0.4356 | 0.1726 | 0.6450 |
