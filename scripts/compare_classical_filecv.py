#!/usr/bin/env python3
"""Compare handcrafted features and classical models using fixed train file-folds."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.search_classical_filecv import file_features, fit_files, predict_files, score


CLASS_NAMES = ("healthy", "degrading", "fault")
FEATURE_NAMES = ("vib_stats_8d", "vib_stats_26d", "vib_temp_stats_32d")


def candidates() -> list[dict]:
    rows = []
    for feature in FEATURE_NAMES:
        for c_value, gamma in itertools.product((0.1, 1.0, 10.0, 100.0), ("scale", 0.001, 0.01, 0.1)):
            rows.append({"feature": feature, "model": "svm", "C": c_value, "gamma": gamma})
        for c_value in (0.1, 1.0, 10.0, 100.0):
            rows.append({"feature": feature, "model": "logreg", "C": c_value})
        for max_depth, min_leaf in itertools.product((None, 12), (1, 2)):
            rows.append({"feature": feature, "model": "rf", "max_depth": max_depth, "min_samples_leaf": min_leaf})
    return rows


def build_model(spec: dict, seed: int) -> Pipeline:
    if spec["model"] == "svm":
        classifier = SVC(
            C=spec["C"], gamma=spec["gamma"], kernel="rbf", class_weight="balanced", random_state=seed
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])
    if spec["model"] == "logreg":
        classifier = LogisticRegression(
            C=spec["C"], max_iter=3000, class_weight="balanced", random_state=seed
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=spec["max_depth"],
        min_samples_leaf=spec["min_samples_leaf"],
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("clf", classifier)])


def spec_text(spec: dict) -> str:
    if spec["model"] == "svm":
        return f"C={spec['C']}, gamma={spec['gamma']}"
    if spec["model"] == "logreg":
        return f"C={spec['C']}"
    return f"depth={spec['max_depth']}, leaf={spec['min_samples_leaf']}, trees=200"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="classical_baselines/configs/revision_svm_vib8_stratified.yaml")
    parser.add_argument("--output-dir", default="paper/revision_artifacts/classical_feature_filecv")
    parser.add_argument("--run-dir", default="runs/revision_classical_feature_filecv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-windows-per-file", type=int, default=32)
    args = parser.parse_args()

    base = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    feature_data = {}
    for feature in FEATURE_NAMES:
        cfg = copy.deepcopy(base)
        cfg["classical"]["feature_name"] = feature
        feature_data[feature] = {
            "train": file_features(cfg, "train", args.max_windows_per_file),
            "val": file_features(cfg, "val", args.max_windows_per_file),
        }
    labels = np.asarray([row["label"] for row in feature_data["vib_stats_8d"]["train"]], dtype=int)
    split_indices = list(StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=20260805).split(np.zeros(len(labels)), labels))

    leaderboard = []
    for spec in candidates():
        train_files = feature_data[spec["feature"]]["train"]
        folds = []
        for fold, (fit_indices, eval_indices) in enumerate(split_indices, start=1):
            model = build_model(spec, 20260805 + fold)
            fit_files(model, [train_files[index] for index in fit_indices])
            truth, predicted, _ = predict_files(model, [train_files[index] for index in eval_indices])
            folds.append(score(truth, predicted))
        leaderboard.append({
            **spec,
            "settings": spec_text(spec),
            "aggregation": "mean_decision" if spec["model"] in {"svm", "logreg"} else "mean_probability",
            "cv_macro_f1_mean": float(np.mean([row["macro_f1"] for row in folds])),
            "cv_macro_f1_std": float(np.std([row["macro_f1"] for row in folds], ddof=1)),
            "cv_accuracy_mean": float(np.mean([row["accuracy"] for row in folds])),
            "folds": folds,
        })
    leaderboard.sort(key=lambda row: (-row["cv_macro_f1_mean"], row["cv_macro_f1_std"], -row["cv_accuracy_mean"], row["feature"], row["model"], row["settings"]))
    winner = leaderboard[0]
    final_model = build_model(winner, 20260805)
    fit_files(final_model, feature_data[winner["feature"]]["train"])
    val_truth, val_predicted, val_records = predict_files(final_model, feature_data[winner["feature"]]["val"])
    validation = score(val_truth, val_predicted)

    run_dir = ROOT / args.run_dir
    output_dir = ROOT / args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "model.pkl").open("wb") as stream:
        pickle.dump(final_model, stream)
    with (run_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("file_id", "y_true", "y_pred", "n_windows", "score_0", "score_1", "score_2")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(val_records)
    np.savetxt(run_dir / "validation_confusion_matrix.csv", confusion_matrix(val_truth, val_predicted, labels=[0, 1, 2]), fmt="%d", delimiter=",")
    validation_report = classification_report(val_truth, val_predicted, labels=[0, 1, 2], target_names=CLASS_NAMES, digits=4, zero_division=0)
    (run_dir / "validation_report.txt").write_text(validation_report + "\n", encoding="utf-8")
    result = {
        "protocol": "fixed five-fold stratified file-grouped CV on train; global CV winner evaluated once on validation; locked test not instantiated",
        "limitation": "All files belong to run1; results estimate within-run file generalization only.",
        "max_windows_per_file": args.max_windows_per_file,
        "winner": winner,
        "validation": validation,
        "leaderboard": leaderboard,
    }
    result_text = json.dumps(result, indent=2) + "\n"
    (run_dir / "results.json").write_text(result_text, encoding="utf-8")
    (output_dir / "results.json").write_text(result_text, encoding="utf-8")
    lines = [
        "# Classical feature and algorithm file-CV ablation",
        "",
        "All candidates used the same five train-file folds. The global CV winner alone was evaluated on validation; the locked test split was not instantiated.",
        "",
        f"- Winner: `{winner['model']}` with `{winner['feature']}` and `{winner['settings']}`.",
        f"- Winner train-CV Macro-F1: `{winner['cv_macro_f1_mean']:.4f} ± {winner['cv_macro_f1_std']:.4f}`.",
        f"- Winner validation Accuracy: `{validation['accuracy']:.4f}`; Macro-F1: `{validation['macro_f1']:.4f}`.",
        "- Limitation: the manifest contains only `run1`; this remains within-run file-CV.",
        "",
        "| Rank | Feature | Model | Settings | Aggregation | CV Macro-F1 | Std | CV Accuracy |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(leaderboard, start=1):
        lines.append(
            f"| {rank} | {row['feature']} | {row['model']} | {row['settings']} | {row['aggregation']} | "
            f"{row['cv_macro_f1_mean']:.4f} | {row['cv_macro_f1_std']:.4f} | {row['cv_accuracy_mean']:.4f} |"
        )
    summary = "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
