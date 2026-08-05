#!/usr/bin/env python3
"""File-grouped SVM selection on train, followed by validation-only evaluation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classical_baselines.pipeline import build_split_items, extract_window_features


CLASS_NAMES = ("healthy", "degrading", "fault")


def file_features(cfg: dict, split: str, max_windows: int) -> list[dict]:
    win = int(round(float(cfg["window_seconds"]) * int(cfg["sampling_rate"])))
    hop = int(round(float(cfg["hop_seconds"]) * int(cfg["sampling_rate"])))
    rows = []
    for item in build_split_items(cfg, split):
        matrix = extract_window_features(
            item,
            win,
            hop,
            cfg["classical"]["feature_name"],
            (cfg.get("debug", {}) or {}).get("seconds_cap"),
            int(cfg["sampling_rate"]),
            cfg.get("cache_dir"),
            max_windows=max_windows,
        )
        rows.append({"file_id": item["file"], "label": int(item["label"]), "features": matrix})
    return rows


def build_model(c_value: float, gamma: str | float, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=c_value, gamma=gamma, kernel="rbf", class_weight="balanced", random_state=seed)),
    ])


def fit_files(model: Pipeline, files: list[dict]) -> None:
    matrix = np.concatenate([row["features"] for row in files], axis=0)
    labels = np.concatenate([np.full(len(row["features"]), row["label"], dtype=int) for row in files])
    model.fit(matrix, labels)


def predict_files(model: Pipeline, files: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    truth, predicted, records = [], [], []
    for row in files:
        scores = np.asarray(model.decision_function(row["features"]), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        mean_score = scores.mean(axis=0)
        prediction = int(model.classes_[int(np.argmax(mean_score))])
        truth.append(row["label"])
        predicted.append(prediction)
        records.append({
            "file_id": row["file_id"],
            "y_true": row["label"],
            "y_pred": prediction,
            "n_windows": len(row["features"]),
            **{f"score_{label}": float(mean_score[index]) for index, label in enumerate(model.classes_)},
        })
    return np.asarray(truth), np.asarray(predicted), records


def score(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
    }


def parse_gamma(value: str) -> str | float:
    return value if value == "scale" else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="classical_baselines/configs/revision_svm_vib8_stratified.yaml")
    parser.add_argument("--output-dir", default="paper/revision_artifacts/svm_filecv_validation")
    parser.add_argument("--run-dir", default="runs/revision_svm_filecv_validation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-windows-per-file", type=int, default=32)
    parser.add_argument("--c-values", default="0.1,1,10,100")
    parser.add_argument("--gammas", default="scale,0.001,0.01,0.1")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    train_files = file_features(cfg, "train", args.max_windows_per_file)
    val_files = file_features(cfg, "val", args.max_windows_per_file)
    file_labels = np.asarray([row["label"] for row in train_files], dtype=int)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=20260805)
    candidates = []
    c_values = [float(value) for value in args.c_values.split(",")]
    gammas = [parse_gamma(value.strip()) for value in args.gammas.split(",")]
    for c_value, gamma in itertools.product(c_values, gammas):
        fold_scores = []
        for fold, (fit_indices, eval_indices) in enumerate(splitter.split(np.zeros(len(file_labels)), file_labels), start=1):
            model = build_model(c_value, gamma, seed=20260805 + fold)
            fit_files(model, [train_files[index] for index in fit_indices])
            truth, predicted, _ = predict_files(model, [train_files[index] for index in eval_indices])
            fold_scores.append(score(truth, predicted))
        candidates.append({
            "C": c_value,
            "gamma": gamma,
            "cv_macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_scores])),
            "cv_macro_f1_std": float(np.std([row["macro_f1"] for row in fold_scores], ddof=1)),
            "cv_accuracy_mean": float(np.mean([row["accuracy"] for row in fold_scores])),
            "folds": fold_scores,
        })
    candidates.sort(key=lambda row: (-row["cv_macro_f1_mean"], row["cv_macro_f1_std"], -row["cv_accuracy_mean"], row["C"], str(row["gamma"])))
    winner = candidates[0]
    final_model = build_model(winner["C"], winner["gamma"], seed=20260805)
    fit_files(final_model, train_files)
    val_truth, val_predicted, records = predict_files(final_model, val_files)
    validation = score(val_truth, val_predicted)
    validation["per_class_f1"] = {
        CLASS_NAMES[label]: float(f1_score(val_truth == label, val_predicted == label, zero_division=0))
        for label in range(3)
    }

    report = {
        "protocol": "5-fold stratified file-grouped CV on train; one validation evaluation; locked test not instantiated",
        "dataset_runs": ["run1"],
        "limitation": "File-grouped within-run CV does not estimate cross-run or cross-bearing generalization.",
        "feature_name": cfg["classical"]["feature_name"],
        "aggregation": "mean_decision_score",
        "max_windows_per_file": args.max_windows_per_file,
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "winner": winner,
        "validation": validation,
        "candidates": candidates,
    }
    output_dir = ROOT / args.output_dir
    run_dir = ROOT / args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(report, indent=2) + "\n"
    (output_dir / "results.json").write_text(json_text, encoding="utf-8")
    (run_dir / "results.json").write_text(json_text, encoding="utf-8")
    with (run_dir / "model.pkl").open("wb") as stream:
        pickle.dump(final_model, stream)
    with (run_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ("file_id", "y_true", "y_pred", "n_windows", "score_0", "score_1", "score_2")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    np.savetxt(run_dir / "validation_confusion_matrix.csv", confusion_matrix(val_truth, val_predicted, labels=[0, 1, 2]), fmt="%d", delimiter=",")
    (run_dir / "validation_report.txt").write_text(
        classification_report(val_truth, val_predicted, labels=[0, 1, 2], target_names=CLASS_NAMES, digits=4, zero_division=0) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# SVM file-grouped CV and validation",
        "",
        "Five-fold stratified file-grouped CV was performed on train files only. The selected model was evaluated once on validation; the locked test split was not instantiated.",
        "",
        f"- Winner: `C={winner['C']}`, `gamma={winner['gamma']}`, mean decision-score aggregation.",
        f"- Train CV Macro-F1: `{winner['cv_macro_f1_mean']:.4f} ± {winner['cv_macro_f1_std']:.4f}`.",
        f"- Validation Accuracy: `{validation['accuracy']:.4f}`.",
        f"- Validation Macro-F1: `{validation['macro_f1']:.4f}`.",
        f"- Validation class F1: Healthy `{validation['per_class_f1']['healthy']:.4f}`, Degrading `{validation['per_class_f1']['degrading']:.4f}`, Fault `{validation['per_class_f1']['fault']:.4f}`.",
        "- Limitation: all files belong to `run1`; this is within-run file-grouped CV, not cross-run/cross-bearing validation.",
        "",
        "| Rank | C | Gamma | CV Macro-F1 mean | CV std | CV Accuracy |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(candidates, start=1):
        lines.append(f"| {rank} | {row['C']} | {row['gamma']} | {row['cv_macro_f1_mean']:.4f} | {row['cv_macro_f1_std']:.4f} | {row['cv_accuracy_mean']:.4f} |")
    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
