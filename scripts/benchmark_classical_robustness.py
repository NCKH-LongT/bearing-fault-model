#!/usr/bin/env python3
"""Validation-only robustness and CPU efficiency benchmark for the P2 winner."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classical_baselines.features import resolve_feature_extractor
from classical_baselines.pipeline import build_split_items, make_windows, read_signal_csv
from scripts.search_classical_filecv import file_features


CLASS_NAMES = ("healthy", "degrading", "fault")
CONDITIONS = ("clean", "noise_20db", "noise_10db", "missing_vib_x", "missing_vib_y", "missing_temperature", "temperature_drift_plus_2c")


def selected_windows(signal: np.ndarray, win: int, hop: int, maximum: int) -> list[tuple[int, int]]:
    windows = make_windows(len(signal), win, hop)
    if len(windows) > maximum:
        indices = np.linspace(0, len(windows) - 1, maximum, dtype=int)
        windows = [windows[index] for index in indices]
    return windows


def perturb(window: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    output = np.asarray(window, dtype=np.float32).copy()
    if condition.startswith("noise_"):
        snr_db = float(condition.split("_")[1].removesuffix("db"))
        for channel in (0, 1):
            signal_power = float(np.mean(output[:, channel] ** 2))
            noise_std = np.sqrt(signal_power / (10.0 ** (snr_db / 10.0)) + 1e-12)
            output[:, channel] += rng.normal(0.0, noise_std, len(output)).astype(np.float32)
    elif condition == "missing_vib_x":
        output[:, 0] = 0.0
    elif condition == "missing_vib_y":
        output[:, 1] = 0.0
    elif condition == "temperature_drift_plus_2c":
        output[:, 2:4] += 2.0
    return output


def evaluate_condition(cfg: dict, model, items: list[dict], condition: str, maximum: int, temp_impute: np.ndarray) -> tuple[dict, float, int]:
    extractor = resolve_feature_extractor("vib_temp_stats_32d")
    win = int(round(float(cfg["window_seconds"]) * int(cfg["sampling_rate"])))
    hop = int(round(float(cfg["hop_seconds"]) * int(cfg["sampling_rate"])))
    truth, predicted = [], []
    feature_seconds = 0.0
    windows_total = 0
    for file_index, item in enumerate(items):
        signal = read_signal_csv(item["path"], cache_dir=cfg.get("cache_dir"))
        features = []
        for window_index, (start, end) in enumerate(selected_windows(signal, win, hop, maximum)):
            rng = np.random.default_rng(20260805 + file_index * 1000 + window_index)
            window = perturb(signal[start:end], condition, rng)
            started = time.perf_counter()
            vector = extractor(window)
            feature_seconds += time.perf_counter() - started
            if condition == "missing_temperature":
                vector[26:32] = temp_impute
            features.append(vector)
        matrix = np.stack(features)
        scores = model.predict_proba(matrix).mean(axis=0)
        truth.append(int(item["label"]))
        predicted.append(int(model.classes_[int(np.argmax(scores))]))
        windows_total += len(matrix)
    truth_array = np.asarray(truth)
    predicted_array = np.asarray(predicted)
    class_f1 = f1_score(truth_array, predicted_array, labels=[0, 1, 2], average=None, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(truth_array, predicted_array)),
        "macro_f1": float(f1_score(truth_array, predicted_array, labels=[0, 1, 2], average="macro", zero_division=0)),
        "class_f1": {CLASS_NAMES[index]: float(class_f1[index]) for index in range(3)},
    }
    return metrics, feature_seconds, windows_total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="classical_baselines/configs/revision_svm_vib8_stratified.yaml")
    parser.add_argument("--model", default="runs/revision_classical_feature_filecv/model.pkl")
    parser.add_argument("--output-dir", default="paper/revision_artifacts/classical_robustness_efficiency")
    parser.add_argument("--run-dir", default="runs/revision_classical_robustness_efficiency")
    parser.add_argument("--max-windows-per-file", type=int, default=32)
    parser.add_argument("--latency-repeats", type=int, default=100)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    model_path = ROOT / args.model
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    if hasattr(model[-1], "n_jobs"):
        model[-1].n_jobs = 1
    train_cfg = dict(cfg)
    train_cfg["classical"] = dict(cfg["classical"])
    train_cfg["classical"]["feature_name"] = "vib_temp_stats_32d"
    train_rows = file_features(train_cfg, "train", args.max_windows_per_file)
    train_matrix = np.concatenate([row["features"] for row in train_rows], axis=0)
    temp_impute = train_matrix[:, 26:32].mean(axis=0)
    validation_items = build_split_items(cfg, "val")

    conditions = {}
    clean_feature_seconds = 0.0
    clean_windows = 0
    for condition in CONDITIONS:
        metrics, feature_seconds, windows_total = evaluate_condition(
            cfg, model, validation_items, condition, args.max_windows_per_file, temp_impute
        )
        conditions[condition] = metrics
        if condition == "clean":
            clean_feature_seconds, clean_windows = feature_seconds, windows_total
    clean_macro = conditions["clean"]["macro_f1"]
    for condition, metrics in conditions.items():
        metrics["macro_f1_drop_from_clean"] = clean_macro - metrics["macro_f1"]

    sample = train_matrix[: min(256, len(train_matrix))]
    model.predict_proba(sample)
    latency_values = []
    for _ in range(args.latency_repeats):
        started = time.perf_counter()
        model.predict_proba(sample)
        latency_values.append(time.perf_counter() - started)
    classifier = model[-1]
    tree_nodes = int(sum(estimator.tree_.node_count for estimator in classifier.estimators_))
    efficiency = {
        "cpu_feature_ms_per_window": 1000.0 * clean_feature_seconds / clean_windows,
        "cpu_feature_windows_per_second": clean_windows / clean_feature_seconds,
        "cpu_inference_ms_per_window": 1000.0 * float(np.mean(latency_values)) / len(sample),
        "cpu_inference_windows_per_second": len(sample) / float(np.mean(latency_values)),
        "latency_repeats": args.latency_repeats,
        "latency_batch_windows": len(sample),
        "model_bytes": model_path.stat().st_size,
        "trees": len(classifier.estimators_),
        "total_tree_nodes": tree_nodes,
        "max_tree_depth": max(estimator.tree_.max_depth for estimator in classifier.estimators_),
    }
    efficiency["estimated_end_to_end_ms_per_32_window_file"] = 32.0 * (
        efficiency["cpu_feature_ms_per_window"] + efficiency["cpu_inference_ms_per_window"]
    )
    efficiency["estimated_files_per_second_32_windows"] = 1000.0 / efficiency["estimated_end_to_end_ms_per_32_window_file"]
    report = {
        "protocol": "validation-only robustness; train-only imputation; locked test not instantiated",
        "model": "Random Forest, vibration 26-D + temperature 6-D",
        "conditions": conditions,
        "efficiency": efficiency,
    }
    output_dir = ROOT / args.output_dir
    run_dir = ROOT / args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(report, indent=2) + "\n"
    (output_dir / "results.json").write_text(report_text, encoding="utf-8")
    (run_dir / "results.json").write_text(report_text, encoding="utf-8")
    lines = [
        "# Classical robustness and efficiency",
        "",
        "Validation only; missing-temperature imputation was fitted on train features and the locked test split was not instantiated.",
        "",
        "| Condition | Accuracy | Macro-F1 | Drop | Healthy F1 | Degrading F1 | Fault F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = conditions[condition]
        lines.append(
            f"| {condition} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['macro_f1_drop_from_clean']:.4f} | "
            f"{row['class_f1']['healthy']:.4f} | {row['class_f1']['degrading']:.4f} | {row['class_f1']['fault']:.4f} |"
        )
    lines += [
        "",
        "## CPU efficiency",
        "",
        f"- Feature extraction: `{efficiency['cpu_feature_ms_per_window']:.3f} ms/window` (`{efficiency['cpu_feature_windows_per_second']:.1f}` windows/s).",
        f"- Batched RF inference: `{efficiency['cpu_inference_ms_per_window']:.4f} ms/window` (`{efficiency['cpu_inference_windows_per_second']:.1f}` windows/s).",
        f"- Estimated end-to-end for a 32-window file: `{efficiency['estimated_end_to_end_ms_per_32_window_file']:.2f} ms/file` (`{efficiency['estimated_files_per_second_32_windows']:.2f}` files/s), excluding CSV/NPY I/O.",
        f"- Model: `{efficiency['model_bytes'] / 1024:.1f} KiB`, `{efficiency['trees']}` trees, `{efficiency['total_tree_nodes']}` total nodes, max depth `{efficiency['max_tree_depth']}`.",
        "",
        "Timing is machine-dependent and was measured on the current CPU; use the JSON artifact for exact protocol fields.",
    ]
    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
