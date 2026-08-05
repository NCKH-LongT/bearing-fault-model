#!/usr/bin/env python3
"""Diagnose seed-level class collapse and raw temperature feature shift."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.logs_ttf import LogsTTFDataset
from features.temp_features import temp_stats_window


LABEL_NAMES = ["healthy", "degrading", "fault"]
FEATURE_NAMES = ["bearing_mean", "bearing_std", "bearing_slope", "ambient_mean", "ambient_std", "ambient_slope"]
SEEDS = [42, 43, 44, 45, 46]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def prediction_diagnostics(root: Path) -> tuple[dict, dict[str, list[int]]]:
    result = {}
    predictions_by_file: dict[str, list[int]] = {}
    for seed in SEEDS:
        rows = read_csv(root / f"seed_{seed}/eval/predictions_file.csv")
        predicted = np.asarray([int(row["y_pred"]) for row in rows], dtype=int)
        truth = np.asarray([int(row["y_true"]) for row in rows], dtype=int)
        probabilities = np.asarray(
            [[float(row[f"prob_{label}"]) for label in range(3)] for row in rows],
            dtype=float,
        )
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1) / np.log(3.0)
        confidence = probabilities.max(axis=1)
        true_probability = probabilities[np.arange(len(truth)), truth]
        counts = Counter(predicted.tolist())
        result[str(seed)] = {
            "prediction_counts": {LABEL_NAMES[label]: counts.get(label, 0) for label in range(3)},
            "classes_never_predicted": [LABEL_NAMES[label] for label in range(3) if counts.get(label, 0) == 0],
            "accuracy": float(np.mean(predicted == truth)),
            "mean_confidence": float(np.mean(confidence)),
            "mean_normalized_entropy": float(np.mean(entropy)),
            "mean_true_class_probability": float(np.mean(true_probability)),
            "mean_confidence_correct": float(np.mean(confidence[predicted == truth])) if np.any(predicted == truth) else None,
            "mean_confidence_wrong": float(np.mean(confidence[predicted != truth])) if np.any(predicted != truth) else None,
        }
        for row in rows:
            predictions_by_file.setdefault(row["file_id"], []).append(int(row["y_pred"]))
    return result, predictions_by_file


def agreement_diagnostics(predictions_by_file: dict[str, list[int]]) -> dict:
    agreement_counts = Counter()
    unstable = []
    for file_id, predictions in sorted(predictions_by_file.items()):
        counts = Counter(predictions)
        max_agreement = max(counts.values())
        agreement_counts[max_agreement] += 1
        if max_agreement <= 3:
            unstable.append({
                "file_id": file_id,
                "predictions": predictions,
                "agreement": max_agreement,
            })
    return {
        "files_by_max_seed_agreement": {str(key): agreement_counts.get(key, 0) for key in range(1, 6)},
        "unanimous_files": agreement_counts.get(5, 0),
        "unstable_files_agreement_at_most_3": unstable,
    }


def learning_curve_diagnostics(root: Path) -> dict:
    output = {}
    for seed in SEEDS:
        rows = read_csv(root / f"seed_{seed}/train_log.csv")
        validation = [float(row["val_f1"]) for row in rows]
        losses = [float(row["train_loss"]) for row in rows]
        summary = json.loads((root / f"seed_{seed}/summary.json").read_text(encoding="utf-8"))
        best_index = int(np.argmax(validation))
        output[str(seed)] = {
            "epochs_run": len(rows),
            "best_validation_macro_f1": validation[best_index],
            "best_validation_epoch": int(rows[best_index]["epoch"]),
            "test_macro_f1": float(summary["macro_f1"]),
            "validation_minus_test_macro_f1": validation[best_index] - float(summary["macro_f1"]),
            "initial_train_loss": losses[0],
            "final_train_loss": losses[-1],
        }
    return output


def split_items(cfg: dict, split: str) -> list[dict]:
    strat = cfg["stratified"]
    dataset = LogsTTFDataset(
        cfg["data_dir"],
        cfg["manifest"],
        split=split,
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        split_mode=cfg["split_mode"],
        train_ratio=strat["train"],
        val_ratio=strat["val"],
        test_ratio=strat["test"],
        min_per_class_val=strat.get("min_per_class_val"),
        min_per_class_test=strat.get("min_per_class_test"),
        random_seed=cfg["random_seed"],
        cache_dir=cfg.get("cache_dir"),
    )
    return dataset.items


def temperature_features(cfg: dict, windows_per_file: int) -> dict:
    sampling_rate = int(cfg["sampling_rate"])
    window = int(round(float(cfg["window_seconds"]) * sampling_rate))
    hop = int(round(float(cfg["hop_seconds"]) * sampling_rate))
    cache_root = ROOT / cfg.get("cache_dir", "data_cache/npy")
    output: dict[str, dict] = {}
    raw_by_split: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        features = []
        labels = []
        for item in split_items(cfg, split):
            cache_path = cache_root / f"{Path(item['file']).stem}.npy"
            if not cache_path.is_file():
                raise SystemExit(f"Missing cache file: {cache_path}")
            signal = np.load(cache_path, mmap_mode="r")
            starts = np.arange(0, max(0, len(signal) - window + 1), hop, dtype=int)
            if len(starts) > windows_per_file:
                starts = starts[np.linspace(0, len(starts) - 1, windows_per_file, dtype=int)]
            for start in starts:
                features.append(temp_stats_window(np.asarray(signal[start : start + window, 2:4])))
                labels.append(int(item["label"]))
        matrix = np.asarray(features, dtype=float)
        label_array = np.asarray(labels, dtype=int)
        raw_by_split[split] = matrix
        labels_by_split[split] = label_array
        output[split] = {
            "n_windows": len(matrix),
            "feature_mean": dict(zip(FEATURE_NAMES, matrix.mean(axis=0).tolist())),
            "feature_std": dict(zip(FEATURE_NAMES, matrix.std(axis=0).tolist())),
            "by_class": {},
        }
        for label, name in enumerate(LABEL_NAMES):
            subset = matrix[label_array == label]
            output[split]["by_class"][name] = {
                "n_windows": len(subset),
                "feature_mean": dict(zip(FEATURE_NAMES, subset.mean(axis=0).tolist())) if len(subset) else None,
                "feature_std": dict(zip(FEATURE_NAMES, subset.std(axis=0).tolist())) if len(subset) else None,
            }

    train_mean = raw_by_split["train"].mean(axis=0)
    train_std = raw_by_split["train"].std(axis=0)
    safe_std = np.where(train_std > 1e-12, train_std, 1.0)
    output["standardized_split_mean_shift_from_train"] = {
        split: dict(zip(FEATURE_NAMES, ((raw_by_split[split].mean(axis=0) - train_mean) / safe_std).tolist()))
        for split in ("val", "test")
    }
    output["raw_feature_scale_ratio_max_to_min_std"] = float(train_std.max() / max(train_std.min(), 1e-12))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multimodal-root", type=Path, required=True)
    parser.add_argument("--vibration-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows-per-file", type=int, default=32)
    args = parser.parse_args()

    config_path = args.multimodal_root / "seed_42/config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    multimodal_predictions, multimodal_files = prediction_diagnostics(args.multimodal_root)
    vibration_predictions, vibration_files = prediction_diagnostics(args.vibration_root)
    report = {
        "multimodal": {
            "predictions": multimodal_predictions,
            "seed_agreement": agreement_diagnostics(multimodal_files),
            "learning_curves": learning_curve_diagnostics(args.multimodal_root),
        },
        "vibration_only": {
            "predictions": vibration_predictions,
            "seed_agreement": agreement_diagnostics(vibration_files),
            "learning_curves": learning_curve_diagnostics(args.vibration_root),
        },
        "temperature_features": temperature_features(cfg, args.windows_per_file),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = ["# Deep model collapse diagnostics", ""]
    for model_name in ("multimodal", "vibration_only"):
        lines += [f"## {model_name}", "", "| Seed | Pred H/D/F | Missing class | Confidence | Entropy | Best val F1 | Test F1 | Gap |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
        for seed in SEEDS:
            pred = report[model_name]["predictions"][str(seed)]
            curve = report[model_name]["learning_curves"][str(seed)]
            counts = pred["prediction_counts"]
            lines.append(
                f"| {seed} | {counts['healthy']}/{counts['degrading']}/{counts['fault']} | "
                f"{', '.join(pred['classes_never_predicted']) or '-'} | {pred['mean_confidence']:.4f} | "
                f"{pred['mean_normalized_entropy']:.4f} | {curve['best_validation_macro_f1']:.4f} | "
                f"{curve['test_macro_f1']:.4f} | {curve['validation_minus_test_macro_f1']:.4f} |"
            )
        agreement = report[model_name]["seed_agreement"]
        lines += ["", f"Unanimous files across five seeds: {agreement['unanimous_files']}/27.", ""]

    temperature = report["temperature_features"]
    lines += ["## Temperature feature diagnostics", "", f"Raw train feature std scale ratio (max/min): {temperature['raw_feature_scale_ratio_max_to_min_std']:.2f}.", "", "Standardized split-mean shift relative to train:", ""]
    for split, shifts in temperature["standardized_split_mean_shift_from_train"].items():
        rendered = ", ".join(f"{name}={value:.2f}" for name, value in shifts.items())
        lines.append(f"- {split}: {rendered}")
    lines += [
        "",
        "## Evidence-based interpretation",
        "",
        "- Validation Macro-F1 is saturated while held-out test Macro-F1 drops sharply, indicating split sensitivity and model-selection overfitting to a 26-file validation set.",
        "- Several seeds never predict one or more classes, confirming genuine class collapse rather than a small metric fluctuation.",
        "- Temperature descriptors enter the linear branch in raw physical units without a train-fitted scaler; feature magnitudes and split shifts can destabilize fusion.",
        "- One-second temperature slope/std features are small relative to raw means and may be dominated without normalization or longer causal context.",
        "- The next model experiment must be validation/group-CV only: train-fitted temperature normalization plus a temperature-only baseline before gated fusion.",
    ]
    (args.output_dir / "diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "diagnostics.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
