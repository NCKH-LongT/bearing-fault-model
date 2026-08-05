#!/usr/bin/env python3
"""File-level bootstrap confidence intervals and paired locked-test comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


LABELS = [0, 1, 2]
SEEDS = [42, 43, 44, 45, 46]


def read_predictions(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"file_id", "y_true", "y_pred"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"Invalid prediction file: {path}")
    return {
        row["file_id"]: {
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
        }
        for row in rows
    }


def aligned_arrays(predictions: list[dict[str, dict]]) -> tuple[list[str], np.ndarray, list[np.ndarray]]:
    file_sets = [set(table) for table in predictions]
    if any(file_set != file_sets[0] for file_set in file_sets[1:]):
        raise SystemExit("Prediction files do not contain the same file IDs.")
    file_ids = sorted(file_sets[0])
    truths = np.asarray([predictions[0][file_id]["y_true"] for file_id in file_ids], dtype=int)
    outputs = []
    for table in predictions:
        current_truths = np.asarray([table[file_id]["y_true"] for file_id in file_ids], dtype=int)
        if not np.array_equal(truths, current_truths):
            raise SystemExit("Ground-truth labels differ across prediction files.")
        outputs.append(np.asarray([table[file_id]["y_pred"] for file_id in file_ids], dtype=int))
    return file_ids, truths, outputs


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    f1_values = []
    for label in LABELS:
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        false_positive = int(np.sum((y_true != label) & (y_pred == label)))
        false_negative = int(np.sum((y_true == label) & (y_pred != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else (2.0 * true_positive) / denominator)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1_values)),
    }


def interval(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
    }


def bootstrap_model(y_true: np.ndarray, predictions: list[np.ndarray], rng: np.random.Generator, samples: int) -> dict:
    point_per_run = [metrics(y_true, prediction) for prediction in predictions]
    draws = {"accuracy": [], "macro_f1": []}
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), len(y_true))
        for metric_name in draws:
            values = [metrics(y_true[indices], prediction[indices])[metric_name] for prediction in predictions]
            draws[metric_name].append(float(np.mean(values)))
    return {
        "point_mean_across_runs": {
            name: float(np.mean([row[name] for row in point_per_run]))
            for name in draws
        },
        "bootstrap_95_ci": {name: interval(values) for name, values in draws.items()},
        "per_run": point_per_run,
    }


def bootstrap_difference(
    y_true: np.ndarray,
    predictions_a: list[np.ndarray],
    predictions_b: list[np.ndarray],
    rng: np.random.Generator,
    samples: int,
) -> dict:
    draws = {"accuracy": [], "macro_f1": []}
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), len(y_true))
        for metric_name in draws:
            score_a = np.mean([metrics(y_true[indices], pred[indices])[metric_name] for pred in predictions_a])
            score_b = np.mean([metrics(y_true[indices], pred[indices])[metric_name] for pred in predictions_b])
            draws[metric_name].append(float(score_a - score_b))
    return {
        name: {
            **interval(values),
            "probability_a_greater_than_b": float(np.mean(np.asarray(values) > 0)),
        }
        for name, values in draws.items()
    }


def mcnemar_exact(y_true: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray) -> dict:
    correct_a = prediction_a == y_true
    correct_b = prediction_b == y_true
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "exact_p_value": p_value}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multimodal-root", type=Path, required=True)
    parser.add_argument("--vibration-root", type=Path, required=True)
    parser.add_argument("--svm-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    multimodal_tables = [read_predictions(args.multimodal_root / f"seed_{seed}/eval/predictions_file.csv") for seed in SEEDS]
    vibration_tables = [read_predictions(args.vibration_root / f"seed_{seed}/eval/predictions_file.csv") for seed in SEEDS]
    svm_table = read_predictions(args.svm_predictions)
    file_ids, y_true, all_predictions = aligned_arrays(multimodal_tables + vibration_tables + [svm_table])
    multimodal_predictions = all_predictions[: len(SEEDS)]
    vibration_predictions = all_predictions[len(SEEDS) : 2 * len(SEEDS)]
    svm_predictions = [all_predictions[-1]]

    rng = np.random.default_rng(args.seed)
    report = {
        "evaluation_unit": "file",
        "n_files": len(file_ids),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "models": {
            "multimodal": bootstrap_model(y_true, multimodal_predictions, rng, args.bootstrap_samples),
            "vibration_only": bootstrap_model(y_true, vibration_predictions, rng, args.bootstrap_samples),
            "svm_vib8": bootstrap_model(y_true, svm_predictions, rng, args.bootstrap_samples),
        },
        "paired_bootstrap": {
            "multimodal_minus_vibration_only": bootstrap_difference(
                y_true, multimodal_predictions, vibration_predictions, rng, args.bootstrap_samples
            ),
            "multimodal_minus_svm": bootstrap_difference(
                y_true, multimodal_predictions, svm_predictions, rng, args.bootstrap_samples
            ),
        },
        "mcnemar_by_seed": {
            "multimodal_vs_svm": {
                str(seed): mcnemar_exact(y_true, multimodal_predictions[index], svm_predictions[0])
                for index, seed in enumerate(SEEDS)
            },
            "multimodal_vs_vibration_only": {
                str(seed): mcnemar_exact(y_true, multimodal_predictions[index], vibration_predictions[index])
                for index, seed in enumerate(SEEDS)
            },
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "statistics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Locked test file-level statistical analysis",
        "",
        f"Evaluation unit: file; n={len(file_ids)}; bootstrap replicates={args.bootstrap_samples}.",
        "",
        "| Model | Accuracy (95% CI) | Macro-F1 (95% CI) |",
        "|---|---:|---:|",
    ]
    for name in ("svm_vib8", "multimodal", "vibration_only"):
        model = report["models"][name]
        accuracy = model["bootstrap_95_ci"]["accuracy"]
        macro_f1 = model["bootstrap_95_ci"]["macro_f1"]
        lines.append(
            f"| {name} | {model['point_mean_across_runs']['accuracy']:.4f} "
            f"[{accuracy['lower_95']:.4f}, {accuracy['upper_95']:.4f}] | "
            f"{model['point_mean_across_runs']['macro_f1']:.4f} "
            f"[{macro_f1['lower_95']:.4f}, {macro_f1['upper_95']:.4f}] |"
        )
    lines += ["", "## Paired bootstrap differences (A - B)", ""]
    for comparison, values in report["paired_bootstrap"].items():
        lines.append(f"### {comparison}")
        lines.append("")
        for metric_name, stats in values.items():
            lines.append(
                f"- {metric_name}: {stats['mean']:.4f} "
                f"[95% CI {stats['lower_95']:.4f}, {stats['upper_95']:.4f}]; "
                f"P(A>B)={stats['probability_a_greater_than_b']:.4f}."
            )
        lines.append("")
    lines.append("McNemar exact results for every seed are stored in `statistics.json`.")
    (args.output_dir / "statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "statistics.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
