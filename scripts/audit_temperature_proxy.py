#!/usr/bin/env python3
"""Audit temperature-only signal, modality permutation, and TTF association on train folds."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_classical_filecv import build_model
from scripts.search_classical_filecv import file_features, fit_files, predict_files, score


TEMP_NAMES = ("bearing_mean", "bearing_std", "bearing_slope", "ambient_mean", "ambient_std", "ambient_slope")


def permuted_files(files: list[dict], columns: slice, rng: np.random.Generator) -> list[dict]:
    order = rng.permutation(len(files))
    output = []
    for target_index, source_index in enumerate(order):
        row = {**files[target_index], "features": files[target_index]["features"].copy()}
        source = files[source_index]["features"]
        if len(source) != len(row["features"]):
            indices = np.linspace(0, len(source) - 1, len(row["features"]), dtype=int)
            source = source[indices]
        row["features"][:, columns] = source[:, columns]
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="classical_baselines/configs/revision_svm_vib8_stratified.yaml")
    parser.add_argument("--results", default="paper/revision_artifacts/classical_feature_filecv/results.json")
    parser.add_argument("--output-dir", default="paper/revision_artifacts/temperature_proxy_audit")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-windows-per-file", type=int, default=32)
    parser.add_argument("--permutation-repeats", type=int, default=30)
    args = parser.parse_args()

    base = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    result = json.loads((ROOT / args.results).read_text(encoding="utf-8"))
    winner = result["winner"]
    if winner["feature"] != "vib_temp_stats_32d":
        raise SystemExit("Proxy audit currently expects the 32-D multimodal winner")

    multimodal_cfg = copy.deepcopy(base)
    multimodal_cfg["classical"]["feature_name"] = "vib_temp_stats_32d"
    train_files = file_features(multimodal_cfg, "train", args.max_windows_per_file)
    labels = np.asarray([row["label"] for row in train_files], dtype=int)
    folds = list(StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=20260805).split(np.zeros(len(labels)), labels))
    groups = {"vibration_26d": slice(0, 26), "temperature_6d": slice(26, 32)}
    fold_models = []
    baseline_folds = []
    for fold, (fit_indices, eval_indices) in enumerate(folds, start=1):
        model = build_model(winner, 20260805 + fold)
        fit_files(model, [train_files[index] for index in fit_indices])
        if hasattr(model[-1], "n_jobs"):
            model[-1].n_jobs = 1
        eval_files = [train_files[index] for index in eval_indices]
        truth, predicted, _ = predict_files(model, eval_files)
        baseline_folds.append(score(truth, predicted)["macro_f1"])
        fold_models.append((model, eval_files, truth))

    permutation = {}
    for group_index, (group_name, columns) in enumerate(groups.items()):
        repeat_means = []
        for repeat in range(args.permutation_repeats):
            fold_values = []
            for fold, (model, eval_files, truth) in enumerate(fold_models):
                rng = np.random.default_rng(20260805 + group_index * 100000 + repeat * 100 + fold)
                _, predicted, _ = predict_files(model, permuted_files(eval_files, columns, rng))
                fold_values.append(score(truth, predicted)["macro_f1"])
            repeat_means.append(float(np.mean(fold_values)))
        permutation[group_name] = {
            "permuted_macro_f1_mean": float(np.mean(repeat_means)),
            "permuted_macro_f1_std": float(np.std(repeat_means, ddof=1)),
            "macro_f1_drop": float(np.mean(baseline_folds) - np.mean(repeat_means)),
            "repeat_values": repeat_means,
        }

    temp_cfg = copy.deepcopy(base)
    temp_cfg["classical"]["feature_name"] = "temp_stats_6d"
    temp_files = file_features(temp_cfg, "train", args.max_windows_per_file)
    ttf = np.asarray([row["ttf_percent"] for row in temp_files], dtype=float)
    file_means = np.stack([row["features"].mean(axis=0) for row in temp_files])
    correlations = {}
    for index, name in enumerate(TEMP_NAMES):
        rho, p_value = spearmanr(file_means[:, index], ttf)
        correlations[name] = {"spearman_rho": float(rho), "p_value": float(p_value)}

    audit = {
        "protocol": "train-file folds only; locked test and validation not instantiated",
        "limitation": "All files are from run1; TTF association can reflect trajectory/time proxy rather than transferable fault physics.",
        "winner": winner,
        "baseline_cv_macro_f1_mean": float(np.mean(baseline_folds)),
        "baseline_cv_macro_f1_std": float(np.std(baseline_folds, ddof=1)),
        "permutation_repeats": args.permutation_repeats,
        "file_grouped_modality_permutation": permutation,
        "temperature_file_mean_spearman_vs_ttf": correlations,
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Temperature proxy audit",
        "",
        "All analyses use train-file folds only; validation and the locked test split were not instantiated.",
        "",
        f"Baseline multimodal CV Macro-F1: `{audit['baseline_cv_macro_f1_mean']:.4f} ± {audit['baseline_cv_macro_f1_std']:.4f}`.",
        "",
        "## File-grouped modality permutation",
        "",
        "| Permuted modality | Macro-F1 after permutation | Repeat std | Macro-F1 drop |",
        "|---|---:|---:|---:|",
    ]
    for name, row in permutation.items():
        lines.append(f"| {name} | {row['permuted_macro_f1_mean']:.4f} | {row['permuted_macro_f1_std']:.4f} | {row['macro_f1_drop']:.4f} |")
    lines += ["", "## Temperature association with TTF", "", "| Descriptor | Spearman rho | p-value |", "|---|---:|---:|"]
    for name, row in correlations.items():
        lines.append(f"| {name} | {row['spearman_rho']:.4f} | {row['p_value']:.3e} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- A large drop after file-grouped temperature permutation means the model materially relies on temperature beyond vibration within this run.",
        "- Strong temperature–TTF correlation also means temperature may encode trajectory position; this cannot be separated from transferable degradation signal with only run1.",
        "- Do not claim cross-run sensor-fusion generalization until an independent run/bearing is evaluated.",
    ]
    (output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
