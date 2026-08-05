#!/usr/bin/env python3
"""Run a fixed multi-seed temperature-only validation experiment without test access."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("healthy", "degrading", "fault")


def best_validation(path: Path) -> tuple[float, float, int]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    valid = [row for row in rows if row.get("val_f1", "").lower() != "nan"]
    winner = max(valid, key=lambda row: float(row["val_f1"]))
    return float(winner["val_acc"]), float(winner["val_f1"]), int(winner["epoch"])


def report_f1(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) == 5 and tokens[0] in CLASS_NAMES:
            values[f"f1_{tokens[0]}"] = float(tokens[3])
    return values


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/revision_temperature_only_validation.yaml")
    parser.add_argument("--output-root", default="runs/revision_temperature_only_validation")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--artifact-dir", default="paper/revision_artifacts/temperature_only_validation")
    parser.add_argument("--continue", dest="resume", action="store_true")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    output_root = (ROOT / args.output_root).resolve()
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    summaries = []
    for seed in seeds:
        run_dir = output_root / f"seed_{seed}"
        summary_path = run_dir / "validation_summary.json"
        if args.resume and summary_path.is_file():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            print(f"[skip] validation seed {seed}", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(base)
        cfg["train"]["seed"] = seed
        cfg["log"]["out_dir"] = str(run_dir.relative_to(ROOT))
        derived_path = run_dir / "config.yaml"
        derived_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        command = [sys.executable, "train_logs.py", "--config", str(derived_path.relative_to(ROOT))]
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if process.wait():
                raise SystemExit(f"Temperature-only validation failed for seed {seed}")
        accuracy, macro_f1, epoch = best_validation(run_dir / "train_log.csv")
        summary = {
            "seed": seed,
            "best_epoch": epoch,
            "val_accuracy": accuracy,
            "val_macro_f1": macro_f1,
            **report_f1(run_dir / "validation_report.txt"),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)

    metrics = ("val_accuracy", "val_macro_f1", "f1_healthy", "f1_degrading", "f1_fault")
    aggregate = {"protocol": "validation-only; locked test not evaluated", "seeds": seeds, "runs": summaries, "metrics": {}}
    for metric in metrics:
        values = [float(row[metric]) for row in summaries]
        mean, std = mean_std(values)
        aggregate["metrics"][metric] = {"mean": mean, "std": std, "values": values}
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_text = json.dumps(aggregate, indent=2) + "\n"
    (output_root / "aggregate.json").write_text(aggregate_text, encoding="utf-8")
    lines = [
        "# Temperature-only validation",
        "",
        "Exploratory validation-only experiment; the locked test split was not evaluated.",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for metric in metrics:
        values = aggregate["metrics"][metric]
        lines.append(f"| {metric} | {values['mean']:.4f} | {values['std']:.4f} |")
    lines += ["", "| Seed | Best epoch | Accuracy | Macro-F1 | H F1 | D F1 | F F1 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(
            f"| {row['seed']} | {row['best_epoch']} | {row['val_accuracy']:.4f} | {row['val_macro_f1']:.4f} | "
            f"{row['f1_healthy']:.4f} | {row['f1_degrading']:.4f} | {row['f1_fault']:.4f} |"
        )
    summary_text = "\n".join(lines) + "\n"
    (output_root / "summary.md").write_text(summary_text, encoding="utf-8")
    artifact_dir = (ROOT / args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "aggregate.json").write_text(aggregate_text, encoding="utf-8")
    (artifact_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    print(output_root / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
