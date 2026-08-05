#!/usr/bin/env python3
"""Resumable validation-only search and locked multi-seed confirmation for revision."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "configs/revision_search_5060.yaml"
SEARCH_ROOT = ROOT / "runs/revision_search_5060"
CONFIRM_ROOT = ROOT / "runs/revision_confirm_5060"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def repo_relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit(f"Path must be inside the repository: {path}") from exc


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, path)


def run(cmd: list[str], log_path: Path) -> int:
    print("$", " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return proc.wait()


def best_validation(train_log: Path) -> tuple[float, int]:
    best = -1.0
    best_epoch = 0
    with train_log.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["val_f1"])
                epoch = int(float(row["epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
            if score > best:
                best, best_epoch = score, epoch
    return best, best_epoch


def parse_report(path: Path) -> dict:
    result: dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        tokens = line.split()
        if len(tokens) == 5 and tokens[0] in {"healthy", "degrading", "fault"}:
            result[f"f1_{tokens[0]}"] = float(tokens[3])
        elif line.startswith("accuracy") and len(tokens) >= 2:
            result["accuracy"] = float(tokens[-2])
        elif line.startswith("macro avg") and len(tokens) >= 5:
            result["macro_f1"] = float(tokens[-2])
    return result


def candidates(base: dict, trials: int, search_root: Path) -> list[dict]:
    space = list(
        itertools.product(
            [64, 96, 128],
            [8, 16, 24],
            [1.0e-4, 2.0e-4, 3.0e-4, 5.0e-4],
            [1.0e-4, 1.0e-3, 1.0e-2],
            [0.0, 0.05, 0.10],
            [False, True],
        )
    )
    random.Random(20260803).shuffle(space)
    out: list[dict] = []
    for idx, (batch, samples_per_file, lr, wd, smoothing, class_weights) in enumerate(space[:trials], start=1):
        cfg = copy.deepcopy(base)
        cfg["train"].update(
            {
                "batch_size": batch,
                "samples_per_file": samples_per_file,
                "lr": lr,
                "weight_decay": wd,
                "label_smoothing": smoothing,
                "use_class_weights": class_weights,
                "seed": 42,
                "deterministic": False,
            }
        )
        # Keep the fresh revision file split from the base config fixed across every trial.
        cfg["log"]["out_dir"] = str(repo_relative(search_root / f"trial_{idx:03d}"))
        out.append(cfg)
    return out


def rebuild_leaderboard(search_root: Path) -> list[dict]:
    rows: list[dict] = []
    for result_path in sorted(search_root.glob("trial_*/result.json")):
        try:
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    rows.sort(key=lambda row: float(row.get("val_macro_f1", -1.0)), reverse=True)
    if rows:
        keys = [
            "rank",
            "trial",
            "status",
            "val_macro_f1",
            "best_epoch",
            "batch_size",
            "samples_per_file",
            "lr",
            "weight_decay",
            "label_smoothing",
            "use_class_weights",
            "minutes",
        ]
        with (search_root / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rank, row in enumerate(rows, start=1):
                writer.writerow({key: rank if key == "rank" else row.get(key, "") for key in keys})
    return rows


def status(args) -> int:
    root = resolve_path(args.output_root)
    rows = rebuild_leaderboard(root) if root.is_dir() else []
    complete = [row for row in rows if row.get("status") == "complete"]
    failed = [row for row in rows if row.get("status") == "failed"]
    print(f"Search root: {repo_relative(root)}")
    print(f"Completed: {len(complete)}/{args.expected}; failed: {len(failed)}")
    if complete:
        winner = complete[0]
        print(
            "Current leader: "
            f"trial={int(winner['trial']):03d} "
            f"val_macro_f1={float(winner['val_macro_f1']):.4f} "
            f"best_epoch={int(winner['best_epoch'])}"
        )
    else:
        print("Current leader: unavailable")
    winner_path = root / "best_search_config.yaml"
    search_complete = len(complete) + len(failed) >= args.expected
    winner_locked = search_complete and winner_path.is_file()
    print(f"Search complete: {'yes' if search_complete else 'no'}")
    print(f"Winner locked: {'yes' if winner_locked else 'no'}")
    return 0


def search(args) -> int:
    base = yaml.safe_load(resolve_path(args.base_config).read_text(encoding="utf-8"))
    root = resolve_path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    for idx, cfg in enumerate(candidates(base, args.trials, root), start=1):
        trial_dir = root / f"trial_{idx:03d}"
        result_path = trial_dir / "result.json"
        if result_path.is_file() and args.continue_search:
            print(f"[skip] completed trial {idx:03d}")
            continue
        trial_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = trial_dir / "config.yaml"
        write_yaml(cfg_path, cfg)
        started = time.time()
        rc = run([sys.executable, "train_logs.py", "--config", str(repo_relative(cfg_path))], trial_dir / "console.log")
        result = {
            "trial": idx,
            "status": "failed" if rc else "complete",
            "val_macro_f1": -1.0,
            "best_epoch": 0,
            "batch_size": cfg["train"]["batch_size"],
            "samples_per_file": cfg["train"]["samples_per_file"],
            "lr": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["weight_decay"],
            "label_smoothing": cfg["train"]["label_smoothing"],
            "use_class_weights": cfg["train"]["use_class_weights"],
            "minutes": round((time.time() - started) / 60.0, 2),
        }
        train_log = trial_dir / "train_log.csv"
        if rc == 0 and train_log.is_file():
            result["val_macro_f1"], result["best_epoch"] = best_validation(train_log)
        write_json(result_path, result)
        rows = rebuild_leaderboard(root)
        if rows:
            print(f"[leader] trial={rows[0]['trial']} val_macro_f1={rows[0]['val_macro_f1']:.4f}")

    rows = rebuild_leaderboard(root)
    complete = [row for row in rows if row.get("status") == "complete"]
    if not complete:
        raise SystemExit("No successful search trials.")
    winner_dir = root / f"trial_{int(complete[0]['trial']):03d}"
    shutil.copy2(winner_dir / "config.yaml", root / "best_search_config.yaml")
    (root / "best_trial.txt").write_text(str(repo_relative(winner_dir)) + "\n", encoding="utf-8")
    print(f"Best validation trial: {repo_relative(winner_dir)}")
    print("Test remains locked. Run the confirm phase only after the search space is final.")
    return 0


def mean_std(values: list[float]) -> tuple[float, float]:
    import statistics

    if len(values) < 2:
        return (values[0] if values else float("nan"), 0.0)
    return statistics.mean(values), statistics.stdev(values)


def confirm(args) -> int:
    search_root = resolve_path(args.output_root)
    best_path = search_root / "best_search_config.yaml"
    if not best_path.is_file():
        raise SystemExit(f"Missing {best_path}; complete search first.")
    base = yaml.safe_load(best_path.read_text(encoding="utf-8"))
    confirm_root = resolve_path(args.confirm_root)
    confirm_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    summaries: list[dict] = []
    for seed in seeds:
        run_dir = confirm_root / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        if summary_path.is_file() and args.continue_search:
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            print(f"[skip] completed confirmation seed {seed}")
            continue
        cfg = copy.deepcopy(base)
        # Keep the locked revision data split unchanged; vary only training randomness.
        cfg["train"]["seed"] = seed
        cfg["train"]["deterministic"] = True
        cfg["train"]["val_max_windows"] = 0
        cfg["log"]["out_dir"] = str(repo_relative(run_dir))
        cfg_path = run_dir / "config.yaml"
        write_yaml(cfg_path, cfg)
        rc = run([sys.executable, "train_logs.py", "--config", str(repo_relative(cfg_path))], run_dir / "console.log")
        if rc:
            raise SystemExit(f"Confirmation training failed for seed {seed}.")
        rc = run(
            [
                sys.executable,
                "eval_logs.py",
                "--config",
                str(repo_relative(cfg_path)),
                "--ckpt",
                str(repo_relative(run_dir / "best.pt")),
            ],
            run_dir / "console.log",
        )
        if rc:
            raise SystemExit(f"Confirmation evaluation failed for seed {seed}.")
        summary = {"seed": seed, **parse_report(run_dir / "eval/report.txt")}
        write_json(summary_path, summary)
        summaries.append(summary)

    metrics = ["accuracy", "macro_f1", "f1_healthy", "f1_degrading", "f1_fault"]
    aggregate = {"seeds": seeds, "runs": summaries, "metrics": {}}
    for metric in metrics:
        values = [float(row[metric]) for row in summaries if metric in row]
        mean, std = mean_std(values)
        aggregate["metrics"][metric] = {"mean": mean, "std": std, "values": values}
    write_json(confirm_root / "aggregate.json", aggregate)

    lines = ["# Locked multi-seed confirmation", "", "Test was evaluated only after validation-only search was completed.", ""]
    lines += ["| Metric | Mean | Std |", "|---|---:|---:|"]
    for metric in metrics:
        stats = aggregate["metrics"][metric]
        lines.append(f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} |")
    (confirm_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Confirmation summary: {repo_relative(confirm_root / 'summary.md')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    search_parser = sub.add_parser("search", help="Run/resume validation-only hyperparameter trials.")
    search_parser.add_argument("--base-config", default=str(DEFAULT_BASE))
    search_parser.add_argument("--output-root", default=str(SEARCH_ROOT))
    search_parser.add_argument("--trials", type=int, default=24)
    search_parser.add_argument("--continue", dest="continue_search", action="store_true")

    confirm_parser = sub.add_parser("confirm", help="Lock the winner and evaluate the test across seeds.")
    confirm_parser.add_argument("--output-root", default=str(SEARCH_ROOT))
    confirm_parser.add_argument("--confirm-root", default=str(CONFIRM_ROOT))
    confirm_parser.add_argument("--seeds", default="42,43,44,45,46")
    confirm_parser.add_argument("--continue", dest="continue_search", action="store_true")
    status_parser = sub.add_parser("status", help="Summarize a validation search without evaluating test data.")
    status_parser.add_argument("--output-root", default=str(SEARCH_ROOT))
    status_parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    if args.phase == "search":
        return search(args)
    if args.phase == "confirm":
        return confirm(args)
    return status(args)


if __name__ == "__main__":
    raise SystemExit(main())
