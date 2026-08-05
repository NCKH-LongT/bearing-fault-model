#!/usr/bin/env python3
"""Step-by-step runner for reproducing and revising the bearing paper."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ARTIFACTS = PAPER / "revision_artifacts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

PRIMARY_CFG = "configs/revision_primary_multimodal.yaml"
VIB_CFG = "configs/revision_primary_vibration_only.yaml"
SVM_CFG = "classical_baselines/configs/revision_svm_vib8_stratified.yaml"
PRIMARY_RUN = ROOT / "runs/revision/primary_multimodal"
VIB_RUN = ROOT / "runs/revision/primary_vibration_only"
SVM_RUN = ROOT / "runs/revision/svm_vib8_stratified"
LEGACY_INIT = ROOT / "runs/logs_stft_strat/auto_r22/best.pt"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run(cmd: list[str], *, cwd: Path = ROOT, dry_run: bool = False) -> None:
    print("$", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {rel(path)}")


def check_python_modules(modules: Iterable[str]) -> list[str]:
    return [name for name in modules if importlib.util.find_spec(name) is None]


def preflight(
    *,
    check_training: bool = True,
    check_latex: bool = True,
    require_legacy: bool = False,
) -> None:
    print("[preflight] Checking requested prerequisites")
    if check_training:
        require(ROOT / "data/manifest.csv", "manifest")
        require(ROOT / PRIMARY_CFG, "primary config")
        require(ROOT / VIB_CFG, "vibration-only config")
        require(ROOT / SVM_CFG, "SVM config")
        require(ROOT / "train_logs.py", "training entry point")
        require(ROOT / "eval_logs.py", "evaluation entry point")

        missing_modules = check_python_modules(["numpy", "yaml", "torch", "sklearn", "matplotlib"])
        if missing_modules:
            message = "Missing Python modules: " + ", ".join(missing_modules)
            if "torch" in missing_modules:
                message += "\nInstall PyTorch as described in paper/REVISION_STEP_BY_STEP.md, then rerun preflight."
            raise SystemExit(message)

        with (ROOT / "data/manifest.csv").open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        missing_data = [r.get("file", "") for r in rows if not (ROOT / "data" / r.get("file", "")).is_file()]
        if missing_data:
            sample = ", ".join(missing_data[:5])
            raise SystemExit(f"Missing {len(missing_data)} signal files; first entries: {sample}")

        if require_legacy:
            require(LEGACY_INIT, "legacy auto_r22 initialization checkpoint")
        elif not LEGACY_INIT.is_file():
            print(f"[preflight] Warning: {rel(LEGACY_INIT)} is absent; secondary legacy reproduction is unavailable.")

        counts = Counter((r.get("fault_type") or "").strip().lower() for r in rows)
        print(f"[preflight] Training inputs OK: {len(rows)} manifest rows; class counts={dict(counts)}")

    if check_latex:
        require(PAPER / "main.tex", "LaTeX entry point")
        for tool in ("latexmk", "pdflatex", "bibtex"):
            if shutil.which(tool) is None:
                raise SystemExit(f"Missing executable: {tool}")
        print("[preflight] LaTeX tools OK")


def smoke(*, dry_run: bool = False) -> None:
    """Exercise one real sample through preprocessing, forward and backward."""
    print("[smoke] Running one real sample through STFT, fusion model and backward pass")
    if dry_run:
        print("[dry-run] Would load one real training file and run a CPU forward/backward pass")
        return
    import torch
    import yaml

    from datasets.logs_ttf import LogsTTFDataset
    from features.spectrogram import SpectrogramTransform
    from features.temp_features import resolve_temp_feature
    from models.resnet2d import ResNet18Small

    with (ROOT / PRIMARY_CFG).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    transform = SpectrogramTransform(
        n_fft=cfg["stft"]["n_fft"],
        hop_length=cfg["stft"]["hop_length"],
        window=cfg["stft"]["window"],
        log_add=cfg["stft"]["log_add"],
        target_size=tuple(cfg["input_size"]),
        training=False,
    )
    temp_fn, temp_dim = resolve_temp_feature("stats6")
    strat = cfg["stratified"]
    ds = LogsTTFDataset(
        cfg["data_dir"],
        cfg["manifest"],
        split="train",
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        split_mode="stratified",
        train_ratio=strat["train"],
        val_ratio=strat["val"],
        test_ratio=strat["test"],
        min_per_class_val=strat.get("min_per_class_val"),
        min_per_class_test=strat.get("min_per_class_test"),
        random_seed=cfg["random_seed"],
        transform=transform,
        temp_feature_fn=temp_fn,
        temp_feat_dim=temp_dim,
        seconds_cap=2.0,
    )
    x, temp, target = ds[0]
    model = ResNet18Small(in_ch=2, num_classes=cfg["num_classes"], temp_feat_dim=temp_dim)
    logits = model(x.unsqueeze(0), temp.unsqueeze(0))
    loss = torch.nn.functional.cross_entropy(logits, target.unsqueeze(0))
    loss.backward()
    expected_x = (2, *tuple(cfg["input_size"]))
    if tuple(x.shape) != expected_x or tuple(temp.shape) != (6,) or tuple(logits.shape) != (1, 3):
        raise SystemExit(
            f"Unexpected shapes: x={tuple(x.shape)}, temp={tuple(temp.shape)}, logits={tuple(logits.shape)}"
        )
    print(
        f"[smoke] OK: x={tuple(x.shape)}, temp={tuple(temp.shape)}, "
        f"logits={tuple(logits.shape)}, loss={float(loss.item()):.4f}"
    )


def require_training_device(*, allow_cpu: bool, dry_run: bool) -> None:
    if dry_run:
        return
    import torch

    if torch.cuda.is_available():
        print(f"[device] CUDA training on {torch.cuda.get_device_name(0)}")
        return
    if allow_cpu:
        print("[device] Warning: CUDA is unavailable; CPU training was explicitly allowed.")
        return
    raise SystemExit(
        "CUDA is unavailable. Fix the NVIDIA driver/install a CUDA PyTorch wheel, "
        "or explicitly pass --allow-cpu-training (very slow)."
    )


def _make_dataset(cfg: dict, split: str, *, temporal: bool):
    from datasets.logs_ttf import LogsTTFDataset

    strat = cfg.get("stratified", {}) or {}
    ttf_cfg = cfg.get("temporal_ttf", {}) or {}
    default_ranges = {"train": (0.0, 60.0), "val": (60.0, 70.0), "test": (70.0, 100.1)}
    ttf_split = tuple(ttf_cfg.get(split, default_ranges[split])) if temporal else (0.0, 100.1)
    return LogsTTFDataset(
        cfg["data_dir"],
        cfg["manifest"],
        split=split,
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        ttf_split=ttf_split,
        split_mode="temporal" if temporal else "stratified",
        train_ratio=strat.get("train", 0.6),
        val_ratio=strat.get("val", 0.2),
        test_ratio=strat.get("test", 0.2),
        min_per_class_val=strat.get("min_per_class_val"),
        min_per_class_test=strat.get("min_per_class_test"),
        random_seed=cfg.get("random_seed", 42),
        transform=None,
        temp_feature_fn=None,
        temp_feat_dim=0,
        exclude_list=cfg.get("exclude_list"),
    )


def _split_summary(items: list[dict]) -> tuple[int, Counter]:
    names = {0: "healthy", 1: "degrading", 2: "fault"}
    return len(items), Counter(names[int(item["label"])] for item in items)


def audit(*, dry_run: bool = False) -> None:
    print("[audit] Auditing file-level splits and legacy overlap")
    if dry_run:
        print(f"[dry-run] Would write {rel(ARTIFACTS / 'protocol_audit.md')}")
        return

    import yaml

    with (ROOT / PRIMARY_CFG).open("r", encoding="utf-8") as f:
        primary_cfg = yaml.safe_load(f)
    with (ROOT / "configs/best_temporal.yaml").open("r", encoding="utf-8") as f:
        temporal_cfg = yaml.safe_load(f)

    strat = {s: _make_dataset(primary_cfg, s, temporal=False).items for s in ("train", "val", "test")}
    temporal = {s: _make_dataset(temporal_cfg, s, temporal=True).items for s in ("train", "val", "test")}

    def files(items: list[dict]) -> set[str]:
        return {str(item["file"]) for item in items}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = ARTIFACTS / "protocol_audit.md"
    lines = [
        "# Protocol audit",
        "",
        "Generated by `paper/run_revision.py audit`.",
        "",
        "## File and class counts",
        "",
        "| Protocol | Split | Files | Healthy | Degrading | Fault |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for protocol, groups in (("Primary stratified", strat), ("Legacy temporal", temporal)):
        for split, items in groups.items():
            total, counts = _split_summary(items)
            lines.append(
                f"| {protocol} | {split} | {total} | {counts['healthy']} | "
                f"{counts['degrading']} | {counts['fault']} |"
            )

    lines += ["", "## File overlap", ""]
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = files(strat[left]) & files(strat[right])
        lines.append(f"- Primary stratified `{left}` ∩ `{right}`: **{len(overlap)} files**.")

    legacy_overlap = files(strat["train"]) & files(temporal["test"])
    legacy_fault_overlap = {
        item["file"] for item in strat["train"] if int(item["label"]) == 2
    } & files(temporal["test"])
    lines += [
        f"- Stratified train ∩ temporal test: **{len(legacy_overlap)} files**.",
        f"- Stratified-train Fault ∩ temporal test: **{len(legacy_fault_overlap)} files**.",
        "",
        "## Interpretation",
        "",
        "- The primary stratified split is file-disjoint and multi-class, but it is a within-run random file split.",
        "- The legacy temporal test has no Healthy files.",
        "- Initializing temporal training from a stratified checkpoint exposes the model to temporal-test files; it is not a leakage-free generalization test.",
        "- The `[0,100]%` evaluation includes training/validation regions and must be described as whole-trajectory retrospective evaluation.",
        "- Exact window counts are intentionally omitted because counting them requires scanning the full raw dataset; file-level support is the evaluation unit used by the paper.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] Wrote {rel(output)}")


def copy_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    print(f"[sync] {rel(src)} -> {rel(dst)}")
    if dry_run:
        return
    require(src, "artifact directory")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def primary(*, dry_run: bool = False) -> None:
    print("[primary] Training and evaluating the revised multi-modal primary model")
    run([sys.executable, "train_logs.py", "--config", PRIMARY_CFG], dry_run=dry_run)
    run(
        [
            sys.executable,
            "eval_logs.py",
            "--config",
            PRIMARY_CFG,
            "--ckpt",
            "runs/revision/primary_multimodal/best.pt",
        ],
        dry_run=dry_run,
    )
    copy_tree(PRIMARY_RUN / "eval", ARTIFACTS / "primary_multimodal", dry_run=dry_run)


def baselines(*, dry_run: bool = False) -> None:
    print("[baselines] Running matched vibration-only CNN and classical SVM")
    run([sys.executable, "train_logs.py", "--config", VIB_CFG], dry_run=dry_run)
    run(
        [
            sys.executable,
            "eval_logs.py",
            "--config",
            VIB_CFG,
            "--ckpt",
            "runs/revision/primary_vibration_only/best.pt",
        ],
        dry_run=dry_run,
    )
    run([sys.executable, "classical_baselines/train_classical.py", "--config", SVM_CFG], dry_run=dry_run)
    copy_tree(VIB_RUN / "eval", ARTIFACTS / "primary_vibration_only", dry_run=dry_run)
    if dry_run:
        print(f"[sync] Would copy reports from {rel(SVM_RUN)}")
    else:
        require(SVM_RUN, "SVM output")
        destination = ARTIFACTS / "svm_vib8"
        destination.mkdir(parents=True, exist_ok=True)
        for path in SVM_RUN.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)


def secondary(*, dry_run: bool = False) -> None:
    print("[secondary] Reproducing legacy temporal/whole-trajectory artifacts")
    require(LEGACY_INIT, "legacy auto_r22 initialization checkpoint")
    run([sys.executable, "scripts/run_paper_sync.py", "--sync-figures"], dry_run=dry_run)


def latex(*, dry_run: bool = False) -> None:
    print("[latex] Building paper/main.pdf")
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=PAPER, dry_run=dry_run)
    run(["latexmk", "-c", "main.tex"], cwd=PAPER, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=("preflight", "smoke", "audit", "primary", "baselines", "secondary", "latex", "all"),
        help="Revision step to run. 'all' excludes the scientifically secondary legacy reproduction.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print expensive/mutating commands without running them.")
    parser.add_argument(
        "--allow-cpu-training",
        action="store_true",
        help="Allow primary/baseline training without CUDA. This can be very slow.",
    )
    args = parser.parse_args()

    if args.step == "preflight":
        preflight()
    elif args.step == "smoke":
        preflight(check_latex=False)
        smoke(dry_run=args.dry_run)
    elif args.step == "audit":
        preflight(check_latex=False)
        audit(dry_run=args.dry_run)
    elif args.step == "primary":
        preflight(check_latex=False)
        require_training_device(allow_cpu=args.allow_cpu_training, dry_run=args.dry_run)
        primary(dry_run=args.dry_run)
    elif args.step == "baselines":
        preflight(check_latex=False)
        require_training_device(allow_cpu=args.allow_cpu_training, dry_run=args.dry_run)
        baselines(dry_run=args.dry_run)
    elif args.step == "secondary":
        preflight(check_latex=False, require_legacy=True)
        secondary(dry_run=args.dry_run)
    elif args.step == "latex":
        preflight(check_training=False)
        latex(dry_run=args.dry_run)
    elif args.step == "all":
        preflight()
        smoke(dry_run=args.dry_run)
        audit(dry_run=args.dry_run)
        require_training_device(allow_cpu=args.allow_cpu_training, dry_run=args.dry_run)
        primary(dry_run=args.dry_run)
        baselines(dry_run=args.dry_run)
        latex(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
