#!/usr/bin/env python3
"""Package locked revision winners for a GitHub Release without raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist/bearing-revision-v2-artifacts.tar.gz"


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"Missing required artifact: {source.relative_to(ROOT)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def add_confirmation(source_root: Path, destination_root: Path) -> None:
    for filename in ("summary.md", "aggregate.json"):
        copy_required(source_root / filename, destination_root / filename)
    for seed in (42, 43, 44, 45, 46):
        source_seed = source_root / f"seed_{seed}"
        destination_seed = destination_root / f"seed_{seed}"
        for relative in (
            Path("best.pt"),
            Path("config.yaml"),
            Path("summary.json"),
            Path("eval/report.txt"),
            Path("eval/confusion_matrix.csv"),
            Path("eval/predictions_file.csv"),
        ):
            copy_required(source_seed / relative, destination_seed / relative)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="bearing-release-") as temporary:
        package_root = Path(temporary) / "bearing-revision-v2-artifacts"
        add_confirmation(
            ROOT / "runs/revision_confirm_5060_v2",
            package_root / "multimodal",
        )
        add_confirmation(
            ROOT / "runs/revision_confirm_vibration_5060_v2",
            package_root / "vibration_only",
        )

        svm_destination = package_root / "svm"
        for filename in ("model.pkl", "report_test.txt", "confusion_matrix_test.csv", "predictions_test.csv"):
            copy_required(ROOT / "runs/revision/svm_vib8_stratified" / filename, svm_destination / filename)
        copy_required(
            ROOT / "classical_baselines/configs/revision_svm_vib8_stratified.yaml",
            svm_destination / "config.yaml",
        )

        for source_root, destination_name in (
            (ROOT / "runs/revision_search_5060_v2", "multimodal_search"),
            (ROOT / "runs/revision_search_vibration_5060_v2", "vibration_search"),
        ):
            destination = package_root / "search" / destination_name
            for filename in ("leaderboard.csv", "best_search_config.yaml", "best_trial.txt"):
                copy_required(source_root / filename, destination / filename)

        statistics_root = ROOT / "paper/revision_artifacts/locked_test_statistics"
        for filename in ("statistics.md", "statistics.json"):
            copy_required(statistics_root / filename, package_root / "statistics" / filename)

        diagnostics_root = ROOT / "paper/revision_artifacts/deep_collapse_diagnostics"
        for filename in ("diagnostics.md", "diagnostics.json"):
            copy_required(diagnostics_root / filename, package_root / "diagnostics" / filename)

        temperature_root = ROOT / "paper/revision_artifacts/temperature_only_validation"
        for filename in ("summary.md", "aggregate.json"):
            copy_required(temperature_root / filename, package_root / "temperature_validation" / filename)

        temperature_runs = ROOT / "runs/revision_temperature_only_validation"
        for filename in ("summary.md", "aggregate.json"):
            copy_required(temperature_runs / filename, package_root / "temperature_only" / filename)
        for seed in (42, 43, 44, 45, 46):
            for filename in (
                "best.pt",
                "config.yaml",
                "train_log.csv",
                "validation_report.txt",
                "validation_confusion_matrix.csv",
                "validation_summary.json",
            ):
                copy_required(
                    temperature_runs / f"seed_{seed}" / filename,
                    package_root / "temperature_only" / f"seed_{seed}" / filename,
                )

        normalized_runs = ROOT / "runs/revision_multimodal_normalized_validation"
        for filename in ("summary.md", "aggregate.json"):
            copy_required(normalized_runs / filename, package_root / "multimodal_normalized_validation" / filename)
        for seed in (42, 43, 44, 45, 46):
            for filename in (
                "best.pt",
                "config.yaml",
                "train_log.csv",
                "validation_report.txt",
                "validation_confusion_matrix.csv",
                "validation_summary.json",
            ):
                copy_required(
                    normalized_runs / f"seed_{seed}" / filename,
                    package_root / "multimodal_normalized_validation" / f"seed_{seed}" / filename,
                )

        balance_audit = ROOT / "paper/revision_artifacts/train_balance_audit"
        for filename in ("audit.md", "audit.json"):
            copy_required(balance_audit / filename, package_root / "train_balance_audit" / filename)

        svm_filecv = ROOT / "runs/revision_svm_filecv_validation"
        for filename in (
            "summary.md",
            "results.json",
            "model.pkl",
            "validation_predictions.csv",
            "validation_report.txt",
            "validation_confusion_matrix.csv",
        ):
            copy_required(svm_filecv / filename, package_root / "svm_filecv_validation" / filename)

        files = sorted(path for path in package_root.rglob("*") if path.is_file())
        manifest = {
            "artifact_set": "bearing-revision-v2",
            "contains_raw_dataset": False,
            "split_seed": 20260803,
            "training_seeds": [42, 43, 44, 45, 46],
            "files": [
                {
                    "path": str(path.relative_to(package_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ],
        }
        (package_root / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        with tarfile.open(output, "w:gz") as archive:
            archive.add(package_root, arcname=package_root.name)

    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(f"Created {output.relative_to(ROOT)} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print(f"Created {checksum_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
