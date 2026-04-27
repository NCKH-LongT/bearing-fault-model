import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def copy_if_exists(src: Path, dst: Path):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def sync_figures():
    copy_if_exists(ROOT / "runs/logs_stft_strat/auto_r22/eval", ROOT / "figures/stratified")
    copy_if_exists(ROOT / "runs/paper_sync/temporal/eval", ROOT / "figures/temporal")

    fullrange_dir = ROOT / "figures/fullrange"
    fullrange_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(ROOT / "runs/paper_sync/fullrange/eval_vote/report.txt", fullrange_dir / "report.txt")
    copy_if_exists(ROOT / "runs/paper_sync/fullrange/eval_vote/report_present.txt", fullrange_dir / "report_present.txt")
    copy_if_exists(ROOT / "runs/paper_sync/fullrange/eval_vote/confusion_matrix.csv", fullrange_dir / "confusion_matrix_full.csv")
    copy_if_exists(ROOT / "runs/paper_sync/fullrange/eval_vote/confusion_matrix.png", fullrange_dir / "confusion_matrix_full.png")
    copy_if_exists(ROOT / "runs/paper_sync/fullrange/eval_vote/f1_present.png", fullrange_dir / "f1_present_full.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=str(ROOT / ".venv/Scripts/python.exe"))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument(
        "--run-stratified",
        action="store_true",
        help="Also regenerate the paper_sync stratified run. Not recommended for exact paper-number reproduction.",
    )
    ap.add_argument("--sync-figures", action="store_true")
    args = ap.parse_args()

    py = args.python
    strat_cfg = "configs/paper_sync_stratified.yaml"
    temporal_cfg = "configs/best_temporal.yaml"
    full_cfg = "configs/best_fullrange_eval.yaml"

    if args.run_stratified:
        run([py, "train_logs.py", "--config", strat_cfg])
        run([py, "eval_logs.py", "--config", strat_cfg, "--ckpt", "runs/paper_sync/stratified/best.pt"])

    if not args.skip_train:
        run([py, "train_logs.py", "--config", temporal_cfg])

    run([py, "eval_logs.py", "--config", temporal_cfg, "--ckpt", "runs/paper_sync/temporal/best.pt"])
    run([py, "eval_logs.py", "--config", full_cfg, "--ckpt", "runs/paper_sync/temporal/best.pt"])
    run([py, "eval_logs.py", "--config", full_cfg, "--ckpt", "runs/paper_sync/temporal/best.pt", "--agg", "vote"])

    if args.sync_figures:
        sync_figures()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
