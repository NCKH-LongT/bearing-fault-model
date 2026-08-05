#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python. Create/activate the paper environment first." >&2
  exit 1
fi

active_jobs="$(ps -C python -C python3 -C python3.14 -o pid=,args= \
  | grep -E 'train_logs\.py|eval_logs\.py|paper/run_revision\.py primary' || true)"
if [[ -n "$active_jobs" ]]; then
  echo "Another paper train/eval process is running. Wait for it to finish before starting the queue." >&2
  echo "$active_jobs" >&2
  exit 2
fi

source .venv/bin/activate

multimodal_root="${REVISION_MULTIMODAL_ROOT:-runs/revision_search_5060}"
vibration_root="${REVISION_VIBRATION_ROOT:-runs/revision_search_vibration_5060}"

python scripts/cache_dataset_npy.py --workers 2
python paper/run_revision.py preflight
python paper/run_revision.py smoke
python paper/run_revision.py audit

# Hyperparameters are selected only from validation metrics. Test is not evaluated here.
python scripts/search_revision_5060.py search \
  --base-config configs/revision_search_5060.yaml \
  --output-root "$multimodal_root" \
  --trials 24 \
  --continue

# Independently tune the matched vibration-only baseline on the same validation split.
python scripts/search_revision_5060.py search \
  --base-config configs/revision_search_vibration_5060.yaml \
  --output-root "$vibration_root" \
  --trials 16 \
  --continue

echo "Validation searches finished. Inspect both leaderboards before running locked confirmation."
echo "Multimodal: $multimodal_root/leaderboard.csv"
echo "Vibration:  $vibration_root/leaderboard.csv"
