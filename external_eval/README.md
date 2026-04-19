External evaluation workspace (no retraining)

Layout
- `external_eval/data/run2/` and `external_eval/data/run3/`: place your converted CSV files here.
- `external_eval/manifest.csv`: one row per file with columns `file,run_id,ttf_percent,fault_type`.
- `external_eval/config.yaml`: eval-only config pointing to this workspace.

Input CSV format (per file)
- Exactly 4 columns in order: `[vib_x, vib_y, temp_bearing, temp_atm]`.
- If you have only one vibration channel, duplicate it into both `vib_x` and `vib_y`.
- If you have no temperature, set the two temperature columns to 0. Keep `model.use_temp=true` if the checkpoint was trained with temperature.

Manifest
- Column meanings:
  - `file`: path relative to `external_eval/data/`, e.g. `run2/file_0001.csv`.
  - `run_id`: `run2` or `run3`.
  - `ttf_percent`: 0.0–100.1 within each run.
  - `fault_type`: `healthy|degrading|fault`.

Evaluate with an existing checkpoint
1) Copy or export your trained checkpoint, e.g.: `runs/logs_stft_temporal/best.pt`.
2) Adjust `external_eval/config.yaml` if needed (`sampling_rate`, `model.use_temp`).
3) Run:
   python eval_logs.py --config external_eval/config.yaml --ckpt runs/logs_stft_temporal/best.pt

