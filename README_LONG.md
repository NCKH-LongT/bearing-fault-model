# Bearing Fault Classification — Full Pipeline Guide

Dataset in `data/` contains hourly CSV logs (4 columns per file):
- Column order: vibration (x), vibration (y), temperature (bearing), temperature (atmospheric)
- Sampling rate: 25,600 Hz; duration: 78.125 s per file (~2,000,000 rows)
- Goal: Train a classifier to detect/identify bearing faults over run-to-failure logs (TTF)

This guide lists practical, reproducible steps from data checks to a working model.

## End-to-End Steps
1) Verify data (structure, sampling, integrity)
2) Create labels manifest (fault type or health stage) per file
3) Define temporal splits (train/val/test) without leakage
4) Window the signals (1.0 s, hop 0.5 s)
5) Build features for vibration (STFT log-spectrogram)
6) Build features for temperature (mean/std/slope per window)
7) Select model (2D CNN on spectrogram + temperature fusion)
8) Train with class weights + OneCycleLR; log metrics
9) Evaluate (macro-F1, confusion matrix, early/late TTF)
10) Inference script to classify new logs

Below are detailed, actionable instructions for steps (1) and (2). Subsequent steps are outlined with concrete hyperparameters to implement next.

---

## Step 1 — Verify Data
Objective: confirm 4 numeric columns, expected row count, no NaN, basic stats.

Choose a representative file (replace path to your file name as needed):

PowerShell quick checks
- Preview first lines:
  ```powershell
  Get-Content -TotalCount 5 "data/LogFile_2022-06-20-17-00-31.csv"
  ```
- Check there are 4 comma-separated columns:
  ```powershell
  Get-Content -TotalCount 10 "data/LogFile_2022-06-20-17-00-31.csv" |
    ForEach-Object { ($_ -split ',').Length } | Sort-Object -Unique
  # Expect output: 4
  ```
- Count rows (~ 2,000,000 expected; 25,600 Hz × 78.125 s):
  ```powershell
  $lines = (Get-Content -ReadCount 500000 "data/LogFile_2022-06-20-17-00-31.csv" | Measure-Object -Line).Lines
  $lines
  # Optional: verify duration in seconds (≈78.125)
  $duration = $lines / 25600.0; $duration
  ```
- Check for NaN/Inf patterns (sampling first 100k lines for speed):
  ```powershell
  Get-Content -TotalCount 100000 "data/LogFile_2022-06-20-17-00-31.csv" |
    Select-String -SimpleMatch "NaN","nan","Inf","inf"
  ```

Python quick stats (optional but informative)
- Requires `pandas`:
  ```powershell
  python -c "import pandas as pd,sys; f=sys.argv[1]; df=pd.read_csv(f,header=None); \
  print(df.shape); print(df.isna().sum().tolist()); print(df.describe().T)" \
  data/LogFile_2022-06-20-17-00-31.csv
  ```
Interpretation
- shape ≈ (2000000, 4)
- NaN counts = [0,0,0,0]
- Value ranges consistent with vibration (x,y) and temperatures (bearing, atmospheric)

If anomalies are found
- Drop rows with NaN (if rare) or exclude affected files from training until cleaned.
- Keep a list of excluded files in `data/exclude.txt` (optional) and handle in loader.

---

## Step 2 — Create Labels Manifest
Objective: produce `data/manifest.csv` mapping each file → label(s) and TTF progress.

Case A: Single run-to-failure without explicit fault-type per hour
- Use time progression as proxy for health stage (3-class baseline):
  - healthy: 0–60% of runtime
  - degrading: 60–90%
  - fault: 90–100%
- Compute `ttf_percent` by sorting files chronologically and mapping index → [0,100].

PowerShell one-liner to generate manifest
- This treats all files as one run (`run1`). Edit thresholds or fault_type later if you have detailed labels.
  ```powershell
  $files = Get-ChildItem -File "data/LogFile_*.csv" | Sort-Object Name
  $n = $files.Count
  $rows = for ($i=0; $i -lt $n; $i++) {
    $f = $files[$i]
    $p = if ($n -gt 1) { [math]::Round(($i * 100.0) / ($n - 1), 3) } else { 0.0 }
    $cls = if ($p -lt 60) { 'healthy' } elseif ($p -lt 90) { 'degrading' } else { 'fault' }
    [PSCustomObject]@{ file=$f.Name; run_id='run1'; ttf_percent=$p; fault_type=$cls }
  }
  $rows | Export-Csv -NoTypeInformation -Encoding UTF8 "data/manifest.csv"
  ```
- Inspect the first records:
  ```powershell
  Get-Content -TotalCount 10 "data/manifest.csv"
  ```

Case B: You have specific fault types per period/run
- Replace `fault_type` assignment with your true labels.
- If multiple runs exist, set `run_id` accordingly (e.g., `runA`, `runB`) and compute `ttf_percent` within each run independently.

Manifest schema (CSV)
- Columns: `file,run_id,ttf_percent,fault_type`
- Example rows:
  ```csv
  file,run_id,ttf_percent,fault_type
  LogFile_2022-06-20-17-00-31.csv,run1,0.000,healthy
  LogFile_2022-06-22-08-00-31.csv,run1,98.457,fault
  ```

Validation
- Count per class (sanity check proportions):
  ```powershell
  Import-Csv "data/manifest.csv" | Group-Object fault_type | Select-Object Name,Count
  ```

Notes
- You can refine stage thresholds (e.g., 0–50/50–85/85–100) or move to true fault-type labels later; downstream code reads `manifest.csv` and remains unchanged.

---

## Step 3 — Temporal Splits (no leakage)
- Split by time within `run_id`: e.g., train=first 70%, val=next 10%, test=last 20% based on `ttf_percent`.
- Or stronger: Leave-One-Run-Out if multiple runs exist.

## Step 4 — Windowing
- Window length: 1.0 s (25,600 samples), hop: 0.5 s (12,800 samples).
- Do not mix windows across different files or across different `fault_type` boundaries.

## Step 5 — Vibration Features (STFT)
- Preprocess: detrend, per-file z-score.
- STFT: `n_fft=4096`, `hop_length=1024`, Hann; log(1+|S|).
- Stack channels → 2×F×T; normalize per-frequency band; resize/crop to 224×224.

## Step 6 — Temperature Features
- Per window compute for bearing and atmospheric: mean, std, slope; z-score per run.

## Step 7 — Model
- 2D CNN (ResNet-18 small) with `in_channels=2` for vibration; late-fusion MLP for temperature into classifier head.
- Loss: CrossEntropy with class weights; Optim: AdamW; Scheduler: OneCycleLR.

## Step 8 — Training
- Batch 64–128; epochs 80–150; lr max ~3e-4; weight_decay 1e-2; early stopping on val macro-F1.
- Log accuracy, macro-F1, confusion matrix.

## Step 9 — Evaluation
- Report macro-F1 and per-class F1; confusion matrix.
- Early vs late TTF: evaluate subsets `ttf_percent ∈ [70,90]` and `∈ (90,100]` for early detection performance.

## Step 10 — Inference
- Script to read new CSV(s), window, build features, predict per-window, aggregate per-file by majority/mean logit.

## Next Steps (Implementation Plan)
- Add modules (suggested layout):
  - `datasets/logs_ttf.py` (reads CSV, uses `manifest.csv`, windowing)
  - `features/spectrogram.py` (STFT + normalization)
  - `features/temp_features.py` (temperature stats per window)
  - `models/resnet2d.py` (2D CNN) + `models/fusion_head.py` (temp fusion)
  - `train_logs.py`, `eval_logs.py` (training, evaluation)
  - `configs/logs_stft.yaml` (fs=25600, window=1.0, hop=0.5, n_fft=4096)

Once Steps (1) and (2) are complete and `manifest.csv` exists, we can scaffold code for Steps (3–10).
