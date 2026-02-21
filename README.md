# Bearing Health Classification (Run-to-Failure, STFT + Temperature)

This repository contains a lightweight multi-modal pipeline for 3-stage bearing health classification (healthy, degrading, fault) on run-to-failure logs. It uses STFT log-spectrograms from two vibration axes and 6-D temperature descriptors, with a leakage-aware temporal evaluation aligned to the time-to-failure (TTF) axis.

Highlights
- Two-phase training: stratified (development) → temporal fine-tune (deployment realism).
- Vibration features: STFT log-magnitude with per-window z-score and per-frequency normalization; images resized to `input_size`.
- Temperature features: 6-D per window (mean/std/slope for bearing/ambient channels).
- Model: small ResNet-18-style CNN for vibration (in_ch=2) + linear projection for temperature + late-fusion classifier.
- File-wise prediction: average pre-softmax logits across windows (mean-logit) to produce a file label.

Repository Layout
- `configs/`: YAML configs for stratified and temporal runs.
- `datasets/`: dataset loader with TTF-aware splitting (`LogsTTFDataset`).
- `features/`: STFT transform and temperature features.
- `models/`: ResNet2D small with temperature fusion.
- `runs/`: outputs (checkpoints, eval figures/reports).
- `paper_artifacts/`: curated figures/reports for the paper.

Data & Manifest
- Input CSV per file: columns `[vib_x, vib_y, temp_bearing, temp_atm]` sampled at 25.6 kHz.
- `data/manifest.csv` columns: `file, run_id, ttf_percent, fault_type` (labels: healthy/degrading/fault).
- Windowing: `window_seconds=1.0`, `hop_seconds=0.5` (synchronous slices for vibration and temperature).

Setup
- Python 3.10+, PyTorch >= 2.0 recommended.
- Install dependencies via your environment manager; optional: enable CUDA for GPU/AMP.

Quick Start
1) Stratified training (development)
   - `python train_logs.py --config configs/logs_stft_train_strat.yaml`
   - Evaluate: `python eval_logs.py --config configs/logs_stft_train_strat.yaml --ckpt runs/logs_stft_strat/best.pt --show`

2) Temporal fine-tune (deployment-oriented)
   - `python train_logs.py --config configs/logs_stft_full_temporal.yaml`
   - Evaluate: `python eval_logs.py --config configs/logs_stft_full_temporal.yaml --ckpt runs/logs_stft_temporal/best.pt --show`

Key Settings (see configs)
- STFT: `n_fft=4096`, `hop_length=1024`, `window='hann'`, `log_add=1.0`.
- Image size: `input_size=[224,224]` (or `[160,160]` for lighter runs).
- Optimizer: AdamW; label smoothing; AMP; balanced sampling; early stopping.
- Temporal split: train `[0,60]`, val `[60,70]`, test `[70,100.1]` (TTF percent).

Reproducing Paper Figures
- Run training/eval as above; generated artifacts under `runs/.../eval`.
- Curated copies are in `paper_artifacts/` (stratified, temporal, temporal_alltest).

Attribution
- Upstream: CNN-for-Paderborn-Bearing-Dataset (mdzalfirdausi) — https://github.com/mdzalfirdausi/CNN-for-Paderborn-Bearing-Dataset (accessed 2026-02-20).
- We adapt high-level training/evaluation scaffolding and extend it for run-to-failure with: (i) TTF-aligned temporal split to mitigate leakage, (ii) two-axis STFT log-spectrograms with per-window waveform z-score and per-frequency normalization, (iii) 6-D temperature feature fusion via a lightweight head, and (iv) file-wise mean pre-softmax logit aggregation.
- See `ATTRIBUTIONS.md` for detailed provenance and how to cite the original sources.

License
- Please see upstream licenses of any adapted components in `ATTRIBUTIONS.md`. The overall project license will be aligned with upstream requirements (to be finalized by the authors).
