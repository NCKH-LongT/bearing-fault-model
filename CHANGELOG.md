# Changelog

## v0.1-paper-sync (2026-03-01)

Highlights
- Align paper (main.tex) to r49 settings: STFT 2048/512, input 160×160, AdamW lr=2e-4, class-weights + smoothing, val_max_windows=0.
- Add temporal full-range (0–100.1%) results (mean and vote) with figures and a summary table.
- Copy evaluated figures under `figures/` for Overleaf sync (temporal, temporal_alltest, ablation sets).
- Add r49-aligned ablation configs: vib-only, no class-weights, high-res.
- Add drafts/related_and_results_r49.md with concise Related Works + Results text.

Notable paths
- Configs: `configs/exp_r49*.yaml`
- Figures: `figures/temporal/*`, `figures/temporal_alltest/*`, `figures/ablation/*`
- Paper: `main.tex`
- Draft: `drafts/related_and_results_r49.md`

Tag
- Git tag: `v0.1-paper-sync`

