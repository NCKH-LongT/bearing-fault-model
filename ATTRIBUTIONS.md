# Attributions and Provenance

This project adapts and customizes components from upstream open‑source work. We acknowledge and thank the authors and communities below.

Upstream Implementation (Primary)

- Repository: CNN‑for‑Paderborn‑Bearing‑Dataset (mdzalfirdausi)
- URL: https://github.com/mdzalfirdausi/CNN-for-Paderborn-Bearing-Dataset
- Accessed: 2026‑02‑20 (not pinned to a specific commit in this repo; please pin a hash for archival reproducibility)
- Usage in this repo:
  - Inspiration for CNN‑based bearing diagnosis setup and basic training/evaluation scaffolding.
  - We modified/extended the pipeline to target run‑to‑failure logs with: (i) TTF‑aligned temporal split to mitigate leakage, (ii) two‑axis STFT log‑spectrograms with per‑window waveform z‑score and per‑frequency frame‑wise z‑score, (iii) 6‑D temperature feature fusion via a lightweight head, and (iv) file‑wise aggregation via mean pre‑softmax logits for deployment‑oriented predictions.

Model Architecture and Code Paths

- Vibration encoder: small ResNet‑18‑style CNN (`models/resnet2d.py`) with reduced widths (32–256), input channels=2.
- Spectrogram transform: `features/spectrogram.py` (STFT + log1p + per‑frequency normalization + resize).
- Temperature features: `features/temp_features.py` (6‑D per‑window stats: mean/std/slope for two channels).
- Dataset/splits: `datasets/logs_ttf.py` (file‑wise stratified and TTF‑aligned temporal splits; synchronous windowing of vibration/temperature).

Third‑party Dependencies

- PyTorch, NumPy, SciPy, scikit‑learn, Matplotlib. Refer to each project’s LICENSE.

How to Cite (Upstream)

Use the upstream repository’s recommended citation. A BibTeX entry used in our paper:

```
@misc{mdzalfirdausi_paderborn_repo,
  author       = {mdzalfirdausi},
  title        = {CNN-for-Paderborn-Bearing-Dataset},
  howpublished = {GitHub repository},
  year         = {2019},
  note         = {Accessed: 2026-02-20},
  url          = {https://github.com/mdzalfirdausi/CNN-for-Paderborn-Bearing-Dataset}
}
```

License Notes

- Ensure compliance with the upstream license of the above repository. If no explicit license is present upstream, seek permission or limit reuse accordingly. For permissive licenses (MIT/BSD/Apache‑2.0), retain copyright/NOTICE files.

Data and Paper References

- Dataset references (Jung et al., Data in Brief 2024; Mendeley Data) are included in the paper’s bibliography (`refs.bib`).
