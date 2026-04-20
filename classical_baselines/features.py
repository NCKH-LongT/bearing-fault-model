from __future__ import annotations

import numpy as np


def _sanitize_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Expected vibration window with shape (N, 2).")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def vib_stats_8d(vib_window: np.ndarray) -> np.ndarray:
    """
    Simple 8-D vibration baseline:
    [rms_x, std_x, peak_x, crest_x, rms_y, std_y, peak_y, crest_y]

    This is intentionally lightweight and reproducible. It is a sensible
    classical baseline, not a claim that it matches every prior 8-D baseline.
    """
    xw = _sanitize_window(vib_window)
    feats = []
    for ch in range(2):
        x = xw[:, ch]
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        std = float(np.std(x) + 1e-12)
        peak = float(np.max(np.abs(x)))
        crest = float(peak / (rms + 1e-12))
        feats.extend([rms, std, peak, crest])
    return np.asarray(feats, dtype=np.float32)


FEATURE_EXTRACTORS = {
    "vib_stats_8d": vib_stats_8d,
}


def resolve_feature_extractor(name: str):
    key = (name or "vib_stats_8d").strip().lower()
    if key not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown classical feature extractor: {name}")
    return FEATURE_EXTRACTORS[key]

