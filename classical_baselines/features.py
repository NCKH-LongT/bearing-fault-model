from __future__ import annotations

import numpy as np

from features.temp_features import temp_stats_window


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


def vib_stats_26d(vib_window: np.ndarray) -> np.ndarray:
    """Extended 26-D vibration descriptors with shape and band-energy statistics."""
    xw = _sanitize_window(vib_window)
    feats = []
    for ch in range(2):
        x = xw[:, ch].astype(np.float64, copy=False)
        abs_x = np.abs(x)
        mean_abs = float(np.mean(abs_x))
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        std = float(np.std(x) + 1e-12)
        peak = float(np.max(abs_x))
        centered = x - float(np.mean(x))
        normalized = centered / std
        skewness = float(np.mean(normalized ** 3))
        kurtosis = float(np.mean(normalized ** 4))
        crest = float(peak / (rms + 1e-12))
        shape = float(rms / (mean_abs + 1e-12))
        impulse = float(peak / (mean_abs + 1e-12))
        mean_sqrt_abs = float(np.mean(np.sqrt(abs_x)))
        clearance = float(peak / (mean_sqrt_abs ** 2 + 1e-12))
        feats.extend([rms, std, peak, crest, mean_abs, shape, impulse, clearance, skewness, kurtosis])

        spectrum = np.fft.rfft(centered)
        energy = np.abs(spectrum) ** 2
        frequency = np.fft.rfftfreq(len(centered))
        total = float(np.sum(energy) + 1e-12)
        for lower, upper in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5 + 1e-12)):
            mask = (frequency >= lower) & (frequency < upper)
            feats.append(float(np.sum(energy[mask]) / total))
    return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def vib_temp_stats_32d(signal_window: np.ndarray) -> np.ndarray:
    """Concatenate vibration 26-D with bearing/ambient temperature stats 6-D."""
    signal = np.asarray(signal_window, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[1] < 4:
        raise ValueError("Expected multimodal window with at least 4 columns.")
    return np.concatenate([
        vib_stats_26d(signal[:, :2]),
        temp_stats_window(signal[:, 2:4]),
    ]).astype(np.float32, copy=False)


def temp_stats_6d(signal_window: np.ndarray) -> np.ndarray:
    """Temperature-only bearing/ambient mean, std and slope descriptors."""
    signal = np.asarray(signal_window, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[1] < 4:
        raise ValueError("Expected multimodal window with at least 4 columns.")
    return temp_stats_window(signal[:, 2:4])


FEATURE_EXTRACTORS = {
    "vib_stats_8d": vib_stats_8d,
    "vib_stats_26d": vib_stats_26d,
    "vib_temp_stats_32d": vib_temp_stats_32d,
    "temp_stats_6d": temp_stats_6d,
}

FULL_SIGNAL_FEATURES = {"vib_temp_stats_32d", "temp_stats_6d"}


def resolve_feature_extractor(name: str):
    key = (name or "vib_stats_8d").strip().lower()
    if key not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown classical feature extractor: {name}")
    return FEATURE_EXTRACTORS[key]


def feature_uses_full_signal(name: str) -> bool:
    return (name or "").strip().lower() in FULL_SIGNAL_FEATURES
