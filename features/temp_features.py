import numpy as np


def temp_stats_window(temp_win: np.ndarray) -> np.ndarray:
    """
    temp_win: (win,2) -> columns: [temp_bearing, temp_atm]
    Returns 6-dim features: [mean_b, std_b, slope_b, mean_a, std_a, slope_a]
    slope via linear regression on index
    """
    if temp_win.ndim != 2 or temp_win.shape[1] != 2:
        raise ValueError("temp_win must be (win,2)")

    # sanitize input (replace NaN/Inf with local finite values or zeros)
    xw = np.nan_to_num(temp_win.astype(np.float32), nan=np.nan, posinf=np.nan, neginf=np.nan)
    n = xw.shape[0]
    t = np.arange(n, dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)

    feats = []
    for ch in range(2):
        x = xw[:, ch]
        mask = np.isfinite(x)
        if not mask.any():
            feats.extend([0.0, 0.0, 0.0])
            continue
        xf = x[mask]
        tf = t[mask]
        mean = float(np.mean(xf))
        std = float(np.std(xf) + 1e-8)
        # slope via least squares on finite subset
        t_mean = float(tf.mean())
        x_mean = float(xf.mean())
        cov = float(((tf * xf).mean() - t_mean * x_mean))
        slope = float(cov / (tf.var() + 1e-8))
        feats.extend([mean, std, slope])

    return np.array(feats, dtype=np.float32)


def temp_stats_window_with_diff(temp_win: np.ndarray) -> np.ndarray:
    """
    Extension of temp_stats_window that also includes the temperature difference channel.

    Input:
      temp_win: (win,2) with columns [temp_bearing, temp_atm]

    Output (9-D):
      [mean_b, std_b, slope_b, mean_a, std_a, slope_a, mean_diff, std_diff, slope_diff]
    """
    if temp_win.ndim != 2 or temp_win.shape[1] != 2:
        raise ValueError("temp_win must be (win,2)")

    diff = (temp_win[:, 0] - temp_win[:, 1]).reshape(-1, 1)
    tmp = np.concatenate([temp_win, diff], axis=1)  # (win,3)

    xw = np.nan_to_num(tmp.astype(np.float32), nan=np.nan, posinf=np.nan, neginf=np.nan)
    n = xw.shape[0]
    t = np.arange(n, dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)

    feats = []
    for ch in range(3):
        x = xw[:, ch]
        mask = np.isfinite(x)
        if not mask.any():
            feats.extend([0.0, 0.0, 0.0])
            continue
        xf = x[mask]
        tf = t[mask]
        mean = float(np.mean(xf))
        std = float(np.std(xf) + 1e-8)
        t_mean = float(tf.mean())
        x_mean = float(xf.mean())
        cov = float(((tf * xf).mean() - t_mean * x_mean))
        slope = float(cov / (tf.var() + 1e-8))
        feats.extend([mean, std, slope])

    return np.array(feats, dtype=np.float32)


def resolve_temp_feature(feature_type: str):
    """
    Resolve temperature feature extractor by name.

    Known types:
      - "stats6": temp_stats_window (6-D)
      - "stats9_diff": temp_stats_window_with_diff (9-D)
    """
    ft = (feature_type or "stats6").strip().lower()
    if ft == "stats6":
        return temp_stats_window, 6
    if ft in {"stats9_diff", "stats9", "diff9", "stats_with_diff"}:
        return temp_stats_window_with_diff, 9
    raise ValueError(f"Unknown temp feature type: {feature_type}")
