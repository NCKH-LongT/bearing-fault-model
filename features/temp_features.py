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
