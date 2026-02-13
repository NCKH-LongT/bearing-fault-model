import numpy as np


def temp_stats_window(temp_win: np.ndarray) -> np.ndarray:
    """
    temp_win: (win,2) -> columns: [temp_bearing, temp_atm]
    Returns 6-dim features: [mean_b, std_b, slope_b, mean_a, std_a, slope_a]
    slope via linear regression on index
    """
    if temp_win.ndim != 2 or temp_win.shape[1] != 2:
        raise ValueError("temp_win must be (win,2)")

    n = temp_win.shape[0]
    t = np.arange(n, dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)

    feats = []
    for ch in range(2):
        x = temp_win[:, ch].astype(np.float32)
        mean = float(x.mean())
        std = float(x.std() + 1e-8)
        # slope via least squares: slope = cov(t,x) / var(t)
        slope = float(((t * x).mean() - t.mean() * x.mean()) / (t.var() + 1e-8))
        feats.extend([mean, std, slope])

    return np.array(feats, dtype=np.float32)

