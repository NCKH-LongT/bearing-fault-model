from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def stft_log_spectrogram(
    vib_win: np.ndarray,
    n_fft: int = 4096,
    hop_length: int = 1024,
    window: str = "hann",
    log_add: float = 1.0,
    target_size: Tuple[int, int] = (224, 224),
):
    """
    vib_win: (win, 2) float32 waveform for two axes (x,y)
    Return: tensor (2, H, W) log-spectrogram resized to target_size
    """
    if vib_win.ndim != 2 or vib_win.shape[1] != 2:
        raise ValueError("vib_win must be (win,2)")

    # z-score per channel
    x = (vib_win - vib_win.mean(axis=0, keepdims=True)) / (vib_win.std(axis=0, keepdims=True) + 1e-8)
    x = torch.tensor(x.T, dtype=torch.float32)  # (2, win)

    window_t = torch.hann_window(n_fft) if window == "hann" else torch.ones(n_fft)
    S = []
    for c in range(2):
        st = torch.stft(x[c], n_fft=n_fft, hop_length=hop_length, window=window_t, return_complex=True)
        mag = st.abs()  # (freq, frames)
        mag = torch.log1p(mag * log_add)
        # per-frequency normalization
        mag = (mag - mag.mean(dim=1, keepdim=True)) / (mag.std(dim=1, keepdim=True) + 1e-6)
        S.append(mag)
    S = torch.stack(S, dim=0)  # (2, F, T)

    # Resize to target_size (H,W)
    S = S.unsqueeze(0)  # (1,2,F,T)
    S = F.interpolate(S, size=target_size, mode="bilinear", align_corners=False)
    S = S.squeeze(0)
    return S  # (2,H,W)


class SpectrogramTransform:
    def __init__(self, n_fft=4096, hop_length=1024, window="hann", log_add=1.0, target_size=(224, 224), training=False):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.log_add = log_add
        self.target_size = target_size
        self.training = training

    def __call__(self, vib_win: np.ndarray):
        return stft_log_spectrogram(
            vib_win,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            log_add=self.log_add,
            target_size=self.target_size,
        )

