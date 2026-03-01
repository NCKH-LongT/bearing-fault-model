"""
Gated fusion head (skeleton) for vibration + temperature.

This module is optional and designed to be drop-in with minimal changes.
Hook it into your model by importing GatedFusionHead and calling it with
vibration logits and raw temperature 6-D features (already preprocessed).
"""
from typing import Optional

import torch
import torch.nn as nn


class GatedFusionHead(nn.Module):
    def __init__(self, temp_in: int = 6, proj_dim: int = 32, num_classes: int = 3,
                 modality_dropout_p: float = 0.3):
        super().__init__()
        self.modality_dropout_p = modality_dropout_p
        self.proj_t = nn.Linear(temp_in, proj_dim)
        self.gate = nn.Linear(temp_in, 1)
        self.head_t = nn.Linear(proj_dim, num_classes)

    def forward(self, logits_v: torch.Tensor, temp_feat: torch.Tensor,
                train: Optional[bool] = None) -> torch.Tensor:
        # modality dropout (training only)
        if train is None:
            train = self.training
        if train and self.modality_dropout_p > 0.0 and torch.rand(()) < self.modality_dropout_p:
            temp_feat = torch.zeros_like(temp_feat)

        proj_t = self.proj_t(temp_feat)
        alpha = torch.sigmoid(self.gate(temp_feat))  # (B,1)
        logits_t = self.head_t(proj_t)
        return logits_v + alpha * logits_t

