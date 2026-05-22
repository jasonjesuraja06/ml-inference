"""Loss functions used by the improved trainer."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with class weighting (Lin et al. 2017)."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE with label smoothing -> per-sample
        log_probs = F.log_softmax(logits, dim=-1)
        if self.label_smoothing > 0:
            n = logits.size(-1)
            with torch.no_grad():
                smooth = torch.full_like(log_probs, self.label_smoothing / (n - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(smooth * log_probs).sum(dim=-1)
        else:
            ce = F.nll_loss(log_probs, targets, reduction="none")

        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            a = self.alpha.to(logits.device)[targets]
            focal = a * focal
        return focal.mean()
