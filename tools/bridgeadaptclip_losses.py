"""Losses for the BridgeAdaptCLIP-v1 binary native-resolution decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Native-1024 reductions and elementwise losses must remain FP32 under AMP.
        logits = logits.float()
        targets = targets.float()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * ce).mean()


class BinaryDiceLossWithLogits(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # A 1024x1024 FP16 sum can overflow even for probabilities near 0.5.
        probabilities = torch.sigmoid(logits.float()).flatten(1)
        targets = targets.float().flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()
