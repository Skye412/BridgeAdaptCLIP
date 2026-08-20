"""Signed, error-weighted correction loss for BridgeAdaptCLIP-v1.3."""

import torch
import torch.nn.functional as F


def _valid_per_image_mean(weighted_values, weights):
    numerators = weighted_values.flatten(1).sum(dim=1)
    denominators = weights.flatten(1).sum(dim=1)
    valid = denominators > 1e-6
    if valid.any():
        return (numerators[valid] / denominators[valid]).mean()
    # Keep a differentiable zero when a minibatch has no support for this term.
    return weighted_values.sum() * 0.0


def signed_error_correction_loss(correction, target, error_weight):
    """Encourage positive defect corrections and negative background corrections.

    Each direction is normalized per image and then averaged over images with
    non-zero support. The two available directional terms are balanced equally.
    """
    correction = correction.float()
    target = target.float()
    error_weight = error_weight.float().detach()

    positive_weight = error_weight * target
    negative_weight = error_weight * (1.0 - target)
    positive_loss = _valid_per_image_mean(
        positive_weight * F.softplus(-correction), positive_weight
    )
    negative_loss = _valid_per_image_mean(
        negative_weight * F.softplus(correction), negative_weight
    )

    has_positive = bool((positive_weight.flatten(1).sum(dim=1) > 1e-6).any())
    has_negative = bool((negative_weight.flatten(1).sum(dim=1) > 1e-6).any())
    if has_positive and has_negative:
        total = 0.5 * (positive_loss + negative_loss)
    elif has_positive:
        total = positive_loss
    elif has_negative:
        total = negative_loss
    else:
        total = correction.sum() * 0.0
    return total, positive_loss, negative_loss
