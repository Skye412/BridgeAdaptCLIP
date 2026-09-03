"""Positive-prioritized hard-pixel ranking for BridgeAdaptCLIP-v1.7."""

import torch
import torch.nn.functional as F


def positive_prioritized_ranking_loss(
    final_logits,
    target,
    hard_positive_count=256,
    hard_negative_count=256,
    raise_positive_weight=0.8,
):
    """Preserve pairwise ranking values while routing most gradient to positives."""
    if hard_positive_count <= 0 or hard_negative_count <= 0:
        raise ValueError('hard pixel counts must be positive')
    if not 0.0 <= raise_positive_weight <= 1.0:
        raise ValueError('raise_positive_weight must be in [0, 1]')

    logits = final_logits.float().flatten(1)
    target = target.float().flatten(1)
    suppress_negative_weight = 1.0 - raise_positive_weight
    image_losses = []
    raise_losses = []
    suppress_losses = []
    hard_positive_means = []
    hard_negative_means = []

    for image_logits, image_target in zip(logits, target):
        positives = image_logits[image_target > 0.5]
        negatives = image_logits[image_target <= 0.5]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue

        positive_count = min(hard_positive_count, positives.numel())
        negative_count = min(hard_negative_count, negatives.numel())
        hard_positives = torch.topk(
            positives, k=positive_count, largest=False, sorted=False
        ).values
        hard_negatives = torch.topk(
            negatives, k=negative_count, largest=True, sorted=False
        ).values

        raise_positive = F.softplus(
            hard_negatives.detach()[:, None] - hard_positives[None, :]
        ).mean()
        suppress_negative = F.softplus(
            hard_negatives[:, None] - hard_positives.detach()[None, :]
        ).mean()
        image_losses.append(
            raise_positive_weight * raise_positive
            + suppress_negative_weight * suppress_negative
        )
        raise_losses.append(raise_positive)
        suppress_losses.append(suppress_negative)
        hard_positive_means.append(hard_positives.mean())
        hard_negative_means.append(hard_negatives.mean())

    if not image_losses:
        zero = logits.sum() * 0.0
        return zero, zero, zero, zero, zero, 0

    return (
        torch.stack(image_losses).mean(),
        torch.stack(raise_losses).mean(),
        torch.stack(suppress_losses).mean(),
        torch.stack(hard_positive_means).mean(),
        torch.stack(hard_negative_means).mean(),
        len(image_losses),
    )
