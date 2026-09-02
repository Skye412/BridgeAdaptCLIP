"""AP-aligned hard-pixel ranking loss for BridgeAdaptCLIP-v1.5."""

import torch
import torch.nn.functional as F


def hard_pixel_ranking_loss(
    final_logits,
    target,
    hard_positive_count=256,
    hard_negative_count=256,
):
    """Rank each image's lowest defect logits above its highest background logits.

    Images without positive pixels do not contribute. Counts are capped by the
    available support, and the loss is averaged first over all selected pairs
    within an image and then over valid images.
    """
    if hard_positive_count <= 0 or hard_negative_count <= 0:
        raise ValueError('hard pixel counts must be positive')

    logits = final_logits.float().flatten(1)
    target = target.float().flatten(1)
    image_losses = []
    hard_positive_means = []
    hard_negative_means = []

    for image_logits, image_target in zip(logits, target):
        positive_logits = image_logits[image_target > 0.5]
        negative_logits = image_logits[image_target <= 0.5]
        if positive_logits.numel() == 0 or negative_logits.numel() == 0:
            continue

        positive_count = min(hard_positive_count, positive_logits.numel())
        negative_count = min(hard_negative_count, negative_logits.numel())
        hard_positives = torch.topk(
            positive_logits, k=positive_count, largest=False, sorted=False
        ).values
        hard_negatives = torch.topk(
            negative_logits, k=negative_count, largest=True, sorted=False
        ).values
        pairwise_differences = hard_negatives[:, None] - hard_positives[None, :]
        image_losses.append(F.softplus(pairwise_differences).mean())
        hard_positive_means.append(hard_positives.mean())
        hard_negative_means.append(hard_negatives.mean())

    if not image_losses:
        differentiable_zero = logits.sum() * 0.0
        return differentiable_zero, differentiable_zero, differentiable_zero, 0

    return (
        torch.stack(image_losses).mean(),
        torch.stack(hard_positive_means).mean(),
        torch.stack(hard_negative_means).mean(),
        len(image_losses),
    )
