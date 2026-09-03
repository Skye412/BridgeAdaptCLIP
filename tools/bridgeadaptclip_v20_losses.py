"""Losses for BridgeAdaptCLIP-v2.0 broad calibration."""

import torch
import torch.nn.functional as F


def broad_gate_and_positive_preservation_losses(
    gate_logits, broad_correction, target, fine_probability
):
    target = target.float()
    fp_target = ((1.0 - target) * fine_probability.detach().float()).detach()
    gate_loss = F.binary_cross_entropy_with_logits(
        gate_logits.float(), fp_target, reduction='mean'
    )
    positive = target.flatten(1)
    correction = broad_correction.float().abs().flatten(1)
    denominators = positive.sum(1)
    valid = denominators > 0
    if valid.any():
        preserve = (
            (positive[valid] * correction[valid]).sum(1)
            / denominators[valid].clamp_min(1e-6)
        ).mean()
    else:
        preserve = broad_correction.float().sum() * 0.0
    return fp_target, gate_loss, preserve


def negative_only_broad_ranking_loss(
    final_logits, fine_logits, target, hard_positive_count=256,
    hard_negative_count=256,
):
    if hard_positive_count <= 0 or hard_negative_count <= 0:
        raise ValueError('hard pixel counts must be positive')
    final_flat = final_logits.float().flatten(1)
    fine_flat = fine_logits.detach().float().flatten(1)
    target_flat = target.float().flatten(1)
    losses, positive_means, negative_means = [], [], []
    for final_image, fine_image, mask in zip(final_flat, fine_flat, target_flat):
        positive_reference = fine_image[mask > 0.5]
        final_negatives = final_image[mask <= 0.5]
        if positive_reference.numel() == 0 or final_negatives.numel() == 0:
            continue
        positives = torch.topk(
            positive_reference,
            k=min(hard_positive_count, positive_reference.numel()),
            largest=False, sorted=False,
        ).values.detach()
        negatives = torch.topk(
            final_negatives,
            k=min(hard_negative_count, final_negatives.numel()),
            largest=True, sorted=False,
        ).values
        losses.append(F.softplus(negatives[:, None] - positives[None, :]).mean())
        positive_means.append(positives.mean())
        negative_means.append(negatives.mean())
    if not losses:
        zero = final_flat.sum() * 0.0
        return zero, zero, zero, 0
    return (
        torch.stack(losses).mean(),
        torch.stack(positive_means).mean(),
        torch.stack(negative_means).mean(),
        len(losses),
    )
