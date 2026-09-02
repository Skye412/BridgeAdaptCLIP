"""Skeleton-balanced hard-positive ranking for BridgeAdaptCLIP-v1.6."""

import torch
import torch.nn.functional as F


def _bottom_k(logits, indices, count):
    if indices.numel() == 0 or count <= 0:
        return indices[:0]
    count = min(count, indices.numel())
    values = logits[indices]
    offsets = torch.topk(values, k=count, largest=False, sorted=False).indices
    return indices[offsets]


def skeleton_balanced_hard_pixel_ranking_loss(
    final_logits,
    target,
    skeleton,
    global_positive_count=128,
    skeleton_positive_count=128,
    hard_negative_count=256,
):
    """Balance hard-positive ranking between full masks and thin skeletons.

    The skeleton pool is filled first from skeleton pixels. If it has fewer
    than the requested count, the lowest-logit non-skeleton positive pixels
    supplement it. Global and skeleton-balanced terms are averaged equally.
    """
    if min(global_positive_count, skeleton_positive_count, hard_negative_count) <= 0:
        raise ValueError('all hard pixel counts must be positive')

    logits = final_logits.float().flatten(1)
    target = target.float().flatten(1)
    skeleton = skeleton.float().flatten(1)
    image_losses = []
    global_losses = []
    thin_losses = []
    global_positive_means = []
    thin_positive_means = []
    hard_negative_means = []
    selected_skeleton_counts = []

    for image_logits, image_target, image_skeleton in zip(logits, target, skeleton):
        positive_mask = image_target > 0.5
        negative_indices = torch.nonzero(~positive_mask, as_tuple=False).flatten()
        positive_indices = torch.nonzero(positive_mask, as_tuple=False).flatten()
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        skeleton_mask = (image_skeleton > 0.5) & positive_mask
        skeleton_indices = torch.nonzero(skeleton_mask, as_tuple=False).flatten()
        non_skeleton_indices = torch.nonzero(
            positive_mask & ~skeleton_mask, as_tuple=False
        ).flatten()

        global_indices = _bottom_k(
            image_logits, positive_indices, global_positive_count
        )
        skeleton_selected = _bottom_k(
            image_logits, skeleton_indices, skeleton_positive_count
        )
        supplement_count = skeleton_positive_count - skeleton_selected.numel()
        supplement = _bottom_k(
            image_logits, non_skeleton_indices, supplement_count
        )
        thin_indices = torch.cat([skeleton_selected, supplement])
        if thin_indices.numel() == 0:
            thin_indices = global_indices

        negative_count = min(hard_negative_count, negative_indices.numel())
        negative_values = image_logits[negative_indices]
        negative_offsets = torch.topk(
            negative_values, k=negative_count, largest=True, sorted=False
        ).indices
        hard_negatives = image_logits[negative_indices[negative_offsets]]
        global_positives = image_logits[global_indices]
        thin_positives = image_logits[thin_indices]

        global_loss = F.softplus(
            hard_negatives[:, None] - global_positives[None, :]
        ).mean()
        thin_loss = F.softplus(
            hard_negatives[:, None] - thin_positives[None, :]
        ).mean()
        global_losses.append(global_loss)
        thin_losses.append(thin_loss)
        image_losses.append(0.5 * (global_loss + thin_loss))
        global_positive_means.append(global_positives.mean())
        thin_positive_means.append(thin_positives.mean())
        hard_negative_means.append(hard_negatives.mean())
        selected_skeleton_counts.append(
            image_logits.new_tensor(float(skeleton_selected.numel()))
        )

    if not image_losses:
        zero = logits.sum() * 0.0
        return zero, zero, zero, zero, zero, zero, zero, 0

    return (
        torch.stack(image_losses).mean(),
        torch.stack(global_losses).mean(),
        torch.stack(thin_losses).mean(),
        torch.stack(global_positive_means).mean(),
        torch.stack(thin_positive_means).mean(),
        torch.stack(hard_negative_means).mean(),
        torch.stack(selected_skeleton_counts).mean(),
        len(image_losses),
    )
