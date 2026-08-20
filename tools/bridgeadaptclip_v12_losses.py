"""Native-resolution error-aware gate losses for BridgeAdaptCLIP-v1.2."""

import torch.nn.functional as F


def error_aware_gate_losses(gate_logits, correction, target, base_probability):
    """Return soft gate target, stable BCE loss, and per-image preservation loss."""
    gate_target = (target.float() - base_probability.float()).abs().detach()
    gate_loss = F.binary_cross_entropy_with_logits(
        gate_logits.float(), gate_target
    )
    preserve_weight = 1.0 - gate_target
    correction_abs = correction.float().abs()
    preserve_per_image = (
        (preserve_weight * correction_abs).flatten(1).sum(dim=1)
        / preserve_weight.flatten(1).sum(dim=1).clamp_min(1e-6)
    )
    preserve_loss = preserve_per_image.mean()
    return gate_target, gate_loss, preserve_loss
