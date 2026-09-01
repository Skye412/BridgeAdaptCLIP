"""Final-logit, error-weighted margin loss for BridgeAdaptCLIP-v1.4."""

import torch.nn.functional as F


def final_logit_margin_loss(final_logits, target, error_weight, margin=1.0):
    """Apply balanced FN/FP margins directly to the final anomaly logits."""
    final_logits = final_logits.float()
    target = target.float()
    error_weight = error_weight.float().detach()

    positive_weight = target * error_weight
    negative_weight = (1.0 - target) * error_weight
    fn_margin_loss = (
        positive_weight * F.softplus(float(margin) - final_logits)
    ).sum() / positive_weight.sum().clamp_min(1e-6)
    fp_margin_loss = (
        negative_weight * F.softplus(float(margin) + final_logits)
    ).sum() / negative_weight.sum().clamp_min(1e-6)
    margin_loss = 0.5 * (fn_margin_loss + fp_margin_loss)
    return margin_loss, fn_margin_loss, fp_margin_loss
