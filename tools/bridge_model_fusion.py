"""Pure probability/logit fusion helpers for BridgeAdaptCLIP variants."""

import torch


FUSION_FORMS = ('probability_linear', 'logit_linear')


def fuse_model_logits(logits_v11, logits_v12, form, weight):
    if not 0.0 <= weight <= 1.0:
        raise ValueError('weight must be in [0, 1]')
    logits_v11 = logits_v11.float()
    logits_v12 = logits_v12.float()
    if form == 'logit_linear':
        return torch.sigmoid((1.0 - weight) * logits_v11 + weight * logits_v12)
    if form == 'probability_linear':
        return (
            (1.0 - weight) * torch.sigmoid(logits_v11)
            + weight * torch.sigmoid(logits_v12)
        )
    raise ValueError(f'Unknown fusion form: {form}')
