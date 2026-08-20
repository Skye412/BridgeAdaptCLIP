"""Pure helpers for BridgeAdaptCLIP zero-cost diagnostics."""

import os

import torch


FUSION_FORMS = ('linear', 'probability_or')


def fuse_probabilities(row0_probability, structural_probability, fusion_form, weight):
    """Fuse probabilities without modifying either input tensor."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError('weight must be in [0, 1]')
    if weight == 0.0:
        return row0_probability.clone()
    if fusion_form == 'linear':
        return (1.0 - weight) * row0_probability + weight * structural_probability
    if fusion_form == 'probability_or':
        return 1.0 - (1.0 - row0_probability) * (1.0 - weight * structural_probability)
    raise ValueError(f'Unknown fusion form: {fusion_form}')


def select_validation_fusion(candidates):
    """Select by validation P-AP, then P-AUROC, with deterministic ordering."""
    if not candidates:
        raise ValueError('At least one validation candidate is required')
    return max(
        enumerate(candidates),
        key=lambda item: (
            item[1]['metrics_percent']['P-AP'],
            item[1]['metrics_percent']['P-AUROC'],
            -item[0],
        ),
    )[1]


def bridge_source_from_path(path):
    name = os.path.basename(str(path)).lower()
    if name.startswith('codebrim_'):
        return 'CODEBRIM'
    if name.startswith('s2ds_'):
        return 'S2DS'
    raise ValueError(f'Cannot infer Bridge2893 source from {path}')


def fusion_grid():
    return [
        {'form': form, 'weight': step / 10.0}
        for form in FUSION_FORMS
        for step in range(11)
    ]
