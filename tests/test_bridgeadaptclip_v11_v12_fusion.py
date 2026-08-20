import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'bridge_model_fusion_standalone', ROOT / 'tools' / 'bridge_model_fusion.py'
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FUSION_FORMS = MODULE.FUSION_FORMS
fuse_model_logits = MODULE.fuse_model_logits


def test_fusion_endpoints_and_midpoints():
    logits_v11 = torch.tensor([[-2.0, 2.0]])
    logits_v12 = torch.tensor([[1.0, -1.0]])
    probability_v11 = torch.sigmoid(logits_v11)
    probability_v12 = torch.sigmoid(logits_v12)

    for form in FUSION_FORMS:
        assert torch.allclose(
            fuse_model_logits(logits_v11, logits_v12, form, 0.0), probability_v11
        )
        assert torch.allclose(
            fuse_model_logits(logits_v11, logits_v12, form, 1.0), probability_v12
        )

    expected_probability = 0.5 * (probability_v11 + probability_v12)
    expected_logit = torch.sigmoid(0.5 * (logits_v11 + logits_v12))
    assert torch.allclose(
        fuse_model_logits(logits_v11, logits_v12, 'probability_linear', 0.5),
        expected_probability,
    )
    assert torch.allclose(
        fuse_model_logits(logits_v11, logits_v12, 'logit_linear', 0.5),
        expected_logit,
    )


def test_fusion_rejects_invalid_weight():
    logits = torch.zeros(1)
    try:
        fuse_model_logits(logits, logits, 'logit_linear', 1.1)
    except ValueError as error:
        assert 'weight' in str(error)
    else:
        raise AssertionError('Expected invalid fusion weight to fail')
