import torch

from adaptcliplib.bridgeadaptclip import BridgeAdaptCLIPV12
from adaptcliplib.supervised_baselines import DeepLabV3PlusResNet50
from tools.supervised_protocol import BinaryProtocolMetrics


def test_fine_control_variants_preserve_native_output_shape():
    semantic = torch.randn(1, 768, 37, 37)
    row0 = torch.full((1, 1, 64, 64), 0.2)
    structural = torch.randn(1, 3, 64, 64)
    for variant in ('strip', 'square', 'semantic_only'):
        model = BridgeAdaptCLIPV12(
            fusion_channels=16, structural_channels=16,
            structural_input_size=64, structural_variant=variant,
        )
        output = model(semantic, row0, structural)
        assert output['mask_logits'].shape == (1, 1, 64, 64)
        if variant == 'semantic_only':
            assert torch.count_nonzero(output['structural_feature']) == 0


def test_deeplabv3plus_is_native_resolution_and_has_low_level_decoder():
    model = DeepLabV3PlusResNet50(pretrained=False).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 128, 128))
    assert output.shape == (1, 1, 128, 128)
    assert model.low_projection[0].in_channels == 256
    assert model.decoder[0].in_channels == 304


def test_streaming_protocol_metrics_are_finite():
    metric = BinaryProtocolMetrics(bins=64)
    prediction = torch.tensor([[[[0.1, 0.9], [0.2, 0.8]]]])
    target = torch.tensor([[[[0, 1], [0, 1]]]])
    metric.update(prediction, target, torch.tensor([1]))
    # Add a normal image so image AUROC is defined.
    metric.update(torch.full_like(prediction, 0.1), torch.zeros_like(target), torch.tensor([0]))
    result = metric.compute()
    assert result['P-AP'] > 99
    assert result['P-AUROC'] > 99
