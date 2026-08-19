import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL_MODULE = load_module('bridgeadaptclip_model_standalone', 'adaptcliplib/bridgeadaptclip.py')
LOSS_MODULE = load_module('bridgeadaptclip_losses_standalone', 'tools/bridgeadaptclip_losses.py')
BridgeAdaptCLIPV1 = MODEL_MODULE.BridgeAdaptCLIPV1
DEGConvLite = MODEL_MODULE.DEGConvLite
BinaryFocalLossWithLogits = LOSS_MODULE.BinaryFocalLossWithLogits
BinaryDiceLossWithLogits = LOSS_MODULE.BinaryDiceLossWithLogits


class BridgeAdaptCLIPV1Tests(unittest.TestCase):
    def test_forward_shape_and_residual_spatial_refinement(self):
        model = BridgeAdaptCLIPV1(
            semantic_channels=16,
            fusion_channels=8,
            structural_channels=8,
            strip_kernel=5,
            structural_input_size=64,
        )
        output = model(
            torch.randn(2, 16, 5, 5),
            torch.rand(2, 1, 20, 20),
            torch.rand(2, 1, 20, 20),
            torch.randn(2, 3, 64, 64),
        )
        self.assertEqual(tuple(output['mask_logits'].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(output['spatial_attention'].shape), (2, 1, 16, 16))
        self.assertTrue(torch.all(output['spatial_attention'] >= 0))
        self.assertTrue(torch.all(output['spatial_attention'] <= 1))
        expected = (1.0 + output['spatial_attention']) * output['semantic_up']
        self.assertTrue(torch.allclose(output['refined_semantic'], expected))

    def test_gradients_reach_semantic_and_structural_inputs(self):
        model = BridgeAdaptCLIPV1(
            semantic_channels=8,
            fusion_channels=8,
            structural_channels=8,
            structural_input_size=32,
        )
        semantic = torch.randn(1, 8, 3, 3, requires_grad=True)
        structural = torch.randn(1, 3, 32, 32, requires_grad=True)
        output = model(
            semantic,
            torch.rand(1, 1, 12, 12),
            torch.rand(1, 1, 12, 12),
            structural,
        )
        output['mask_logits'].mean().backward()
        self.assertIsNotNone(semantic.grad)
        self.assertIsNotNone(structural.grad)
        self.assertGreater(float(semantic.grad.abs().sum()), 0.0)
        self.assertGreater(float(structural.grad.abs().sum()), 0.0)

    def test_degconv_requires_odd_strip_kernel(self):
        with self.assertRaisesRegex(ValueError, 'odd'):
            DEGConvLite(channels=8, strip_kernel=4)

    def test_binary_losses_are_finite_and_differentiable(self):
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, :, 4:8, 5:9] = 1
        focal = BinaryFocalLossWithLogits(alpha=0.75, gamma=2.0)(logits, targets)
        dice = BinaryDiceLossWithLogits()(logits, targets)
        loss = focal + dice
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
