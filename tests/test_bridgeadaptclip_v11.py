import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL_MODULE = load_module('bridgeadaptclip_v11_standalone', 'adaptcliplib/bridgeadaptclip.py')
ROW0_MODULE = (
    load_module('bridge_row0_standalone', 'tools/bridge_row0.py')
    if gaussian_filter is not None else None
)
BridgeAdaptCLIPV11 = MODEL_MODULE.BridgeAdaptCLIPV11


class BridgeAdaptCLIPV11Tests(unittest.TestCase):
    def make_model(self):
        return BridgeAdaptCLIPV11(
            semantic_channels=8,
            fusion_channels=8,
            structural_channels=8,
            structural_input_size=32,
        )

    def make_inputs(self):
        return (
            torch.randn(2, 8, 3, 3),
            torch.rand(2, 1, 32, 32) * 0.8 + 0.1,
            torch.randn(2, 3, 32, 32),
        )

    def test_zero_initialization_reproduces_row0_logits_exactly(self):
        model = self.make_model().eval()
        semantic, row0_probability, structural = self.make_inputs()
        output = model(semantic, row0_probability, structural)
        expected_logits = torch.logit(row0_probability)
        self.assertTrue(torch.equal(output['residual'], torch.zeros_like(output['residual'])))
        self.assertTrue(torch.equal(
            output['gated_residual'], torch.zeros_like(output['gated_residual'])
        ))
        self.assertTrue(torch.equal(output['mask_logits'], expected_logits))
        self.assertTrue(torch.allclose(
            output['gate'],
            torch.full_like(output['gate'], torch.sigmoid(torch.tensor(-4.0))),
        ))

    def test_gate_joint_input_contains_structure_semantics_and_row0(self):
        model = self.make_model()
        first_conv = model.joint_projection[0]
        self.assertEqual(first_conv.in_channels, 8 + 8 + 1)
        output = model(*self.make_inputs())
        self.assertEqual(tuple(output['joint_feature'].shape), (2, 8, 8, 8))
        self.assertEqual(tuple(output['gate'].shape), (2, 1, 32, 32))

    def test_residual_is_bidirectional_and_formula_is_logit_addition(self):
        model = self.make_model().eval()
        semantic, row0_probability, structural = self.make_inputs()
        with torch.no_grad():
            model.residual_head.bias.fill_(2.0)
        positive = model(semantic, row0_probability, structural)
        self.assertTrue(torch.all(positive['mask_logits'] > positive['row0_logits']))
        self.assertTrue(torch.allclose(
            positive['mask_logits'],
            positive['row0_logits'] + positive['gate'] * positive['residual'],
        ))
        with torch.no_grad():
            model.residual_head.bias.fill_(-2.0)
        negative = model(semantic, row0_probability, structural)
        self.assertTrue(torch.all(negative['mask_logits'] < negative['row0_logits']))

    def test_trainable_path_receives_gradient_after_residual_head_opens(self):
        model = self.make_model()
        with torch.no_grad():
            model.residual_head.weight.normal_(std=0.01)
        semantic, row0_probability, structural = self.make_inputs()
        output = model(semantic, row0_probability, structural)
        output['mask_logits'].mean().backward()
        self.assertGreater(float(model.residual_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.structural_stem[0][0].weight.grad.abs().sum()), 0.0)

    @unittest.skipIf(gaussian_filter is None, 'SciPy is not installed locally')
    def test_row0_smoothing_matches_reference_scipy_pipeline(self):
        visual = torch.rand(2, 2, 12, 12)
        textual = torch.rand(2, 2, 12, 12)
        actual = ROW0_MODULE.smooth_row0_probability(visual, textual, sigma=4.0)
        fused = 0.5 * (visual[:, 1] + textual[:, 1])
        expected = torch.stack([
            torch.from_numpy(gaussian_filter(item.numpy(), sigma=4.0))
            for item in fused
        ])
        self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
