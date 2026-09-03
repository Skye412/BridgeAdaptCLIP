import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'bridgeadaptclip_v21_standalone', ROOT / 'adaptcliplib' / 'bridgeadaptclip.py'
)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
BridgeAdaptCLIPV12 = module.BridgeAdaptCLIPV12
BridgeAdaptCLIPV21Fine = module.BridgeAdaptCLIPV21Fine


class BridgeAdaptCLIPV21FineTests(unittest.TestCase):
    def make_inputs(self, batch=1):
        adapted = torch.randn(batch, 8, 37, 37)
        levels = [torch.randn(batch, 1370, 8) for _ in range(4)]
        row0 = torch.rand(batch, 1, 32, 32) * 0.8 + 0.1
        structural = torch.randn(batch, 3, 32, 32)
        return adapted, levels, row0, structural

    def test_zero_shallow_residuals_match_standard_v13_architecture(self):
        torch.manual_seed(4)
        base = BridgeAdaptCLIPV12(
            semantic_channels=8, fusion_channels=8, structural_channels=8,
            structural_input_size=32,
        )
        multi = BridgeAdaptCLIPV21Fine(
            semantic_channels=8, fusion_channels=8, structural_channels=8,
            structural_input_size=32,
        )
        multi.load_state_dict(base.state_dict(), strict=False)
        inputs = self.make_inputs()
        base_output = base(inputs[0], inputs[2], inputs[3])
        multi_output = multi(*inputs)
        self.assertTrue(torch.equal(base_output['mask_logits'], multi_output['mask_logits']))
        for residual in multi_output['multi_level_residuals']:
            self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_all_three_shallow_branches_receive_gradients(self):
        model = BridgeAdaptCLIPV21Fine(
            semantic_channels=8, fusion_channels=8, structural_channels=8,
            structural_input_size=32,
        )
        with torch.no_grad():
            model.residual_head.weight.fill_(0.1)
            for branch in model.multi_level_guidance.branches:
                branch[-1].weight.fill_(0.1)
        output = model(*self.make_inputs())
        output['mask_logits'].mean().backward()
        for branch in model.multi_level_guidance.branches:
            self.assertTrue(all(parameter.grad is not None for parameter in branch.parameters()))

    def test_all_four_clip_levels_have_37_by_37_grid(self):
        model = BridgeAdaptCLIPV21Fine(
            semantic_channels=8, fusion_channels=8, structural_channels=8,
            structural_input_size=32,
        )
        inputs = self.make_inputs()
        self.assertEqual(model.clip_feature_levels, (6, 12, 18, 24))
        for tokens in inputs[1]:
            self.assertEqual(model.multi_level_guidance.tokens_to_grid(tokens).shape[-2:], (37, 37))
        with self.assertRaises(ValueError):
            bad = [torch.randn(1, 100, 8) for _ in range(4)]
            model(inputs[0], bad, inputs[2], inputs[3])

    def test_image_score_is_external_and_unchanged(self):
        model = BridgeAdaptCLIPV21Fine(
            semantic_channels=8, fusion_channels=8, structural_channels=8,
            structural_input_size=32,
        )
        image_score = torch.tensor([0.731])
        before = image_score.clone()
        model(*self.make_inputs())
        self.assertTrue(torch.equal(image_score, before))
        self.assertFalse(any('image' in name for name, _ in model.named_parameters()))


if __name__ == '__main__': unittest.main()
