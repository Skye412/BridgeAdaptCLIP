import unittest
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = load_module('bridgeadaptclip_v20_standalone', 'adaptcliplib/bridgeadaptclip.py')
LOSSES = load_module('bridgeadaptclip_v20_losses_standalone', 'tools/bridgeadaptclip_v20_losses.py')
BridgeAdaptCLIPV20 = MODEL.BridgeAdaptCLIPV20
broad_gate_and_positive_preservation_losses = LOSSES.broad_gate_and_positive_preservation_losses
negative_only_broad_ranking_loss = LOSSES.negative_only_broad_ranking_loss


class BridgeAdaptCLIPV20Tests(unittest.TestCase):
    def test_identity_initialization_and_non_positive_correction(self):
        model = BridgeAdaptCLIPV20(joint_channels=8, broad_channels=8, output_size=32)
        joint = torch.randn(2, 8, 8, 8)
        fine = torch.randn(2, 1, 32, 32)
        row0 = torch.rand(2, 1, 32, 32)
        output = model(joint, fine, row0)
        self.assertEqual(output['broad_feature'].shape[-2:], (4, 4))
        self.assertTrue(torch.all(output['broad_correction'] <= 0))
        expected = -torch.sigmoid(torch.tensor(-4.0)) * torch.nn.functional.softplus(
            torch.tensor(-4.0)
        )
        self.assertTrue(torch.allclose(output['broad_correction'], expected.expand_as(fine)))
        self.assertTrue(torch.allclose(output['mask_logits'], fine + expected))

    def test_frozen_fine_inputs_receive_no_gradient(self):
        model = BridgeAdaptCLIPV20(joint_channels=8, broad_channels=8, output_size=32)
        joint = torch.randn(1, 8, 8, 8, requires_grad=True)
        fine = torch.randn(1, 1, 32, 32, requires_grad=True)
        row0 = torch.rand(1, 1, 32, 32, requires_grad=True)
        model(joint, fine, row0)['mask_logits'].sum().backward()
        self.assertIsNone(joint.grad)
        self.assertIsNone(fine.grad)
        self.assertIsNone(row0.grad)
        self.assertIsNotNone(model.magnitude_head.bias.grad)

    def test_fp_gate_target_is_zero_on_positive_pixels(self):
        target = torch.tensor([[[[1.0, 0.0]]]])
        fine_probability = torch.tensor([[[[0.9, 0.8]]]])
        gate_logits = torch.zeros_like(target, requires_grad=True)
        correction = torch.full_like(target, -1.0, requires_grad=True)
        fp_target, gate_loss, preserve = broad_gate_and_positive_preservation_losses(
            gate_logits, correction, target, fine_probability
        )
        self.assertTrue(torch.equal(fp_target, torch.tensor([[[[0.0, 0.8]]]])))
        self.assertAlmostEqual(float(preserve.detach()), 1.0)
        (gate_loss + preserve).backward()
        self.assertIsNotNone(gate_logits.grad)
        self.assertIsNotNone(correction.grad)

    def test_broad_ranking_routes_gradient_only_to_final_negatives(self):
        final = torch.tensor([[[[-2.0, 2.0, 1.0, 0.0]]]], requires_grad=True)
        fine = torch.tensor([[[[-2.0, -1.0, 1.0, 0.0]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
        loss = negative_only_broad_ranking_loss(final, fine, target, 2, 2)[0]
        loss.backward()
        self.assertTrue(torch.equal(final.grad[..., :2], torch.zeros_like(final.grad[..., :2])))
        self.assertTrue(torch.all(final.grad[..., 2:] > 0))
        self.assertIsNone(fine.grad)


if __name__ == '__main__':
    unittest.main()
