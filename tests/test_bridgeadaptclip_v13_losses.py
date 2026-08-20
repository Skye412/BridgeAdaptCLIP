import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'bridgeadaptclip_v13_losses_standalone',
    ROOT / 'tools' / 'bridgeadaptclip_v13_losses.py',
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BridgeAdaptCLIPV13LossTests(unittest.TestCase):
    def test_correct_signs_have_lower_loss(self):
        target = torch.tensor([[[[1.0, 0.0]]]])
        error = torch.ones_like(target)
        correct = torch.tensor([[[[2.0, -2.0]]]], requires_grad=True)
        wrong = torch.tensor([[[[-2.0, 2.0]]]], requires_grad=True)
        correct_loss, _, _ = MODULE.signed_error_correction_loss(
            correct, target, error
        )
        wrong_loss, _, _ = MODULE.signed_error_correction_loss(
            wrong, target, error
        )
        self.assertLess(float(correct_loss.detach()), float(wrong_loss.detach()))

    def test_positive_and_negative_directions_are_balanced(self):
        target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        error = torch.ones_like(target)
        correction = torch.zeros_like(target, requires_grad=True)
        total, positive, negative = MODULE.signed_error_correction_loss(
            correction, target, error
        )
        self.assertAlmostEqual(
            float(positive.detach()), float(negative.detach()), places=6
        )
        self.assertAlmostEqual(
            float(total.detach()), float(positive.detach()), places=6
        )

    def test_normal_only_batch_is_finite_and_differentiable(self):
        target = torch.zeros(2, 1, 4, 4)
        error = torch.full_like(target, 0.5)
        correction = torch.zeros_like(target, dtype=torch.float16, requires_grad=True)
        total, positive, negative = MODULE.signed_error_correction_loss(
            correction, target, error
        )
        self.assertEqual(total.dtype, torch.float32)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(float(positive.detach()), 0.0)
        total.backward()
        self.assertIsNotNone(correction.grad)


if __name__ == '__main__':
    unittest.main()
