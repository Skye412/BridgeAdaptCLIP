import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'bridgeadaptclip_v14_losses_standalone',
    ROOT / 'tools' / 'bridgeadaptclip_v14_losses.py',
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BridgeAdaptCLIPV14LossTests(unittest.TestCase):
    def test_matches_locked_final_logit_formula(self):
        logits = torch.tensor([[[[2.0, -2.0]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0]]]])
        error = torch.tensor([[[[0.8, 0.4]]]])
        total, fn_loss, fp_loss = MODULE.final_logit_margin_loss(
            logits, target, error, margin=1.0
        )
        expected_fn = F.softplus(torch.tensor(-1.0))
        expected_fp = F.softplus(torch.tensor(-1.0))
        self.assertTrue(torch.allclose(fn_loss, expected_fn))
        self.assertTrue(torch.allclose(fp_loss, expected_fp))
        self.assertTrue(torch.allclose(total, 0.5 * (expected_fn + expected_fp)))

    def test_correct_final_logit_margins_reduce_loss(self):
        target = torch.tensor([[[[1.0, 0.0]]]])
        error = torch.ones_like(target)
        correct = torch.tensor([[[[2.0, -2.0]]]], requires_grad=True)
        wrong = torch.tensor([[[[-2.0, 2.0]]]], requires_grad=True)
        correct_loss, _, _ = MODULE.final_logit_margin_loss(correct, target, error)
        wrong_loss, _, _ = MODULE.final_logit_margin_loss(wrong, target, error)
        self.assertLess(float(correct_loss.detach()), float(wrong_loss.detach()))

    def test_fp32_reduction_and_gradient(self):
        logits = torch.zeros(1, 1, 1024, 1024, dtype=torch.float16, requires_grad=True)
        target = torch.zeros_like(logits)
        target[..., :512, :] = 1
        error = torch.ones_like(logits)
        loss, fn_loss, fp_loss = MODULE.final_logit_margin_loss(
            logits, target, error, margin=1.0
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(fn_loss))
        self.assertTrue(torch.isfinite(fp_loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == '__main__':
    unittest.main()
