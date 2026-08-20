import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'bridgeadaptclip_v12_losses_standalone',
    ROOT / 'tools/bridgeadaptclip_v12_losses.py',
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BridgeAdaptCLIPV12LossTests(unittest.TestCase):
    def test_soft_gate_target_and_bce_logits(self):
        target = torch.tensor([[[[0.0, 1.0]]]])
        base = torch.tensor([[[[0.2, 0.3]]]], requires_grad=True)
        logits = torch.tensor([[[[-1.0, 2.0]]]], requires_grad=True)
        correction = torch.tensor([[[[0.5, -0.5]]]], requires_grad=True)
        gate_target, gate_loss, _ = module.error_aware_gate_losses(
            logits, correction, target, base
        )
        self.assertTrue(torch.allclose(gate_target, torch.tensor([[[[0.2, 0.7]]]])))
        self.assertFalse(gate_target.requires_grad)
        expected = F.binary_cross_entropy_with_logits(logits, gate_target)
        self.assertTrue(torch.allclose(gate_loss, expected))

    def test_preservation_is_normalized_per_image(self):
        target = torch.tensor([
            [[[0.0, 0.0]]],
            [[[1.0, 1.0]]],
        ])
        base = torch.tensor([
            [[[0.0, 0.0]]],
            [[[0.5, 0.5]]],
        ])
        correction = torch.tensor([
            [[[2.0, 4.0]]],
            [[[10.0, 14.0]]],
        ])
        logits = torch.zeros_like(correction)
        _, _, preserve = module.error_aware_gate_losses(
            logits, correction, target, base
        )
        # Image means are 3 and 12; per-image average must be 7.5.
        self.assertAlmostEqual(float(preserve), 7.5)

    def test_native_amp_inputs_reduce_in_float32(self):
        logits = torch.zeros(1, 1, 1024, 1024, dtype=torch.float16)
        correction = torch.ones_like(logits)
        target = torch.ones_like(logits)
        base = torch.zeros_like(logits)
        _, gate_loss, preserve = module.error_aware_gate_losses(
            logits, correction, target, base
        )
        self.assertEqual(gate_loss.dtype, torch.float32)
        self.assertEqual(preserve.dtype, torch.float32)
        self.assertTrue(torch.isfinite(gate_loss))
        self.assertTrue(torch.isfinite(preserve))


if __name__ == '__main__':
    unittest.main()
