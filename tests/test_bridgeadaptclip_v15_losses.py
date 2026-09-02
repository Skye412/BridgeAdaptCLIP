import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'bridgeadaptclip_v15_losses_standalone',
    ROOT / 'tools' / 'bridgeadaptclip_v15_losses.py',
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HardPixelRankingLossTests(unittest.TestCase):
    def test_prefers_positive_logits_above_negative_logits(self):
        target = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
        correctly_ranked = torch.tensor([[[[3.0, 2.0, -2.0, -3.0]]]])
        incorrectly_ranked = -correctly_ranked
        correct_loss, _, _, _ = MODULE.hard_pixel_ranking_loss(
            correctly_ranked, target, 2, 2
        )
        wrong_loss, _, _, _ = MODULE.hard_pixel_ranking_loss(
            incorrectly_ranked, target, 2, 2
        )
        self.assertLess(float(correct_loss), float(wrong_loss))

    def test_selects_lowest_positives_and_highest_negatives(self):
        logits = torch.tensor([[[[4.0, 1.0, 3.0, -2.0, 2.0, 0.0]]]])
        target = torch.tensor([[[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]]])
        _, positive_mean, negative_mean, valid_images = MODULE.hard_pixel_ranking_loss(
            logits, target, hard_positive_count=2, hard_negative_count=2
        )
        self.assertEqual(valid_images, 1)
        self.assertAlmostEqual(float(positive_mean), 2.0)
        self.assertAlmostEqual(float(negative_mean), 1.0)

    def test_uses_all_positive_pixels_when_support_is_small(self):
        logits = torch.tensor([[[[2.0, 0.0, -1.0]]]])
        target = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        _, positive_mean, _, _ = MODULE.hard_pixel_ranking_loss(
            logits, target, 256, 2
        )
        self.assertAlmostEqual(float(positive_mean), 2.0)

    def test_normal_images_do_not_contribute(self):
        logits = torch.randn(2, 1, 2, 2, requires_grad=True)
        target = torch.zeros_like(logits)
        loss, _, _, valid_images = MODULE.hard_pixel_ranking_loss(logits, target)
        self.assertEqual(valid_images, 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_gradients_raise_positives_and_lower_negatives(self):
        logits = torch.zeros(1, 1, 1, 4, requires_grad=True)
        target = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
        loss, _, _, _ = MODULE.hard_pixel_ranking_loss(logits, target, 2, 2)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[0, 0, 0, :2] < 0))
        self.assertTrue(torch.all(logits.grad[0, 0, 0, 2:] > 0))


if __name__ == '__main__':
    unittest.main()
