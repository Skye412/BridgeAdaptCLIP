import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'bridgeadaptclip_v16_losses_standalone',
    ROOT / 'tools' / 'bridgeadaptclip_v16_losses.py',
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SkeletonBalancedRankingLossTests(unittest.TestCase):
    def test_uses_separate_global_and_skeleton_pools(self):
        logits = torch.tensor([[[[-5., 1., 2., 3., 4., 0., -1., -2.]]]])
        target = torch.tensor([[[[1., 1., 1., 1., 1., 0., 0., 0.]]]])
        skeleton = torch.tensor([[[[0., 0., 0., 1., 1., 0., 0., 0.]]]])
        result = MODULE.skeleton_balanced_hard_pixel_ranking_loss(
            logits, target, skeleton, 2, 2, 2
        )
        _, _, _, global_mean, thin_mean, negative_mean, skeleton_count, valid = result
        self.assertEqual(valid, 1)
        self.assertAlmostEqual(float(global_mean), -2.0)
        self.assertAlmostEqual(float(thin_mean), 3.5)
        self.assertAlmostEqual(float(negative_mean), -0.5)
        self.assertAlmostEqual(float(skeleton_count), 2.0)

    def test_supplements_small_skeleton_from_positive_pool(self):
        logits = torch.tensor([[[[-4., -3., 2., 5., 1., 0.]]]])
        target = torch.tensor([[[[1., 1., 1., 1., 0., 0.]]]])
        skeleton = torch.tensor([[[[0., 0., 0., 1., 0., 0.]]]])
        result = MODULE.skeleton_balanced_hard_pixel_ranking_loss(
            logits, target, skeleton, 2, 3, 2
        )
        thin_mean = result[4]
        skeleton_count = result[6]
        self.assertAlmostEqual(float(skeleton_count), 1.0)
        self.assertAlmostEqual(float(thin_mean), (-4.0 - 3.0 + 5.0) / 3.0)

    def test_balanced_formula_is_mean_of_both_terms(self):
        logits = torch.tensor([[[[0., 2., -1., 1.]]]], requires_grad=True)
        target = torch.tensor([[[[1., 1., 0., 0.]]]])
        skeleton = torch.tensor([[[[0., 1., 0., 0.]]]])
        total, global_loss, thin_loss, *_ = (
            MODULE.skeleton_balanced_hard_pixel_ranking_loss(
                logits, target, skeleton, 1, 1, 1
            )
        )
        self.assertTrue(torch.allclose(total, 0.5 * (global_loss + thin_loss)))
        total.backward()
        self.assertIsNotNone(logits.grad)

    def test_normal_image_returns_differentiable_zero(self):
        logits = torch.randn(1, 1, 2, 2, requires_grad=True)
        target = torch.zeros_like(logits)
        skeleton = torch.zeros_like(logits)
        result = MODULE.skeleton_balanced_hard_pixel_ranking_loss(
            logits, target, skeleton
        )
        self.assertEqual(result[-1], 0)
        result[0].backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


if __name__ == '__main__':
    unittest.main()
