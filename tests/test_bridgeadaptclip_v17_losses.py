import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'tools' / f'{name}.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V15 = load_module('bridgeadaptclip_v15_losses')
V17 = load_module('bridgeadaptclip_v17_losses')


class PositivePrioritizedRankingLossTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])

    def test_forward_value_matches_original_pairwise_ranking(self):
        logits = torch.tensor([[[[-2.0, 1.0, 3.0, -1.0]]]])
        original = V15.hard_pixel_ranking_loss(logits, self.target, 2, 2)[0]
        prioritized = V17.positive_prioritized_ranking_loss(
            logits, self.target, 2, 2, raise_positive_weight=0.8
        )[0]
        self.assertTrue(torch.allclose(original, prioritized, atol=1e-6))

    def test_weight_one_routes_gradient_only_to_positives(self):
        logits = torch.zeros(1, 1, 1, 4, requires_grad=True)
        loss = V17.positive_prioritized_ranking_loss(
            logits, self.target, 2, 2, raise_positive_weight=1.0
        )[0]
        loss.backward()
        self.assertTrue(torch.all(logits.grad[..., :2] < 0))
        self.assertTrue(torch.equal(logits.grad[..., 2:], torch.zeros_like(logits.grad[..., 2:])))

    def test_weight_zero_routes_gradient_only_to_negatives(self):
        logits = torch.zeros(1, 1, 1, 4, requires_grad=True)
        loss = V17.positive_prioritized_ranking_loss(
            logits, self.target, 2, 2, raise_positive_weight=0.0
        )[0]
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[..., :2], torch.zeros_like(logits.grad[..., :2])))
        self.assertTrue(torch.all(logits.grad[..., 2:] > 0))

    def test_weight_point_eight_produces_four_to_one_gradient_mass(self):
        logits = torch.zeros(1, 1, 1, 4, requires_grad=True)
        loss = V17.positive_prioritized_ranking_loss(
            logits, self.target, 2, 2, raise_positive_weight=0.8
        )[0]
        loss.backward()
        positive_mass = logits.grad[..., :2].abs().sum()
        negative_mass = logits.grad[..., 2:].abs().sum()
        self.assertAlmostEqual(float(positive_mass / negative_mass), 4.0, places=5)


if __name__ == '__main__':
    unittest.main()
