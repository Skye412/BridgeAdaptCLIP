import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'bridge_residual_diagnostics_standalone',
    ROOT / 'tools/bridge_residual_diagnostics.py',
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResidualDiagnosticTests(unittest.TestCase):
    def test_region_means(self):
        accumulator = module.RegionAccumulator()
        gate = torch.tensor([[[[0.2, 0.8]]]])
        residual = torch.tensor([[[[-2.0, 4.0]]]])
        correction = gate * residual
        error = torch.tensor([[[[0.1, 0.9]]]])
        mask = torch.ones_like(gate, dtype=torch.bool)
        accumulator.update('all', mask, gate, residual, correction, error)
        report = accumulator.finalize()['all']
        self.assertEqual(report['pixel_count'], 2)
        self.assertAlmostEqual(report['mean_gate'], 0.5)
        self.assertAlmostEqual(report['mean_correction'], 1.4)
        self.assertAlmostEqual(report['mean_abs_correction'], 1.8)
        self.assertAlmostEqual(report['mean_positive_correction'], 1.6)
        self.assertAlmostEqual(report['mean_negative_correction'], -0.2)

    def test_pearson_streaming_matches_known_values(self):
        accumulator = module.PearsonAccumulator()
        accumulator.update(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))
        accumulator.update(torch.tensor([3.0, 4.0]), torch.tensor([6.0, 8.0]))
        self.assertAlmostEqual(accumulator.finalize(), 1.0)

    def test_pearson_rejects_shape_mismatch(self):
        accumulator = module.PearsonAccumulator()
        with self.assertRaisesRegex(ValueError, 'same number'):
            accumulator.update(torch.ones(2), torch.ones(3))


if __name__ == '__main__':
    unittest.main()
