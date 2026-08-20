import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'bridge_diagnostics_standalone', ROOT / 'tools' / 'bridge_diagnostics.py'
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
bridge_source_from_path = MODULE.bridge_source_from_path
fuse_probabilities = MODULE.fuse_probabilities
fusion_grid = MODULE.fusion_grid
select_validation_fusion = MODULE.select_validation_fusion


class BridgeDiagnosticTests(unittest.TestCase):
    def test_linear_and_probability_or_fusion(self):
        row0 = torch.tensor([0.2, 0.8])
        structural = torch.tensor([0.5, 0.25])
        linear = fuse_probabilities(row0, structural, 'linear', 0.4)
        probability_or = fuse_probabilities(row0, structural, 'probability_or', 0.4)
        self.assertTrue(torch.allclose(linear, torch.tensor([0.32, 0.58])))
        self.assertTrue(torch.allclose(probability_or, torch.tensor([0.36, 0.82])))
        self.assertTrue(torch.all(probability_or >= row0))

    def test_zero_weight_preserves_row0_for_both_forms(self):
        row0 = torch.rand(3, 5)
        structural = torch.rand(3, 5)
        for form in ('linear', 'probability_or'):
            self.assertTrue(torch.equal(
                fuse_probabilities(row0, structural, form, 0.0), row0
            ))

    def test_selection_uses_validation_metrics_and_is_deterministic(self):
        candidates = [
            {'form': 'linear', 'weight': 0.1, 'metrics_percent': {'P-AP': 70, 'P-AUROC': 90}},
            {'form': 'linear', 'weight': 0.2, 'metrics_percent': {'P-AP': 71, 'P-AUROC': 89}},
            {'form': 'probability_or', 'weight': 0.2, 'metrics_percent': {'P-AP': 71, 'P-AUROC': 91}},
        ]
        selected = select_validation_fusion(candidates)
        self.assertEqual(selected['form'], 'probability_or')
        self.assertEqual(selected['weight'], 0.2)

    def test_grid_and_source_inference(self):
        self.assertEqual(len(fusion_grid()), 22)
        self.assertEqual(bridge_source_from_path('/x/codebrim_a.jpg'), 'CODEBRIM')
        self.assertEqual(bridge_source_from_path('/x/s2ds_a.jpg'), 'S2DS')


if __name__ == '__main__':
    unittest.main()
