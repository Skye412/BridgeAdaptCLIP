import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.crack_external import CrackMorphologyMetrics, load_crack_mask


class CrackExternalTests(unittest.TestCase):
    def test_dataset_specific_mask_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.asarray([[0, 1, 127, 128, 255]], dtype=np.uint8)).save(path)
            cam = load_crack_mask({"mask_path": str(path), "mask_rule": "uint8>=128"})
            binary = load_crack_mask({"mask_path": str(path), "mask_rule": "binary>0"})
            self.assertEqual(cam.tolist(), [[False, False, False, True, True]])
            self.assertEqual(binary.tolist(), [[False, True, True, True, True]])

    def test_perfect_prediction_has_perfect_morphology(self):
        target = np.zeros((64, 64), dtype=bool)
        target[10:54, 30:34] = True
        probability = target.astype(np.float32)
        metric = CrackMorphologyMetrics(threshold=0.5, tolerance=3, min_component_pixels=10)
        metric.update(probability, target)
        report = metric.report()
        for name in ("Boundary-F1", "clDice", "Skeleton-Recall", "Connected-Component-Recall"):
            self.assertAlmostEqual(report[name], 1.0, places=7)

    def test_tolerance_recovers_small_offset(self):
        target = np.zeros((64, 64), dtype=bool)
        target[8:56, 30:32] = True
        probability = np.zeros((64, 64), dtype=np.float32)
        probability[8:56, 32:34] = 1.0
        metric = CrackMorphologyMetrics(threshold=0.5, tolerance=3, min_component_pixels=10)
        metric.update(probability, target)
        report = metric.report()
        self.assertGreater(report["Boundary-F1"], 0.9)
        self.assertEqual(report["Skeleton-Recall"], 1.0)
        self.assertEqual(report["Connected-Component-Recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
