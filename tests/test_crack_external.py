import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.crack_external import (
    CrackMorphologyMetrics,
    load_crack_mask,
    prepare_geometry_canvas,
    restore_geometry_map,
)


class CrackExternalTests(unittest.TestCase):
    def test_top_left_and_symmetric_padding_preserve_native_content(self):
        source = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        image = Image.fromarray(source)
        top_left, meta_a = prepare_geometry_canvas(
            image, "current_top_left_pad", canvas_size=10
        )
        symmetric, meta_b = prepare_geometry_canvas(
            image, "symmetric_pad_native_scale", canvas_size=10
        )
        self.assertEqual(meta_a["content_left"], 0)
        self.assertEqual(meta_a["content_top"], 0)
        self.assertEqual(meta_b["content_left"], 2)
        self.assertEqual(meta_b["content_top"], 3)
        self.assertTrue(np.array_equal(np.asarray(top_left)[:4, :6], source))
        self.assertTrue(np.array_equal(np.asarray(symmetric)[3:7, 2:8], source))
        restored = restore_geometry_map(np.asarray(symmetric)[..., 0], meta_b)
        self.assertTrue(np.array_equal(restored, source[..., 0].astype(np.float32)))

    def test_fit_long_side_and_inverse_mapping(self):
        image = Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8))
        _, metadata = prepare_geometry_canvas(
            image, "fit_long_side_1024", canvas_size=16
        )
        self.assertEqual((metadata["content_width"], metadata["content_height"]), (16, 8))
        self.assertEqual(metadata["content_top"], 4)
        self.assertAlmostEqual(metadata["valid_content_fraction"], 0.5)
        canvas_map = np.full((16, 16), 0.25, dtype=np.float32)
        restored = restore_geometry_map(canvas_map, metadata)
        self.assertEqual(restored.shape, (4, 8))
        self.assertTrue(np.allclose(restored, 0.25))

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
