import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from tools.dacl10k_external import (
    DAMAGE_LABELS,
    StreamingBinaryHistogram,
    build_protocol_masks,
    hann_weight,
    rasterize_damage_labels,
    sliding_window_probability,
    sliding_window_outputs,
    tile_starts,
    valid_core_window_outputs,
)


class DACL10KExternalTests(unittest.TestCase):
    def test_tile_starts_force_last_boundary(self):
        self.assertEqual(tile_starts(800), [0])
        self.assertEqual(tile_starts(1280), [0, 256])
        self.assertEqual(tile_starts(4000), [0, 768, 1536, 2304, 2976])

    def test_hann_is_clamped_after_outer_product(self):
        weight = hann_weight(16)
        self.assertEqual(weight.shape, (16, 16))
        self.assertAlmostEqual(float(weight.min()), 1e-3, places=7)

    def test_polygon_raster_and_target_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation = {
                "size": {"height": 12, "width": 12},
                "objects": [
                    {"id": 1, "classTitle": "crack", "points": {
                        "exterior": [[1, 1], [8, 1], [8, 8], [1, 8]],
                        "interior": [[[3, 3], [5, 3], [5, 5], [3, 5]]],
                    }},
                    {"id": 2, "classTitle": "rust", "points": {
                        "exterior": [[6, 6], [10, 6], [10, 10], [6, 10]],
                        "interior": [],
                    }},
                    {"id": 3, "classTitle": "bearing", "points": {
                        "exterior": [[0, 9], [2, 9], [2, 11], [0, 11]],
                        "interior": [],
                    }},
                ],
            }
            path = Path(directory) / "sample.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")
            damage = rasterize_damage_labels(path)
            self.assertFalse(damage["crack"][4, 4])
            masks = build_protocol_masks(damage)
            self.assertTrue(masks["Crack"]["positive"][7, 7])
            self.assertFalse(masks["Crack"]["ignore"][7, 7])
            self.assertTrue(masks["Crack"]["ignore"][9, 9])
            self.assertFalse(masks["All-Damage"]["positive"][10, 1])
            self.assertFalse(masks["All-Damage"]["ignore"].any())

    def test_streaming_histogram_matches_exact_metrics(self):
        rng = np.random.default_rng(42)
        scores = rng.random(20000, dtype=np.float32)
        labels = rng.random(20000) < 0.12
        ignore = rng.random(20000) < 0.07
        histogram = StreamingBinaryHistogram(65536)
        for indices in np.array_split(np.arange(len(scores)), 7):
            histogram.update(scores[indices], labels[indices], ignore[indices])
        result = histogram.metrics()
        valid = ~ignore
        exact_ap = average_precision_score(labels[valid], scores[valid])
        exact_auc = roc_auc_score(labels[valid], scores[valid])
        precision, recall, _ = precision_recall_curve(labels[valid], scores[valid])
        exact_f1 = np.max(2 * precision * recall / np.maximum(precision + recall, 1e-15))
        self.assertLess(abs(result["P-AP"] - exact_ap), 1e-3)
        self.assertLess(abs(result["P-AUROC"] - exact_auc), 1e-4)
        self.assertLess(abs(result["P-F1max"] - exact_f1), 1e-3)

    def test_constant_tiles_stitch_without_seams_and_crop_padding(self):
        image = Image.fromarray(np.zeros((700, 1300, 3), dtype=np.uint8))

        def predict(tiles):
            return np.full((len(tiles), 1024, 1024), 0.37, dtype=np.float32)

        result = sliding_window_probability(image, predict)
        self.assertEqual(result.shape, (700, 1300))
        self.assertTrue(np.allclose(result, 0.37, atol=1e-6))

    def test_multi_output_stitching_and_geometry_partition(self):
        image = Image.fromarray(np.zeros((1024, 1800, 3), dtype=np.uint8))

        def predict(tiles):
            shape = (len(tiles), 1024, 1024)
            return {
                "probability": np.full(shape, 0.2, dtype=np.float32),
                "fine_correction": np.full(shape, -1.5, dtype=np.float32),
            }

        outputs, geometry = sliding_window_outputs(image, predict)
        self.assertTrue(np.allclose(outputs["probability"], 0.2, atol=1e-6))
        self.assertTrue(np.allclose(outputs["fine_correction"], -1.5, atol=1e-5))
        self.assertTrue(np.all(geometry["overlap"] ^ geometry["non_overlap"]))
        self.assertTrue(np.all(geometry["edge_dominated"] ^ geometry["center_dominated"]))

    def test_valid_core_constant_prediction_covers_original_shape(self):
        image = Image.fromarray(np.zeros((900, 1600, 3), dtype=np.uint8))

        def predict(tiles):
            shape = (len(tiles), 1024, 1024)
            return {"probability": np.full(shape, 0.42, dtype=np.float32)}

        outputs, metadata = valid_core_window_outputs(image, predict)
        self.assertEqual(outputs["probability"].shape, (900, 1600))
        self.assertTrue(np.allclose(outputs["probability"], 0.42, atol=1e-6))
        self.assertEqual(metadata["tile_count"], 6)
        self.assertEqual(metadata["core_size"], 768)

    def test_valid_core_discards_tile_edge_artifact(self):
        image = Image.fromarray(np.zeros((500, 700, 3), dtype=np.uint8))

        def predict(tiles):
            values = np.ones((len(tiles), 1024, 1024), dtype=np.float32)
            values[:, 128:896, 128:896] = 0.0
            return {"probability": values}

        outputs, metadata = valid_core_window_outputs(image, predict)
        self.assertEqual(metadata["tile_count"], 1)
        self.assertTrue(np.allclose(outputs["probability"], 0.0))


if __name__ == "__main__":
    unittest.main()
