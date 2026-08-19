import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'bridge_masks.py'
SPEC = importlib.util.spec_from_file_location('bridge_masks_standalone', MODULE_PATH)
BRIDGE_MASKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE_MASKS)
decode_bridge_class_masks = BRIDGE_MASKS.decode_bridge_class_masks
load_bridge_native_binary_masks = BRIDGE_MASKS.load_bridge_native_binary_masks


class BridgeNativeMetricMaskTests(unittest.TestCase):
    def test_native_mask_uses_frozen_palette_without_resizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotated = Path(tmp) / 'annotated'
            annotated.mkdir()
            image_path = annotated / 'codebrim_example.jpg'
            mask = np.zeros((4, 4, 3), dtype=np.uint8)
            mask[0, 0] = (255, 0, 0)
            mask[1, 1] = (0, 255, 0)
            mask[2, 2] = (0, 0, 255)
            mask[3, 3] = (255, 255, 0)
            Image.fromarray(mask).save(image_path.with_suffix('.png'))

            class_masks, any_defect = decode_bridge_class_masks(image_path)
            self.assertEqual(set(class_masks), {'Crack', 'Spalling', 'Corrosion', 'Efflorescence'})
            self.assertEqual(int(any_defect.sum()), 4)

            batch = load_bridge_native_binary_masks([image_path], 4)
            self.assertEqual(tuple(batch.shape), (1, 4, 4))
            self.assertEqual(int(batch.sum()), 4)

    def test_native_mask_rejects_resolution_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotated = Path(tmp) / 'annotated'
            annotated.mkdir()
            image_path = annotated / 's2ds_example.jpg'
            mask = np.zeros((4, 4, 3), dtype=np.uint8)
            mask[0, 0] = (255, 0, 0)
            Image.fromarray(mask).save(image_path.with_suffix('.png'))

            with self.assertRaisesRegex(ValueError, 'GT resizing is forbidden'):
                load_bridge_native_binary_masks([image_path], 8)

    def test_native_mask_rejects_unknown_non_background_color(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotated = Path(tmp) / 'annotated'
            annotated.mkdir()
            image_path = annotated / 's2ds_example.jpg'
            mask = np.zeros((4, 4, 3), dtype=np.uint8)
            mask[0, 0] = (255, 255, 255)
            Image.fromarray(mask).save(image_path.with_suffix('.png'))

            with self.assertRaisesRegex(ValueError, 'outside the frozen Bridge2893 palette'):
                decode_bridge_class_masks(image_path)

    def test_normal_query_uses_native_zero_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / 'normal' / 'normal_example.jpg'
            image_path.parent.mkdir()
            batch = load_bridge_native_binary_masks([image_path], 4)
            self.assertEqual(tuple(batch.shape), (1, 4, 4))
            self.assertEqual(int(batch.sum()), 0)


if __name__ == '__main__':
    unittest.main()
