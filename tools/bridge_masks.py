"""Strict frozen-palette decoding for native Bridge2893 raster masks."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFECT_COLORS_BY_SOURCE = {
    'CODEBRIM': {
        'Crack': (255, 0, 0),
        'Spalling': (0, 255, 0),
        'Corrosion': (0, 0, 255),
        'Efflorescence': (255, 255, 0),
    },
    'S2DS': {
        'Crack': (255, 0, 0),
        'Spalling': (0, 255, 0),
        'Corrosion': (0, 0, 255),
        'Efflorescence': (255, 255, 0),
    },
}
DEFECT_NAMES = tuple(DEFECT_COLORS_BY_SOURCE['CODEBRIM'])


def decode_bridge_class_masks(image_path):
    """Decode the frozen Bridge2893 RGB raster mask without resizing it."""
    image_path = Path(str(image_path))
    mask_path = image_path.with_suffix('.png')
    if not mask_path.is_file():
        raise FileNotFoundError(f'Missing Bridge2893 raster mask: {mask_path}')

    rgb_mask = np.asarray(Image.open(mask_path).convert('RGB'))
    source = 'CODEBRIM' if image_path.name.startswith('codebrim_') else 'S2DS'
    class_masks = {
        name: np.all(rgb_mask == np.asarray(color, dtype=np.uint8), axis=-1)
        for name, color in DEFECT_COLORS_BY_SOURCE[source].items()
    }
    any_defect = np.logical_or.reduce(list(class_masks.values()))
    unknown_foreground = np.any(rgb_mask != 0, axis=-1) & ~any_defect
    if unknown_foreground.any():
        raise ValueError(
            f'{mask_path} contains {int(unknown_foreground.sum())} non-background '
            'pixels outside the frozen Bridge2893 palette.'
        )
    return class_masks, any_defect


def load_bridge_native_binary_masks(image_paths, metric_resolution):
    """Load original GT for a query batch, never resizing a defect mask."""
    expected_shape = (metric_resolution, metric_resolution)
    masks = []
    for image_path in image_paths:
        image_path = Path(str(image_path))
        if image_path.parent.name == 'normal':
            masks.append(torch.zeros(expected_shape, dtype=torch.int32))
            continue

        _, any_defect = decode_bridge_class_masks(image_path)
        if any_defect.shape != expected_shape:
            raise ValueError(
                f'Native GT for {image_path} has shape {any_defect.shape}; '
                f'expected exactly {expected_shape}. GT resizing is forbidden.'
            )
        masks.append(torch.from_numpy(any_defect.copy()).to(torch.int32))
    return torch.stack(masks)
