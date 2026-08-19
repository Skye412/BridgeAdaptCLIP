"""Per-defect evaluation for the color-coded Bridge2893 masks."""

import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .effecient_metric import Evaluator


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


def _resize_mask(mask, shape):
    height, width = shape
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
    image = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return torch.from_numpy(np.asarray(image).copy() > 0)


def _update_histograms(positive_hist, negative_hist, predictions, targets, num_thresholds):
    predictions = predictions.detach().cpu().float().clamp(0, 1)
    targets = targets.detach().cpu().bool()
    bins = torch.clamp(
        (predictions * (num_thresholds - 1)).floor().long(),
        min=0,
        max=num_thresholds - 1,
    )
    positive_hist += torch.bincount(bins[targets], minlength=num_thresholds).to(torch.float64)
    negative_hist += torch.bincount(bins[~targets], minlength=num_thresholds).to(torch.float64)


def evaluate_bridge_classes(query_paths, image_scores, pixel_scores, num_thresholds, output_path):
    """Report each defect against normal images while ignoring other defect pixels."""
    states = {
        name: {
            'positive_hist': torch.zeros(num_thresholds, dtype=torch.float64),
            'negative_hist': torch.zeros(num_thresholds, dtype=torch.float64),
            'image_targets': [],
            'image_scores': [],
            'positive_images': 0,
            'normal_images': 0,
            'ignored_pixels': 0,
        }
        for name in DEFECT_NAMES
    }

    for image_path, image_score, pixel_score in zip(query_paths, image_scores, pixel_scores):
        image_path = str(image_path)
        pixel_score = pixel_score.detach().cpu()
        is_normal = os.path.basename(os.path.dirname(image_path)) == 'normal'

        if is_normal:
            normal_target = torch.zeros_like(pixel_score, dtype=torch.bool)
            for state in states.values():
                state['image_targets'].append(0)
                state['image_scores'].append(float(image_score))
                state['normal_images'] += 1
                _update_histograms(
                    state['positive_hist'], state['negative_hist'],
                    pixel_score, normal_target, num_thresholds,
                )
            continue

        mask_path = os.path.splitext(image_path)[0] + '.png'
        rgb_mask = np.asarray(Image.open(mask_path).convert('RGB'))
        source = 'CODEBRIM' if os.path.basename(image_path).startswith('codebrim_') else 'S2DS'
        defect_colors = DEFECT_COLORS_BY_SOURCE[source]
        class_masks = {
            name: np.all(rgb_mask == np.asarray(color, dtype=np.uint8), axis=-1)
            for name, color in defect_colors.items()
        }
        any_defect = np.logical_or.reduce(list(class_masks.values()))

        for name, class_mask in class_masks.items():
            if not class_mask.any():
                continue
            class_target = _resize_mask(class_mask, pixel_score.shape)
            other_defect = _resize_mask(any_defect & ~class_mask, pixel_score.shape)
            valid = ~other_defect
            state = states[name]
            state['image_targets'].append(1)
            state['image_scores'].append(float(image_score))
            state['positive_images'] += 1
            state['ignored_pixels'] += int(other_defect.sum())
            _update_histograms(
                state['positive_hist'], state['negative_hist'],
                pixel_score[valid], class_target[valid], num_thresholds,
            )

    report = {}
    for name, state in states.items():
        image_targets = np.asarray(state['image_targets'], dtype=np.uint8)
        scores = np.asarray(state['image_scores'], dtype=np.float64)
        precision, recall, _ = precision_recall_curve(image_targets, scores)
        image_f1 = np.max(2 * precision * recall / (precision + recall + 1e-12))
        pixel_metrics = Evaluator._metrics_from_histograms(
            state['positive_hist'], state['negative_hist']
        )
        report[name] = {
            'support': {
                'positive_images': state['positive_images'],
                'normal_images': state['normal_images'],
                'positive_pixels_evaluated': int(state['positive_hist'].sum()),
                'negative_pixels_evaluated': int(state['negative_hist'].sum()),
                'other_defect_pixels_ignored': state['ignored_pixels'],
            },
            'metrics_percent': {
                'I-AUROC': 100 * roc_auc_score(image_targets, scores),
                'I-AP': 100 * average_precision_score(image_targets, scores),
                'I-F1max': 100 * image_f1,
                'P-AUROC': 100 * pixel_metrics['auroc'],
                'P-AP': 100 * pixel_metrics['ap'],
                'P-F1max': 100 * pixel_metrics['f1max'],
            },
        }

    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False)
    return report
