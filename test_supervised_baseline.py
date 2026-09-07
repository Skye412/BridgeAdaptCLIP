"""Formal native-1024 evaluation for supervised paper comparisons."""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from adaptcliplib.supervised_baselines import build_supervised_baseline
from dataset import BridgeSupervisedDataset
from tools.bridge_class_metrics import evaluate_bridge_classes
from tools.bridge_masks import decode_bridge_class_masks
from tools.crack_external import CrackMorphologyMetrics
from tools.supervised_protocol import (
    BinaryProtocolMetrics, f1_from_counts, fixed_threshold_counts,
)


@torch.no_grad()
def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = args.amp and device.type == 'cuda'
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_name = checkpoint['model_name']
    model = build_supervised_baseline(model_name, pretrained=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    data = BridgeSupervisedDataset(args.test_data_path, training=False)
    loader = DataLoader(
        data, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    metrics = BinaryProtocolMetrics(args.pixel_thresholds)
    fixed_counts = {'tp': 0, 'fp': 0, 'fn': 0}
    crack_morphology = CrackMorphologyMetrics(threshold=args.val_threshold)
    predictions, masks, paths, anomalies = [], [], [], []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    evaluation_started = time.perf_counter()
    model_forward_seconds = 0.0
    for batch in tqdm(loader, desc=f'test {model_name}'):
        image = batch['img'].to(device, non_blocking=True)
        target = batch['native_mask'].to(device, non_blocking=True).unsqueeze(1)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        forward_started = time.perf_counter()
        with torch.amp.autocast('cuda', enabled=amp):
            probability = torch.sigmoid(model(image).float())
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        model_forward_seconds += time.perf_counter() - forward_started
        metrics.update(probability, target, batch['anomaly'])
        counts = fixed_threshold_counts(probability, target, args.val_threshold)
        for key in fixed_counts:
            fixed_counts[key] += counts[key]
        predictions.append(probability[:, 0].cpu())
        masks.append(target[:, 0].cpu())
        paths.extend(batch['img_path'])
        anomalies.extend(batch['anomaly'].cpu().tolist())
        image_path = str(batch['img_path'][0])
        probability_array = probability[0, 0].cpu().numpy()
        if int(batch['anomaly'][0]) == 0:
            crack_morphology.update(
                probability_array, np.zeros_like(probability_array, dtype=bool)
            )
        else:
            class_masks, any_defect = decode_bridge_class_masks(image_path)
            crack_mask = class_masks['Crack']
            if crack_mask.any():
                # Match diagnostic AP: other annotated defect pixels are ignored.
                valid_probability = probability_array.copy()
                valid_probability[any_defect & ~crack_mask] = 0.0
                crack_morphology.update(valid_probability, crack_mask)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    result = metrics.compute()
    result['P-F1@val-threshold'] = f1_from_counts(fixed_counts)
    result['val_threshold'] = args.val_threshold
    pixel_scores = torch.cat(predictions)
    image_scores = pixel_scores.flatten(1).max(dim=1).values
    per_defect = evaluate_bridge_classes(
        np.asarray(paths), image_scores, pixel_scores, args.pixel_thresholds,
        os.path.join(args.output_dir, 'bridge_defect_metrics.json'),
    )
    result['Macro-diagnostic-P-AP'] = float(np.mean([
        value['metrics_percent']['P-AP'] for value in per_defect.values()
    ]))
    crack_structure = {
        key: (100.0 * value if isinstance(value, float) else value)
        for key, value in crack_morphology.report().items()
    }
    report = {
        'protocol': {
            'task': 'Bridge2893 four-defect-union binary segmentation',
            'input_resolution': 1024, 'metric_resolution': 1024,
            'gt': 'original frozen raster', 'prediction': 'continuous sigmoid',
            'checkpoint': args.checkpoint,
            'checkpoint_selection': 'validation overall P-AP',
        },
        'model': model_name,
        'results_percent': result,
        'per_defect': per_defect,
        'crack_structure_percent': crack_structure,
        'efficiency': {
            'images': len(data),
            'model_forward_seconds': model_forward_seconds,
            'model_images_per_second': len(data) / max(model_forward_seconds, 1e-12),
            'evaluation_wall_seconds': evaluation_seconds,
            'peak_cuda_memory_bytes': (
                torch.cuda.max_memory_allocated(device)
                if device.type == 'cuda' else 0
            ),
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'trainable_parameters': sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
            'batch_size': 1,
        },
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as stream:
        json.dump(report, stream, indent=2)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--test_data_path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--val_threshold', type=float, required=True)
    p.add_argument('--pixel_thresholds', type=int, default=2048)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    return p


if __name__ == '__main__':
    evaluate(parser().parse_args())
