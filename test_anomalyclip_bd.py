"""Test validation-selected AnomalyCLIP-BD on native Bridge2893 GT."""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BridgeDualResolutionDataset
from tools import get_transform, setup_seed
from tools.bridge_class_metrics import evaluate_bridge_classes
from tools.supervised_protocol import (
    BinaryProtocolMetrics, f1_from_counts, fixed_threshold_counts,
)
from train_anomalyclip_bd import features_and_maps, load_upstream


@torch.no_grad()
def evaluate(args):
    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    library, PromptLearner, _, _ = load_upstream(args.upstream_root)
    device = torch.device('cuda')
    details = {
        'Prompt_length': 12,
        'learnabel_text_embedding_depth': 9,
        'learnabel_text_embedding_length': 4,
    }
    model, _ = library.load(
        'ViT-L/14@336px', device=device, design_details=details,
        download_root=args.clip_cache,
    )
    model.visual.DAPM_replace(DPAM_layer=20)
    model.eval()
    prompt = PromptLearner(model.to('cpu'), details)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    prompt.load_state_dict(checkpoint['prompt_learner'])
    prompt.to(device).eval()
    model.to(device)
    transform, _ = get_transform(args.image_size)
    data = BridgeDualResolutionDataset(args.test_data_path, transform, 1024)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    metrics = BinaryProtocolMetrics(args.pixel_thresholds)
    counts = {'tp': 0, 'fp': 0, 'fn': 0}
    predictions, paths = [], []
    for batch in tqdm(loader, desc='test AnomalyCLIP-BD'):
        images = batch['img'].to(device)
        image_logits, maps = features_and_maps(
            library, model, prompt, images, args.features_list, args.image_size
        )
        probability = torch.stack([
            (item[:, 1] + 1.0 - item[:, 0]) / 2.0 for item in maps
        ]).mean(0).unsqueeze(1)
        probability = torch.nn.functional.interpolate(
            probability, size=(1024,1024), mode='bilinear', align_corners=False
        ).clamp(0,1)
        target = batch['native_mask'].to(device).unsqueeze(1)
        metrics.update(probability, target, batch['anomaly'])
        current = fixed_threshold_counts(probability, target, args.val_threshold)
        for key in counts:
            counts[key] += current[key]
        predictions.append(probability[:,0].cpu())
        paths.extend(batch['img_path'])
    result = metrics.compute()
    result['P-F1@val-threshold'] = f1_from_counts(counts)
    result['val_threshold'] = args.val_threshold
    pixel_scores = torch.cat(predictions)
    image_scores = pixel_scores.flatten(1).max(1).values
    per_defect = evaluate_bridge_classes(
        np.asarray(paths), image_scores, pixel_scores, args.pixel_thresholds,
        os.path.join(args.output_dir, 'bridge_defect_metrics.json'),
    )
    result['Macro-diagnostic-P-AP'] = float(np.mean([
        value['metrics_percent']['P-AP'] for value in per_defect.values()
    ]))
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as stream:
        json.dump({
            'model': 'AnomalyCLIP-BD',
            'protocol': {
                'clip_input': 518, 'metric_resolution': 1024,
                'gt': 'original frozen raster',
                'upstream_commit': checkpoint.get('upstream_commit'),
                'checkpoint_selection': 'validation overall P-AP',
            },
            'results_percent': result, 'per_defect': per_defect,
        }, stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream_root', required=True)
    parser.add_argument('--clip_cache', default='/home/skye/.cache/clip')
    parser.add_argument('--test_data_path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--val_threshold', required=True, type=float)
    parser.add_argument('--image_size', type=int, default=518)
    parser.add_argument('--features_list', type=int, nargs='+', default=[6,12,18,24])
    parser.add_argument('--pixel_thresholds', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=10)
    evaluate(parser.parse_args())
