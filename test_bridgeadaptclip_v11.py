"""Evaluate BridgeAdaptCLIP-v1.1 under Bridge2893 Protocol v2."""

import argparse
import json
import os

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import BridgeAdaptCLIPV11, BridgeAdaptCLIPV12, TextualAdapter, VisualAdapter
from dataset import BridgeDualResolutionDataset
from tools import Evaluator, get_logger, get_transform, setup_seed
from tools.bridge_class_metrics import evaluate_bridge_classes
from tools.bridge_row0 import (
    file_sha256,
    resize_row0_probability,
    row0_image_score,
    smooth_row0_probability,
)


def _freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def evaluate(args):
    if args.reference_count != 0:
        raise ValueError('BridgeAdaptCLIP-v1.1 supports zero-reference only.')
    if args.metric_resolution != args.structural_input_size:
        raise ValueError('v1.1 requires metric_resolution == structural_input_size.')
    os.makedirs(args.save_path, exist_ok=True)
    model_slug = args.model_name.lower().replace('-', '').replace('.', '')
    logger = get_logger(
        args.save_path,
        f'bridge2893_{args.seed}seed_0shot_{model_slug}_test_log.txt',
    )
    logger.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=20)
    _freeze(model)

    textual_learner = TextualAdapter(model.to('cpu'), args.model_input_size, args.n_ctx)
    visual_learner = VisualAdapter(
        args.model_input_size, 14, input_dim=768, reduction=args.vl_reduction
    )
    row0_checkpoint = torch.load(args.row0_checkpoint_path, map_location='cpu')
    textual_learner.load_state_dict(row0_checkpoint['textual_learner'])
    visual_learner.load_state_dict(row0_checkpoint['visual_learner'])
    _freeze(textual_learner)
    _freeze(visual_learner)

    bridge_class = (
        BridgeAdaptCLIPV12
        if args.checkpoint_state_key in (
            'bridgeadaptclip_v12', 'bridgeadaptclip_v13', 'bridgeadaptclip_v14',
            'bridgeadaptclip_v15', 'bridgeadaptclip_v16'
        )
        else BridgeAdaptCLIPV11
    )
    bridge_model = bridge_class(
        semantic_channels=768,
        fusion_channels=args.fusion_channels,
        structural_channels=args.structural_channels,
        strip_kernel=args.strip_kernel,
        structural_input_size=args.structural_input_size,
        probability_epsilon=args.probability_epsilon,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    bridge_model.load_state_dict(checkpoint[args.checkpoint_state_key])
    expected_sha = checkpoint.get('row0_checkpoint_sha256')
    actual_sha = file_sha256(args.row0_checkpoint_path)
    if expected_sha and expected_sha != actual_sha:
        raise ValueError('Evaluation Row-0 checkpoint does not match the training checkpoint.')

    model.to(device)
    textual_learner.to(device)
    visual_learner.to(device)
    bridge_model.to(device).eval()
    textual_learner.prepare_static_text_feature(model)
    with torch.no_grad():
        learned_prompts, tokenized_prompts = textual_learner()
        learned_text_features = model.encode_text_learn(
            learned_prompts, tokenized_prompts
        ).float()

    clip_transform, _ = get_transform(image_size=args.model_input_size)
    dataset = BridgeDualResolutionDataset(
        args.test_data_path,
        clip_transform=clip_transform,
        structural_input_size=args.structural_input_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    evaluator = Evaluator(
        'cpu', metrics=args.eval_metrics, sample_level=False,
        pixel_thresholds=args.pixel_thresholds,
        pro_thresholds=args.pro_thresholds,
    )

    records = {key: [] for key in [
        'sample_ids', 'gt_masks', 'pr_masks', 'cls_names',
        'gt_anomalys', 'pr_anomalys', 'query_paths'
    ]}
    amp_enabled = args.amp and device.type == 'cuda'
    gate_sum = 0.0
    residual_sum = 0.0
    pixel_count = 0

    for items in tqdm(loader):
        clip_image = items['img'].to(device, non_blocking=True)
        structural_image = items['structural_img'].to(device, non_blocking=True)
        # Frozen semantic inference intentionally matches formal Row 0 exactly.
        with torch.no_grad():
            image_features, patch_features = model.encode_image(
                clip_image, args.features_list, DPAM_layer=20
            )
            global_visual_logits, visual_map, visual_patch_feature = (
                visual_learner.forward_with_features(
                    image_features, patch_features,
                    textual_learner.static_text_features,
                )
            )
            global_textual_logits, textual_map = (
                textual_learner.compute_global_local_score(
                    image_features, patch_features, learned_text_features
                )
            )
            smoothed_row0 = smooth_row0_probability(
                visual_map, textual_map, sigma=args.sigma
            )
            row0_probability = resize_row0_probability(
                smoothed_row0,
                metric_resolution=args.metric_resolution,
                device=device,
            )
            image_score = row0_image_score(
                global_visual_logits, global_textual_logits, smoothed_row0
            )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            output = bridge_model(
                visual_patch_feature, row0_probability, structural_image
            )
            pixel_score = torch.sigmoid(output['mask_logits'])[:, 0]

        pixel_score = torch.nan_to_num(pixel_score.float(), nan=0.0, posinf=1.0, neginf=0.0)
        image_score = torch.nan_to_num(image_score.float(), nan=0.0, posinf=1.0, neginf=0.0)
        gate_sum += float(output['gate'].float().sum())
        residual_sum += float(output['gated_residual'].float().abs().sum())
        pixel_count += output['gate'].numel()

        records['sample_ids'].append(np.asarray(items['sample_id']))
        records['cls_names'].append(np.asarray(items['cls_name']))
        records['query_paths'].append(np.asarray(items['img_path']))
        records['gt_masks'].append(items['native_mask'].int().cpu())
        records['pr_masks'].append(pixel_score.cpu())
        records['gt_anomalys'].append(items['anomaly'].int().cpu())
        records['pr_anomalys'].append(image_score.cpu())

    results = {
        key: np.concatenate(value) if key in ['sample_ids', 'cls_names', 'query_paths']
        else torch.cat(value)
        for key, value in records.items()
    }
    per_defect = None
    macro_diagnostic_p_ap = None
    if args.bridge_class_metrics:
        per_defect = evaluate_bridge_classes(
            results['query_paths'], results['pr_anomalys'], results['pr_masks'],
            args.pixel_thresholds,
            os.path.join(args.save_path, 'bridge_defect_metrics_10seed_0shot.json'),
        )
        macro_diagnostic_p_ap = float(np.mean([
            report['metrics_percent']['P-AP'] for report in per_defect.values()
        ]))

    rows = {'Name': []}
    full_precision_results = {}
    for cls_name in dataset.obj_list:
        metric_results = evaluator.run(results, cls_name, logger)
        rows['Name'].append(cls_name)
        full_precision_results[cls_name] = {}
        for metric_name in args.eval_metrics:
            value = 100.0 * metric_results[metric_name]
            rows.setdefault(metric_name, []).append(value)
            full_precision_results[cls_name][metric_name] = value
    logger.info('\n' + tabulate(
        rows, headers='keys', tablefmt='pipe', floatfmt='.4f',
        numalign='center', stralign='center',
    ))
    logger.info('mean_gate=%.8f mean_abs_gated_residual=%.8f',
                gate_sum / pixel_count, residual_sum / pixel_count)
    if per_defect:
        logger.info('Macro diagnostic P-AP: %.6f', macro_diagnostic_p_ap)
        for defect, report in per_defect.items():
            logger.info('%s: P-AP=%.6f, P-F1max=%.6f', defect,
                        report['metrics_percent']['P-AP'],
                        report['metrics_percent']['P-F1max'])

    report = {
        'protocol': {
            'protocol_id': 'bridge2893-eval-v2',
            'model_name': args.model_name,
            'model_input_size': args.model_input_size,
            'structural_input_size': args.structural_input_size,
            'metric_resolution': args.metric_resolution,
            'reference_count': 0,
            'semantic_base': 'frozen_original_adaptclip_row0_epoch14',
            'row0_checkpoint_path': args.row0_checkpoint_path,
            'row0_checkpoint_sha256': actual_sha,
            'checkpoint_path': args.checkpoint_path,
            'image_score_policy': 'exact_frozen_row0',
            'prediction_type': 'continuous_sigmoid_gated_logit_residual',
            'gt_source': 'original_frozen_1024_png_raster',
        },
        'results_percent': full_precision_results,
        'macro_diagnostic_P-AP': macro_diagnostic_p_ap,
        'residual_diagnostics': {
            'mean_gate': gate_sum / pixel_count,
            'mean_abs_gated_residual': residual_sum / pixel_count,
        },
    }
    with open(os.path.join(args.save_path, 'bridge2893_10seed_0shot_metrics.json'), 'w', encoding='utf-8') as output:
        json.dump(report, output, indent=2)


def build_parser():
    parser = argparse.ArgumentParser('BridgeAdaptCLIP-v1.1 evaluation')
    parser.add_argument('--test_data_path', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--row0_checkpoint_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--model_name', default='BridgeAdaptCLIP-v1.1')
    parser.add_argument('--checkpoint_state_key', default='bridgeadaptclip_v11')
    parser.add_argument('--pretrained_model', default='ViT-L/14@336px')
    parser.add_argument('--features_list', type=int, nargs='+', default=[6, 12, 18, 24])
    parser.add_argument('--model_input_size', type=int, default=518)
    parser.add_argument('--structural_input_size', type=int, default=1024)
    parser.add_argument('--metric_resolution', type=int, default=1024)
    parser.add_argument('--reference_count', type=int, default=0)
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--vl_reduction', type=int, default=4)
    parser.add_argument('--fusion_channels', type=int, default=128)
    parser.add_argument('--structural_channels', type=int, default=128)
    parser.add_argument('--strip_kernel', type=int, default=5)
    parser.add_argument('--probability_epsilon', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=4.0)
    parser.add_argument('--eval_metrics', nargs='+', default=[
        'I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max'
    ])
    parser.add_argument('--pixel_thresholds', type=int, default=2048)
    parser.add_argument('--pro_thresholds', type=int, default=256)
    parser.add_argument('--bridge_class_metrics', action='store_true')
    parser.add_argument('--amp', action='store_true')
    return parser


if __name__ == '__main__':
    parsed_args = build_parser().parse_args()
    setup_seed(parsed_args.seed)
    evaluate(parsed_args)
