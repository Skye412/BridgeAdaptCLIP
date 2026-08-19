"""Evaluate BridgeAdaptCLIP-v1 under Bridge2893 Protocol v2."""

import argparse
import json
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import (
    BRIDGE_ANOMALY_ANCHORS,
    BRIDGE_NORMAL_ANCHORS,
    BridgeAdaptCLIPV1,
    TextualAdapter,
    VisualAdapter,
)
from dataset import BridgeDualResolutionDataset
from tools import Evaluator, get_logger, get_transform, setup_seed
from tools.bridge_class_metrics import evaluate_bridge_classes


def evaluate(args):
    if args.reference_count != 0:
        raise ValueError('BridgeAdaptCLIP-v1 main path supports zero-reference only.')
    os.makedirs(args.save_path, exist_ok=True)
    logger = get_logger(
        args.save_path,
        f'bridge2893_{args.seed}seed_0shot_bridgeadaptclip_v1_test_log.txt',
    )
    logger.info(args)
    logger.info(
        'Protocol v2: CLIP input=%d, structural input=%d, metric resolution=%d, '
        'native GT, zero-reference',
        args.model_input_size, args.structural_input_size, args.metric_resolution,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=20)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    textual_learner = TextualAdapter(
        model.to('cpu'), args.model_input_size, args.n_ctx,
        static_normal_descriptions=BRIDGE_NORMAL_ANCHORS,
        static_anomaly_descriptions=BRIDGE_ANOMALY_ANCHORS,
    )
    visual_learner = VisualAdapter(
        args.model_input_size, 14, input_dim=768, reduction=args.vl_reduction
    )
    bridge_model = BridgeAdaptCLIPV1(
        semantic_channels=768,
        fusion_channels=args.fusion_channels,
        structural_channels=args.structural_channels,
        strip_kernel=args.strip_kernel,
        structural_input_size=args.structural_input_size,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    textual_learner.load_state_dict(checkpoint['textual_learner'])
    visual_learner.load_state_dict(checkpoint['visual_learner'])
    bridge_model.load_state_dict(checkpoint['bridgeadaptclip_v1'])

    model.to(device)
    textual_learner.to(device).eval()
    visual_learner.to(device).eval()
    bridge_model.to(device).eval()
    textual_learner.prepare_static_text_feature(model)

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

    with torch.no_grad():
        learned_prompts, tokenized_prompts = textual_learner()
        learned_text_features = model.encode_text_learn(
            learned_prompts, tokenized_prompts
        ).float()

    sample_ids = []
    gt_masks = []
    pr_masks = []
    cls_names = []
    gt_anomalys = []
    pr_anomalys = []
    query_paths = []
    amp_enabled = args.amp and device.type == 'cuda'

    for items in tqdm(loader):
        clip_image = items['img'].to(device, non_blocking=True)
        structural_image = items['structural_img'].to(device, non_blocking=True)
        native_mask = items['native_mask']
        image_target = items['anomaly']

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
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
            bridge_output = bridge_model(
                visual_patch_feature,
                visual_map[:, 1:2],
                textual_map[:, 1:2],
                structural_image,
            )
            pixel_score = torch.sigmoid(bridge_output['mask_logits'])[:, 0]

            # Preserve Original AdaptCLIP zero-reference image-level inference.
            semantic_pixel_map = 0.5 * (
                visual_map[:, 1] + textual_map[:, 1]
            )
            semantic_pixel_map = torch.stack([
                torch.from_numpy(gaussian_filter(score.float().cpu().numpy(), sigma=args.sigma))
                for score in semantic_pixel_map
            ]).to(device)
            semantic_map_max = semantic_pixel_map.flatten(1).max(dim=1).values
            global_visual_score = global_visual_logits.softmax(dim=-1)[:, 1]
            global_textual_score = global_textual_logits.softmax(dim=-1)[:, 1]
            image_score = (
                global_visual_score + global_textual_score + semantic_map_max
            ) / 3.0

        if pixel_score.shape[-2:] != (args.metric_resolution, args.metric_resolution):
            raise RuntimeError(
                f'Decoder output {tuple(pixel_score.shape[-2:])} does not match '
                f'metric resolution {args.metric_resolution}.'
            )
        pixel_score = torch.nan_to_num(pixel_score.float(), nan=0.0, posinf=1.0, neginf=0.0)
        image_score = torch.nan_to_num(image_score.float(), nan=0.0, posinf=1.0, neginf=0.0)

        sample_ids.append(np.asarray(items['sample_id']))
        cls_names.append(np.asarray(items['cls_name']))
        query_paths.append(np.asarray(items['img_path']))
        gt_masks.append(native_mask.int().cpu())
        pr_masks.append(pixel_score.cpu())
        gt_anomalys.append(image_target.int().cpu())
        pr_anomalys.append(image_score.cpu())

    results = {
        'sample_ids': np.concatenate(sample_ids),
        'gt_masks': torch.cat(gt_masks),
        'pr_masks': torch.cat(pr_masks),
        'cls_names': np.concatenate(cls_names),
        'gt_anomalys': torch.cat(gt_anomalys),
        'pr_anomalys': torch.cat(pr_anomalys),
        'query_paths': np.concatenate(query_paths),
    }

    per_defect = None
    macro_diagnostic_p_ap = None
    if args.bridge_class_metrics:
        per_defect_path = os.path.join(
            args.save_path, 'bridge_defect_metrics_10seed_0shot.json'
        )
        per_defect = evaluate_bridge_classes(
            results['query_paths'], results['pr_anomalys'], results['pr_masks'],
            args.pixel_thresholds, per_defect_path,
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
        rows, headers='keys', tablefmt='pipe', floatfmt='.1f',
        numalign='center', stralign='center',
    ))
    if per_defect:
        logger.info('Macro diagnostic P-AP: %.4f', macro_diagnostic_p_ap)
        for defect, report in per_defect.items():
            metrics = report['metrics_percent']
            logger.info(
                '%s: P-AP=%.4f, P-F1max=%.4f',
                defect, metrics['P-AP'], metrics['P-F1max'],
            )

    output_report = {
        'protocol': {
            'protocol_id': 'bridge2893-eval-v2',
            'model_name': 'BridgeAdaptCLIP-v1',
            'model_input_size': args.model_input_size,
            'structural_input_size': args.structural_input_size,
            'metric_resolution': args.metric_resolution,
            'reference_count': 0,
            'gt_source': 'original_frozen_1024_png_raster',
            'prediction_type': 'continuous_sigmoid_decoder_score',
            'pixel_threshold_bins': args.pixel_thresholds,
            'checkpoint_path': args.checkpoint_path,
        },
        'results_percent': full_precision_results,
        'macro_diagnostic_P-AP': macro_diagnostic_p_ap,
    }
    output_path = os.path.join(
        args.save_path, 'bridge2893_10seed_0shot_metrics.json'
    )
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(output_report, output_file, indent=2)


def build_parser():
    parser = argparse.ArgumentParser('BridgeAdaptCLIP-v1 evaluation')
    parser.add_argument('--test_data_path', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--save_path', required=True)
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
    args = build_parser().parse_args()
    setup_seed(args.seed)
    evaluate(args)
