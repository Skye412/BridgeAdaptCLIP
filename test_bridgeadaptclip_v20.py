"""Evaluate BridgeAdaptCLIP-v2.0 under native-1024 Protocol v2."""

import argparse
import json
import os

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import BridgeAdaptCLIPV12, BridgeAdaptCLIPV20, TextualAdapter, VisualAdapter
from dataset import BridgeDualResolutionDataset
from tools import Evaluator, get_logger, get_transform, setup_seed
from tools.bridge_class_metrics import evaluate_bridge_classes
from tools.bridge_masks import DEFECT_NAMES, decode_bridge_class_masks
from tools.bridge_row0 import (
    file_sha256, resize_row0_probability, row0_image_score, smooth_row0_probability,
)


def _freeze(module):
    for parameter in module.parameters(): parameter.requires_grad_(False)
    module.eval()


def evaluate(args):
    os.makedirs(args.save_path, exist_ok=True)
    logger = get_logger(args.save_path, 'bridge2893_10seed_0shot_bridgeadaptclipv20_test_log.txt')
    logger.info(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    clip_model.visual.DAPM_replace(DPAM_layer=20); _freeze(clip_model)
    textual = TextualAdapter(clip_model.to('cpu'), args.model_input_size, args.n_ctx)
    visual = VisualAdapter(args.model_input_size, 14, input_dim=768, reduction=args.vl_reduction)
    row0_checkpoint = torch.load(args.row0_checkpoint_path, map_location='cpu')
    textual.load_state_dict(row0_checkpoint['textual_learner'])
    visual.load_state_dict(row0_checkpoint['visual_learner']); _freeze(textual); _freeze(visual)
    fine_model = BridgeAdaptCLIPV12(
        semantic_channels=768, fusion_channels=args.fusion_channels,
        structural_channels=args.structural_channels, strip_kernel=args.strip_kernel,
        structural_input_size=args.structural_input_size,
        probability_epsilon=args.probability_epsilon,
    )
    fine_checkpoint = torch.load(args.fine_checkpoint_path, map_location='cpu')
    fine_model.load_state_dict(fine_checkpoint[args.fine_checkpoint_state_key]); _freeze(fine_model)
    broad_model = BridgeAdaptCLIPV20(
        joint_channels=args.fusion_channels, broad_channels=args.broad_channels,
        output_size=args.structural_input_size,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    broad_model.load_state_dict(checkpoint['bridgeadaptclip_v20']); _freeze(broad_model)
    if checkpoint.get('row0_checkpoint_sha256') != file_sha256(args.row0_checkpoint_path):
        raise ValueError('Row-0 checkpoint hash mismatch')
    if checkpoint.get('fine_checkpoint_sha256') != file_sha256(args.fine_checkpoint_path):
        raise ValueError('Fine checkpoint hash mismatch')
    clip_model.to(device); textual.to(device); visual.to(device)
    fine_model.to(device); broad_model.to(device)
    textual.prepare_static_text_feature(clip_model)
    with torch.no_grad():
        prompts, tokens = textual()
        learned_text = clip_model.encode_text_learn(prompts, tokens).float()

    clip_transform, _ = get_transform(image_size=args.model_input_size)
    dataset = BridgeDualResolutionDataset(
        args.test_data_path, clip_transform=clip_transform,
        structural_input_size=args.structural_input_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    evaluator = Evaluator('cpu', metrics=args.eval_metrics, sample_level=False,
                          pixel_thresholds=args.pixel_thresholds, pro_thresholds=args.pro_thresholds)
    records = {key: [] for key in (
        'sample_ids','gt_masks','pr_masks','cls_names','gt_anomalys','pr_anomalys','query_paths'
    )}
    sums = {key: 0.0 for key in ('gate','magnitude','abs_correction','correction')}
    pixel_count = 0
    region_stats = {
        key: {'count': 0, 'correction_sum': 0.0, 'abs_correction_sum': 0.0, 'gate_sum': 0.0}
        for key in ('background', 'positive', 'fp_like', 'correct_background', *DEFECT_NAMES)
    }
    corr_stats = {'n': 0, 'x': 0.0, 'y': 0.0, 'xx': 0.0, 'yy': 0.0, 'xy': 0.0}
    amp_enabled = args.amp and device.type == 'cuda'
    for items in tqdm(loader):
        clip_image = items['img'].to(device, non_blocking=True)
        structural = items['structural_img'].to(device, non_blocking=True)
        with torch.no_grad():
            image_features, patch_features = clip_model.encode_image(
                clip_image, args.features_list, DPAM_layer=20
            )
            gv, visual_map, visual_patch = visual.forward_with_features(
                image_features, patch_features, textual.static_text_features
            )
            gt, textual_map = textual.compute_global_local_score(
                image_features, patch_features, learned_text
            )
            smoothed = smooth_row0_probability(visual_map, textual_map, sigma=args.sigma)
            row0_probability = resize_row0_probability(
                smoothed, metric_resolution=args.metric_resolution, device=device
            )
            image_score = row0_image_score(gv, gt, smoothed)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                fine_output = fine_model(visual_patch, row0_probability, structural)
                output = broad_model(
                    fine_output['joint_feature'], fine_output['mask_logits'], row0_probability
                )
            pixel_score = torch.sigmoid(output['mask_logits'])[:, 0]
        target = items['native_mask'].to(device, non_blocking=True).unsqueeze(1) > 0.5
        correction = output['broad_correction'].float()
        gate = output['broad_gate'].float()
        fine_probability = output['fine_probability'].float()
        masks = {
            'background': ~target,
            'positive': target,
            'fp_like': (~target) & (fine_probability >= 0.5),
            'correct_background': (~target) & (fine_probability < 0.5),
        }
        for key, mask in masks.items():
            count = int(mask.sum())
            if count:
                state = region_stats[key]; state['count'] += count
                state['correction_sum'] += float(correction[mask].sum())
                state['abs_correction_sum'] += float(correction[mask].abs().sum())
                state['gate_sum'] += float(gate[mask].sum())
        for index, image_path in enumerate(items['img_path']):
            if os.path.basename(os.path.dirname(str(image_path))) == 'normal':
                continue
            class_masks, _ = decode_bridge_class_masks(str(image_path))
            for defect, class_mask in class_masks.items():
                mask = torch.from_numpy(class_mask.copy()).to(device=device, dtype=torch.bool)
                count = int(mask.sum())
                if count:
                    state = region_stats[defect]; state['count'] += count
                    values = correction[index, 0][mask]; gates = gate[index, 0][mask]
                    state['correction_sum'] += float(values.sum())
                    state['abs_correction_sum'] += float(values.abs().sum())
                    state['gate_sum'] += float(gates.sum())
        error_target = ((~target).float() * fine_probability).double()
        gate_double = gate.double()
        corr_stats['n'] += gate.numel()
        corr_stats['x'] += float(gate_double.sum()); corr_stats['y'] += float(error_target.sum())
        corr_stats['xx'] += float((gate_double * gate_double).sum())
        corr_stats['yy'] += float((error_target * error_target).sum())
        corr_stats['xy'] += float((gate_double * error_target).sum())
        pixel_score = torch.nan_to_num(pixel_score.float(), nan=0., posinf=1., neginf=0.)
        image_score = torch.nan_to_num(image_score.float(), nan=0., posinf=1., neginf=0.)
        for key, tensor_key in (
            ('gate','broad_gate'), ('magnitude','broad_magnitude'),
            ('abs_correction','broad_correction'), ('correction','broad_correction')
        ):
            value = output[tensor_key].float()
            sums[key] += float(value.abs().sum() if key == 'abs_correction' else value.sum())
        pixel_count += output['broad_gate'].numel()
        records['sample_ids'].append(np.asarray(items['sample_id']))
        records['cls_names'].append(np.asarray(items['cls_name']))
        records['query_paths'].append(np.asarray(items['img_path']))
        records['gt_masks'].append(items['native_mask'].int().cpu())
        records['pr_masks'].append(pixel_score.cpu())
        records['gt_anomalys'].append(items['anomaly'].int().cpu())
        records['pr_anomalys'].append(image_score.cpu())
    results = {key: np.concatenate(value) if key in ('sample_ids','cls_names','query_paths')
               else torch.cat(value) for key, value in records.items()}
    per_defect = None; macro = None
    if args.bridge_class_metrics:
        per_defect = evaluate_bridge_classes(
            results['query_paths'], results['pr_anomalys'], results['pr_masks'],
            args.pixel_thresholds, os.path.join(args.save_path, 'bridge_defect_metrics_10seed_0shot.json')
        )
        macro = float(np.mean([x['metrics_percent']['P-AP'] for x in per_defect.values()]))
    rows = {'Name': []}; full = {}
    for cls_name in dataset.obj_list:
        metric_results = evaluator.run(results, cls_name, logger)
        rows['Name'].append(cls_name); full[cls_name] = {}
        for metric in args.eval_metrics:
            value = 100.0 * metric_results[metric]
            rows.setdefault(metric, []).append(value); full[cls_name][metric] = value
    logger.info('\n' + tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.4f'))
    diagnostics = {key: value / pixel_count for key, value in sums.items()}
    diagnostics['regions'] = {
        key: {
            'pixel_count': state['count'],
            'mean_correction': state['correction_sum'] / max(state['count'], 1),
            'mean_abs_correction': state['abs_correction_sum'] / max(state['count'], 1),
            'mean_gate': state['gate_sum'] / max(state['count'], 1),
        }
        for key, state in region_stats.items()
    }
    n = corr_stats['n']
    covariance = corr_stats['xy'] - corr_stats['x'] * corr_stats['y'] / n
    variance_x = corr_stats['xx'] - corr_stats['x'] ** 2 / n
    variance_y = corr_stats['yy'] - corr_stats['y'] ** 2 / n
    diagnostics['pearson_gate_vs_fp_target'] = covariance / max(
        (max(variance_x, 0.0) * max(variance_y, 0.0)) ** 0.5, 1e-12
    )
    diagnostics['fp_gate_over_correct_background_ratio'] = (
        diagnostics['regions']['fp_like']['mean_gate']
        / max(diagnostics['regions']['correct_background']['mean_gate'], 1e-12)
    )
    logger.info('broad diagnostics: %s', diagnostics)
    if per_defect:
        logger.info('Macro diagnostic P-AP: %.6f', macro)
        for defect, report in per_defect.items():
            logger.info('%s: P-AP=%.6f', defect, report['metrics_percent']['P-AP'])
    report = {
        'protocol': {
            'protocol_id':'bridge2893-eval-v2', 'model_name':'BridgeAdaptCLIP-v2.0',
            'model_input_size':args.model_input_size, 'structural_input_size':args.structural_input_size,
            'metric_resolution':args.metric_resolution, 'reference_count':0,
            'fine_base':'frozen_bridgeadaptclip_v13_epoch3',
            'fine_checkpoint_path':args.fine_checkpoint_path,
            'checkpoint_path':args.checkpoint_path, 'image_score_policy':'exact_frozen_row0',
            'prediction_type':'fine_logits_plus_non_positive_broad_correction',
            'gt_source':'original_frozen_1024_png_raster',
        },
        'results_percent':full, 'macro_diagnostic_P-AP':macro,
        'broad_diagnostics':diagnostics,
    }
    with open(os.path.join(args.save_path, 'bridge2893_10seed_0shot_metrics.json'),'w',encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)


def build_parser():
    p=argparse.ArgumentParser('BridgeAdaptCLIP-v2.0 evaluation')
    for name in ('test_data_path','checkpoint_path','row0_checkpoint_path','fine_checkpoint_path','save_path'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--fine_checkpoint_state_key',default='bridgeadaptclip_v13')
    p.add_argument('--pretrained_model',default='ViT-L/14@336px')
    p.add_argument('--features_list',type=int,nargs='+',default=[6,12,18,24])
    p.add_argument('--model_input_size',type=int,default=518); p.add_argument('--structural_input_size',type=int,default=1024)
    p.add_argument('--metric_resolution',type=int,default=1024); p.add_argument('--n_ctx',type=int,default=12)
    p.add_argument('--vl_reduction',type=int,default=4); p.add_argument('--fusion_channels',type=int,default=128)
    p.add_argument('--structural_channels',type=int,default=128); p.add_argument('--broad_channels',type=int,default=128)
    p.add_argument('--strip_kernel',type=int,default=5); p.add_argument('--probability_epsilon',type=float,default=1e-6)
    p.add_argument('--batch_size',type=int,default=2); p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=10); p.add_argument('--sigma',type=float,default=4.0)
    p.add_argument('--eval_metrics',nargs='+',default=['I-AUROC','I-AP','I-F1max','P-AUROC','P-AP','P-F1max'])
    p.add_argument('--pixel_thresholds',type=int,default=2048); p.add_argument('--pro_thresholds',type=int,default=256)
    p.add_argument('--bridge_class_metrics',action='store_true'); p.add_argument('--amp',action='store_true')
    return p


if __name__ == '__main__':
    args=build_parser().parse_args(); setup_seed(args.seed); evaluate(args)
