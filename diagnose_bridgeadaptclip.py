"""Run the three locked zero-cost BridgeAdaptCLIP-v1 diagnostics."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
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
from tools import get_transform, setup_seed
from tools.bridge_class_metrics import evaluate_bridge_classes
from tools.bridge_diagnostics import (
    bridge_source_from_path,
    fuse_probabilities,
    fusion_grid,
    select_validation_fusion,
)
from tools.bridge_masks import DEFECT_NAMES, decode_bridge_class_masks
from tools.effecient_metric import Evaluator


STREAM_ROW0 = 'row0_semantic'
STREAM_FULL_SEMANTIC = 'full_v1_semantic_only'
STREAM_FULL_DECODER = 'full_v1_decoder'
STREAM_SELECTED_FUSION = 'validation_selected_fusion'
PIXEL_METRICS = ('P-AUROC', 'P-AP', 'P-F1max')


def _image_metrics(targets, scores):
    targets = np.asarray(targets, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    precision, recall, _ = precision_recall_curve(targets, scores)
    f1 = np.max(2 * precision * recall / (precision + recall + 1e-12))
    return {
        'I-AUROC': 100.0 * roc_auc_score(targets, scores),
        'I-AP': 100.0 * average_precision_score(targets, scores),
        'I-F1max': 100.0 * f1,
    }


def _histogram_metrics(predictions, targets, num_thresholds):
    metrics = Evaluator._compute_binned_pixel_metrics(
        predictions.reshape(-1), targets.reshape(-1), num_thresholds
    )
    return {
        'P-AUROC': 100.0 * metrics['auroc'],
        'P-AP': 100.0 * metrics['ap'],
        'P-F1max': 100.0 * metrics['f1max'],
    }


def _all_metrics(predictions, targets, image_scores, image_targets, num_thresholds):
    return {
        **_image_metrics(image_targets, image_scores),
        **_histogram_metrics(predictions, targets, num_thresholds),
    }


def _subset_metrics(
    predictions, targets, image_scores, image_targets, indices, num_thresholds
):
    positive_hist = torch.zeros(num_thresholds, dtype=torch.float64)
    negative_hist = torch.zeros(num_thresholds, dtype=torch.float64)
    for index in indices:
        prediction = predictions[index].float().clamp(0, 1)
        target = targets[index].bool()
        bins = torch.clamp(
            (prediction * (num_thresholds - 1)).floor().long(),
            min=0,
            max=num_thresholds - 1,
        )
        positive_hist += torch.bincount(
            bins[target], minlength=num_thresholds
        ).to(torch.float64)
        negative_hist += torch.bincount(
            bins[~target], minlength=num_thresholds
        ).to(torch.float64)
    pixel = Evaluator._metrics_from_histograms(positive_hist, negative_hist)
    subset_targets = image_targets[indices].numpy()
    subset_scores = image_scores[indices].numpy()
    return {
        **_image_metrics(subset_targets, subset_scores),
        'P-AUROC': 100.0 * pixel['auroc'],
        'P-AP': 100.0 * pixel['ap'],
        'P-F1max': 100.0 * pixel['f1max'],
    }


def _smooth_and_resize(score_map, metric_resolution, sigma):
    smoothed = torch.stack([
        torch.from_numpy(gaussian_filter(score.float().cpu().numpy(), sigma=sigma))
        for score in score_map
    ])
    return F.interpolate(
        smoothed[:, None],
        size=(metric_resolution, metric_resolution),
        mode='bilinear',
        align_corners=False,
    )[:, 0]


def _semantic_outputs(
    visual_learner, textual_learner, static_text_features, learned_text_features,
    image_features, patch_features, metric_resolution, sigma,
):
    global_visual_logits, visual_map = visual_learner(
        image_features, patch_features, static_text_features
    )
    global_textual_logits, textual_map = textual_learner.compute_global_local_score(
        image_features, patch_features, learned_text_features
    )
    semantic_518 = 0.5 * (visual_map[:, 1] + textual_map[:, 1])
    semantic_native = _smooth_and_resize(semantic_518, metric_resolution, sigma)
    semantic_max = torch.stack([
        torch.from_numpy(gaussian_filter(score.float().cpu().numpy(), sigma=sigma))
        for score in semantic_518
    ]).flatten(1).max(dim=1).values.to(global_visual_logits.device)
    image_score = (
        global_visual_logits.softmax(dim=-1)[:, 1]
        + global_textual_logits.softmax(dim=-1)[:, 1]
        + semantic_max
    ) / 3.0
    return semantic_native, image_score, visual_map, textual_map


def _load_models(args, device):
    model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=20)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    row0_textual = TextualAdapter(model.to('cpu'), args.model_input_size, args.n_ctx)
    row0_visual = VisualAdapter(
        args.model_input_size, 14, input_dim=768, reduction=args.vl_reduction
    )
    full_textual = TextualAdapter(
        model.to('cpu'), args.model_input_size, args.n_ctx,
        static_normal_descriptions=BRIDGE_NORMAL_ANCHORS,
        static_anomaly_descriptions=BRIDGE_ANOMALY_ANCHORS,
    )
    full_visual = VisualAdapter(
        args.model_input_size, 14, input_dim=768, reduction=args.vl_reduction
    )
    bridge_model = BridgeAdaptCLIPV1(
        semantic_channels=768,
        fusion_channels=args.fusion_channels,
        structural_channels=args.structural_channels,
        strip_kernel=args.strip_kernel,
        structural_input_size=args.structural_input_size,
    )

    row0_checkpoint = torch.load(args.row0_checkpoint, map_location='cpu')
    full_checkpoint = torch.load(args.full_checkpoint, map_location='cpu')
    row0_textual.load_state_dict(row0_checkpoint['textual_learner'])
    row0_visual.load_state_dict(row0_checkpoint['visual_learner'])
    full_textual.load_state_dict(full_checkpoint['textual_learner'])
    full_visual.load_state_dict(full_checkpoint['visual_learner'])
    bridge_model.load_state_dict(full_checkpoint['bridgeadaptclip_v1'])

    model.to(device)
    modules = (row0_textual, row0_visual, full_textual, full_visual, bridge_model)
    for module in modules:
        module.to(device).eval()
    row0_textual.prepare_static_text_feature(model)
    full_textual.prepare_static_text_feature(model)
    with torch.no_grad():
        row0_prompts, row0_tokens = row0_textual()
        row0_learned_text = model.encode_text_learn(row0_prompts, row0_tokens).float()
        full_prompts, full_tokens = full_textual()
        full_learned_text = model.encode_text_learn(full_prompts, full_tokens).float()
    return (
        model, row0_textual, row0_visual, row0_learned_text,
        full_textual, full_visual, full_learned_text, bridge_model,
    )


def collect_predictions(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (
        model, row0_textual, row0_visual, row0_learned_text,
        full_textual, full_visual, full_learned_text, bridge_model,
    ) = _load_models(args, device)
    clip_transform, _ = get_transform(image_size=args.model_input_size)
    dataset = BridgeDualResolutionDataset(
        args.data_path,
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

    predictions = {name: [] for name in (
        STREAM_ROW0, STREAM_FULL_SEMANTIC, STREAM_FULL_DECODER
    )}
    image_scores = {name: [] for name in predictions}
    targets = []
    image_targets = []
    paths = []
    sample_ids = []
    amp_enabled = args.amp and device.type == 'cuda'

    for items in tqdm(loader, desc=f'diagnostic-{args.phase}'):
        clip_image = items['img'].to(device, non_blocking=True)
        structural_image = items['structural_img'].to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            image_features, patch_features = model.encode_image(
                clip_image, args.features_list, DPAM_layer=20
            )
            row0_map, row0_image, _, _ = _semantic_outputs(
                row0_visual, row0_textual, row0_textual.static_text_features,
                row0_learned_text, image_features, patch_features,
                args.metric_resolution, args.sigma,
            )
            full_map, full_image, full_visual_map, full_textual_map = _semantic_outputs(
                full_visual, full_textual, full_textual.static_text_features,
                full_learned_text, image_features, patch_features,
                args.metric_resolution, args.sigma,
            )
            _, _, full_patch_feature = full_visual.forward_with_features(
                image_features, patch_features, full_textual.static_text_features
            )
            decoder = bridge_model(
                full_patch_feature,
                full_visual_map[:, 1:2],
                full_textual_map[:, 1:2],
                structural_image,
            )
            decoder_map = torch.sigmoid(decoder['mask_logits'])[:, 0]

        predictions[STREAM_ROW0].append(row0_map.float().cpu())
        predictions[STREAM_FULL_SEMANTIC].append(full_map.float().cpu())
        predictions[STREAM_FULL_DECODER].append(decoder_map.float().cpu())
        image_scores[STREAM_ROW0].append(row0_image.float().cpu())
        image_scores[STREAM_FULL_SEMANTIC].append(full_image.float().cpu())
        image_scores[STREAM_FULL_DECODER].append(full_image.float().cpu())
        targets.append(items['native_mask'].to(torch.uint8))
        image_targets.append(items['anomaly'].to(torch.uint8))
        paths.extend(items['img_path'])
        sample_ids.extend(items['sample_id'])

    return {
        'predictions': {key: torch.cat(value) for key, value in predictions.items()},
        'image_scores': {key: torch.cat(value) for key, value in image_scores.items()},
        'targets': torch.cat(targets),
        'image_targets': torch.cat(image_targets),
        'paths': np.asarray(paths),
        'sample_ids': np.asarray(sample_ids),
    }


def _pixel_composition(paths):
    report = {
        key: {
            'images': 0, 'normal_images': 0, 'defect_images': 0,
            'defect_pixels': 0,
            'classes': {
                name: {'positive_images': 0, 'pixels': 0}
                for name in DEFECT_NAMES
            },
        }
        for key in ('ALL', 'CODEBRIM', 'S2DS')
    }
    for path in paths:
        source = bridge_source_from_path(path)
        is_normal = os.path.basename(os.path.dirname(path)) == 'normal'
        for key in ('ALL', source):
            report[key]['images'] += 1
            report[key]['normal_images' if is_normal else 'defect_images'] += 1
        if is_normal:
            continue
        class_masks, any_defect = decode_bridge_class_masks(path)
        defect_pixels = int(any_defect.sum())
        for key in ('ALL', source):
            report[key]['defect_pixels'] += defect_pixels
        for name, mask in class_masks.items():
            if not mask.any():
                continue
            pixels = int(mask.sum())
            for key in ('ALL', source):
                report[key]['classes'][name]['positive_images'] += 1
                report[key]['classes'][name]['pixels'] += pixels
    for group in report.values():
        denominator = max(group['defect_pixels'], 1)
        for values in group['classes'].values():
            values['fraction_of_defect_pixels'] = values['pixels'] / denominator
    return report


def _evaluate_streams(data, streams, output_dir, num_thresholds):
    report = {}
    sources = np.asarray([bridge_source_from_path(path) for path in data['paths']])
    for name, prediction in streams.items():
        scores = data['image_scores'].get(name, data['image_scores'][STREAM_ROW0])
        stream_report = {
            'overall_metrics_percent': _all_metrics(
                prediction, data['targets'], scores, data['image_targets'], num_thresholds
            ),
            'source_metrics_percent': {},
        }
        for source in ('CODEBRIM', 'S2DS'):
            indices = np.flatnonzero(sources == source)
            stream_report['source_metrics_percent'][source] = {
                'support_images': int(len(indices)),
                'metrics': _subset_metrics(
                    prediction, data['targets'], scores, data['image_targets'],
                    indices, num_thresholds,
                ),
            }
        defect_path = os.path.join(output_dir, f'{name}_defect_metrics.json')
        stream_report['defect_metrics'] = evaluate_bridge_classes(
            data['paths'], scores, prediction, num_thresholds, defect_path
        )
        report[name] = stream_report
    return report


def _check_reproduction(name, observed, expected, tolerance):
    difference = abs(observed - expected)
    if difference > tolerance:
        raise RuntimeError(
            f'{name} P-AP reproduction failed: observed={observed:.8f}, '
            f'expected={expected:.8f}, difference={difference:.8f}'
        )
    return {'expected': expected, 'observed': observed, 'absolute_difference': difference}


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    data = collect_predictions(args)
    base_streams = data['predictions']
    base_report = _evaluate_streams(
        data, base_streams, args.output_dir, args.pixel_thresholds
    )
    reproduction = {
        STREAM_ROW0: _check_reproduction(
            STREAM_ROW0,
            base_report[STREAM_ROW0]['overall_metrics_percent']['P-AP'],
            args.expected_row0_p_ap,
            args.reproduction_tolerance,
        ),
        STREAM_FULL_DECODER: _check_reproduction(
            STREAM_FULL_DECODER,
            base_report[STREAM_FULL_DECODER]['overall_metrics_percent']['P-AP'],
            args.expected_full_p_ap,
            args.reproduction_tolerance,
        ),
    }

    report = {
        'protocol': {
            'phase': args.phase,
            'selection_split': 'validation',
            'selection_metric': 'native-1024 P-AP',
            'fusion_weights': [step / 10.0 for step in range(11)],
            'fusion_forms': ['linear', 'probability_or'],
            'test_tuning_forbidden': True,
            'row0_checkpoint': args.row0_checkpoint,
            'full_checkpoint': args.full_checkpoint,
            'metric_resolution': args.metric_resolution,
        },
        'reproduction_checks': reproduction,
        'pixel_composition': _pixel_composition(data['paths']),
        'streams': base_report,
    }

    if args.phase == 'validation':
        candidates = []
        for candidate in tqdm(fusion_grid(), desc='validation-fusion-scan'):
            fused = fuse_probabilities(
                base_streams[STREAM_ROW0], base_streams[STREAM_FULL_DECODER],
                candidate['form'], candidate['weight'],
            )
            metrics = _histogram_metrics(
                fused, data['targets'], args.pixel_thresholds
            )
            candidates.append({**candidate, 'metrics_percent': metrics})
        selected = select_validation_fusion(candidates)
        selected_map = fuse_probabilities(
            base_streams[STREAM_ROW0], base_streams[STREAM_FULL_DECODER],
            selected['form'], selected['weight'],
        )
        data['image_scores'][STREAM_SELECTED_FUSION] = data['image_scores'][STREAM_ROW0]
        selected_report = _evaluate_streams(
            data, {STREAM_SELECTED_FUSION: selected_map},
            args.output_dir, args.pixel_thresholds,
        )[STREAM_SELECTED_FUSION]
        report['fusion_scan'] = candidates
        report['selected_fusion'] = {
            **selected,
            'complete_validation_report': selected_report,
        }
        selection_path = os.path.join(args.output_dir, 'fusion_selection.json')
        with open(selection_path, 'w', encoding='utf-8') as output:
            json.dump({
                'selected_on': 'validation_only',
                'selection_metric': 'P-AP',
                'form': selected['form'],
                'weight': selected['weight'],
                'validation_metrics_percent': selected_report['overall_metrics_percent'],
            }, output, indent=2)
    else:
        if not args.fusion_selection:
            raise ValueError('--fusion_selection is required for test')
        with open(args.fusion_selection, 'r', encoding='utf-8') as selection_file:
            selected = json.load(selection_file)
        if selected.get('selected_on') != 'validation_only':
            raise ValueError('Fusion selection was not frozen on validation')
        selected_map = fuse_probabilities(
            base_streams[STREAM_ROW0], base_streams[STREAM_FULL_DECODER],
            selected['form'], float(selected['weight']),
        )
        data['image_scores'][STREAM_SELECTED_FUSION] = data['image_scores'][STREAM_ROW0]
        report['selected_fusion'] = {
            'form': selected['form'],
            'weight': selected['weight'],
            'selected_on': selected['selected_on'],
            'test_report': _evaluate_streams(
                data, {STREAM_SELECTED_FUSION: selected_map},
                args.output_dir, args.pixel_thresholds,
            )[STREAM_SELECTED_FUSION],
        }

    report_path = os.path.join(args.output_dir, f'{args.phase}_diagnostics.json')
    with open(report_path, 'w', encoding='utf-8') as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
    print(json.dumps({
        'report_path': report_path,
        'reproduction_checks': reproduction,
        'selected_fusion': report.get('selected_fusion'),
    }, indent=2))


def build_parser():
    parser = argparse.ArgumentParser('BridgeAdaptCLIP-v1 zero-cost diagnostics')
    parser.add_argument('--phase', choices=('validation', 'test'), required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--row0_checkpoint', required=True)
    parser.add_argument('--full_checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--fusion_selection')
    parser.add_argument('--expected_row0_p_ap', type=float, required=True)
    parser.add_argument('--expected_full_p_ap', type=float, required=True)
    parser.add_argument('--reproduction_tolerance', type=float, default=0.10)
    parser.add_argument('--pretrained_model', default='ViT-L/14@336px')
    parser.add_argument('--features_list', type=int, nargs='+', default=[6, 12, 18, 24])
    parser.add_argument('--model_input_size', type=int, default=518)
    parser.add_argument('--structural_input_size', type=int, default=1024)
    parser.add_argument('--metric_resolution', type=int, default=1024)
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--vl_reduction', type=int, default=4)
    parser.add_argument('--fusion_channels', type=int, default=128)
    parser.add_argument('--structural_channels', type=int, default=128)
    parser.add_argument('--strip_kernel', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=4.0)
    parser.add_argument('--pixel_thresholds', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--amp', action='store_true')
    return parser


if __name__ == '__main__':
    arguments = build_parser().parse_args()
    setup_seed(arguments.seed)
    run(arguments)
