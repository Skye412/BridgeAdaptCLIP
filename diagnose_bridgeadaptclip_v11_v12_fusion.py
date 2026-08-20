"""Validation-selected offline fusion of BridgeAdaptCLIP v1.1 and v1.2."""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import BridgeAdaptCLIPV11, BridgeAdaptCLIPV12, TextualAdapter, VisualAdapter
from dataset import BridgeDualResolutionDataset
from tools import Evaluator, get_transform, setup_seed
from tools.bridge_class_metrics import evaluate_bridge_classes
from tools.bridge_model_fusion import FUSION_FORMS, fuse_model_logits
from tools.bridge_row0 import resize_row0_probability, row0_image_score, smooth_row0_probability


def _freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


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


def _pixel_metrics(prediction, target, thresholds):
    values = Evaluator._compute_binned_pixel_metrics(
        prediction.reshape(-1), target.reshape(-1), thresholds
    )
    return {
        'P-AUROC': 100.0 * values['auroc'],
        'P-AP': 100.0 * values['ap'],
        'P-F1max': 100.0 * values['f1max'],
    }


def _build_residual_model(model_class, checkpoint_path, state_key, args):
    model = model_class(
        semantic_channels=768,
        fusion_channels=args.fusion_channels,
        structural_channels=args.structural_channels,
        strip_kernel=args.strip_kernel,
        structural_input_size=args.structural_input_size,
        probability_epsilon=args.probability_epsilon,
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint[state_key])
    _freeze(model)
    return model


def collect(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    clip_model.visual.DAPM_replace(DPAM_layer=20)
    _freeze(clip_model)

    textual = TextualAdapter(clip_model.to('cpu'), args.model_input_size, args.n_ctx)
    visual = VisualAdapter(
        args.model_input_size, 14, input_dim=768, reduction=args.vl_reduction
    )
    row0 = torch.load(args.row0_checkpoint, map_location='cpu')
    textual.load_state_dict(row0['textual_learner'])
    visual.load_state_dict(row0['visual_learner'])
    _freeze(textual)
    _freeze(visual)

    v11 = _build_residual_model(
        BridgeAdaptCLIPV11, args.v11_checkpoint, 'bridgeadaptclip_v11', args
    )
    v12 = _build_residual_model(
        BridgeAdaptCLIPV12, args.v12_checkpoint, 'bridgeadaptclip_v12', args
    )
    for module in (clip_model, textual, visual, v11, v12):
        module.to(device).eval()
    textual.prepare_static_text_feature(clip_model)
    with torch.no_grad():
        prompts, tokens = textual()
        learned_text = clip_model.encode_text_learn(prompts, tokens).float()

    clip_transform, _ = get_transform(image_size=args.model_input_size)
    dataset = BridgeDualResolutionDataset(
        args.data_path,
        clip_transform=clip_transform,
        structural_input_size=args.structural_input_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    records = {key: [] for key in (
        'v11_logits', 'v12_logits', 'targets', 'image_targets', 'image_scores',
        'paths', 'sample_ids', 'cls_names',
    )}
    amp_enabled = args.amp and device.type == 'cuda'
    for items in tqdm(loader, desc=f'{args.phase}-v11-v12-fusion'):
        clip_image = items['img'].to(device, non_blocking=True)
        structural_image = items['structural_img'].to(device, non_blocking=True)
        with torch.no_grad():
            image_features, patch_features = clip_model.encode_image(
                clip_image, args.features_list, DPAM_layer=20
            )
            global_visual, visual_map, visual_patch = visual.forward_with_features(
                image_features, patch_features, textual.static_text_features
            )
            global_textual, textual_map = textual.compute_global_local_score(
                image_features, patch_features, learned_text
            )
            smoothed_row0 = smooth_row0_probability(
                visual_map, textual_map, sigma=args.sigma
            )
            row0_probability = resize_row0_probability(
                smoothed_row0, metric_resolution=args.metric_resolution, device=device
            )
            image_score = row0_image_score(
                global_visual, global_textual, smoothed_row0
            )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            out_v11 = v11(visual_patch, row0_probability, structural_image)
            out_v12 = v12(visual_patch, row0_probability, structural_image)
        # Half precision storage bounds host memory; metrics/fusion are computed in FP32.
        records['v11_logits'].append(out_v11['mask_logits'][:, 0].half().cpu())
        records['v12_logits'].append(out_v12['mask_logits'][:, 0].half().cpu())
        records['targets'].append(items['native_mask'].to(torch.uint8))
        records['image_targets'].append(items['anomaly'].to(torch.uint8))
        records['image_scores'].append(image_score.float().cpu())
        records['paths'].extend(items['img_path'])
        records['sample_ids'].extend(items['sample_id'])
        records['cls_names'].extend(items['cls_name'])

    return {
        'v11_logits': torch.cat(records['v11_logits']),
        'v12_logits': torch.cat(records['v12_logits']),
        'targets': torch.cat(records['targets']),
        'image_targets': torch.cat(records['image_targets']),
        'image_scores': torch.cat(records['image_scores']),
        'paths': np.asarray(records['paths']),
        'sample_ids': np.asarray(records['sample_ids']),
        'cls_names': np.asarray(records['cls_names']),
    }


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    data = collect(args)
    image_metrics = _image_metrics(data['image_targets'], data['image_scores'])
    protocol = {
        'phase': args.phase,
        'selection_split': 'validation',
        'selection_metric': 'native-1024 overall P-AP',
        'forms': list(FUSION_FORMS),
        'weights_are_v12': True,
        'weights': [step / 10.0 for step in range(11)],
        'test_tuning_forbidden': True,
        'v11_checkpoint': args.v11_checkpoint,
        'v12_checkpoint': args.v12_checkpoint,
    }
    if args.phase == 'validation':
        candidates = []
        for form in FUSION_FORMS:
            for step in range(11):
                weight = step / 10.0
                prediction = fuse_model_logits(
                    data['v11_logits'], data['v12_logits'], form, weight
                )
                metrics = _pixel_metrics(
                    prediction, data['targets'], args.pixel_thresholds
                )
                candidates.append({
                    'form': form,
                    'v12_weight': weight,
                    'metrics_percent': metrics,
                })
                del prediction
        selected = max(
            enumerate(candidates),
            key=lambda item: (
                item[1]['metrics_percent']['P-AP'],
                item[1]['metrics_percent']['P-AUROC'],
                -item[0],
            ),
        )[1]
        selection = {
            'selected_on': 'validation_only',
            'selection_metric': 'native-1024 overall P-AP',
            **selected,
        }
        with open(os.path.join(args.output_dir, 'fusion_selection.json'), 'w', encoding='utf-8') as output:
            json.dump(selection, output, indent=2)
        report = {
            'protocol': protocol,
            'image_metrics_percent': image_metrics,
            'candidates': candidates,
            'selected_fusion': selection,
        }
    else:
        if not args.fusion_selection:
            raise ValueError('--fusion_selection is required for formal Test')
        with open(args.fusion_selection, 'r', encoding='utf-8') as source:
            selection = json.load(source)
        if selection.get('selected_on') != 'validation_only':
            raise ValueError('Fusion was not frozen on Validation')
        prediction = fuse_model_logits(
            data['v11_logits'], data['v12_logits'],
            selection['form'], float(selection['v12_weight']),
        )
        pixel_metrics = _pixel_metrics(
            prediction, data['targets'], args.pixel_thresholds
        )
        defect_metrics = evaluate_bridge_classes(
            data['paths'], data['image_scores'], prediction,
            args.pixel_thresholds,
            os.path.join(args.output_dir, 'bridge_defect_metrics.json'),
        )
        macro = float(np.mean([
            values['metrics_percent']['P-AP'] for values in defect_metrics.values()
        ]))
        report = {
            'protocol': protocol,
            'frozen_selection': selection,
            'results_percent': {**image_metrics, **pixel_metrics},
            'macro_diagnostic_P-AP': macro,
            'defect_metrics': defect_metrics,
        }
    report_path = os.path.join(args.output_dir, f'{args.phase}_fusion.json')
    with open(report_path, 'w', encoding='utf-8') as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser('Validation-selected v1.1/v1.2 fusion')
    parser.add_argument('--phase', choices=('validation', 'test'), required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--row0_checkpoint', required=True)
    parser.add_argument('--v11_checkpoint', required=True)
    parser.add_argument('--v12_checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--fusion_selection')
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
    parser.add_argument('--probability_epsilon', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=4.0)
    parser.add_argument('--pixel_thresholds', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--amp', action='store_true')
    return parser


if __name__ == '__main__':
    parsed = build_parser().parse_args()
    setup_seed(parsed.seed)
    run(parsed)
