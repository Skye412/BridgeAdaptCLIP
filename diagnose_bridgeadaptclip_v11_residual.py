"""Zero-training spatial diagnosis of BridgeAdaptCLIP-v1.1 Epoch 2."""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import BridgeAdaptCLIPV11, BridgeAdaptCLIPV12, TextualAdapter, VisualAdapter
from dataset import BridgeDualResolutionDataset
from tools import get_transform, setup_seed
from tools.bridge_masks import decode_bridge_class_masks
from tools.bridge_residual_diagnostics import PearsonAccumulator, RegionAccumulator
from tools.bridge_row0 import file_sha256, resize_row0_probability, smooth_row0_probability


def _freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def _crack_masks(paths, shape):
    masks = []
    for path in paths:
        if os.path.basename(os.path.dirname(path)) == 'normal':
            masks.append(torch.zeros(shape, dtype=torch.bool))
        else:
            class_masks, _ = decode_bridge_class_masks(path)
            masks.append(torch.from_numpy(class_masks['Crack'].copy()))
    return torch.stack(masks)


def _sample_maps(sample_lists, maps, samples_per_image, generator):
    batch, _, height, width = maps['gate'].shape
    population = height * width
    for image_index in range(batch):
        indices = torch.randint(
            population, (samples_per_image,), generator=generator
        ).to(maps['gate'].device)
        for name, tensor in maps.items():
            values = tensor[image_index, 0].reshape(-1)[indices]
            sample_lists[name].append(values.float().cpu())


def diagnose(args):
    if args.metric_resolution != args.structural_input_size:
        raise ValueError('metric_resolution must equal structural_input_size.')
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=20)
    _freeze(model)
    textual = TextualAdapter(model.to('cpu'), args.model_input_size, args.n_ctx)
    visual = VisualAdapter(
        args.model_input_size, 14, input_dim=768, reduction=args.vl_reduction
    )
    row0_checkpoint = torch.load(args.row0_checkpoint_path, map_location='cpu')
    textual.load_state_dict(row0_checkpoint['textual_learner'])
    visual.load_state_dict(row0_checkpoint['visual_learner'])
    _freeze(textual)
    _freeze(visual)

    bridge_class = (
        BridgeAdaptCLIPV12
        if args.checkpoint_state_key in (
            'bridgeadaptclip_v12', 'bridgeadaptclip_v13', 'bridgeadaptclip_v14',
            'bridgeadaptclip_v15', 'bridgeadaptclip_v16', 'bridgeadaptclip_v17'
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
    actual_row0_sha = file_sha256(args.row0_checkpoint_path)
    if checkpoint.get('row0_checkpoint_sha256') != actual_row0_sha:
        raise ValueError('Row-0 checkpoint hash mismatch.')

    model.to(device)
    textual.to(device)
    visual.to(device)
    bridge_model.to(device).eval()
    textual.prepare_static_text_feature(model)
    with torch.no_grad():
        learned_prompts, tokenized_prompts = textual()
        learned_text_features = model.encode_text_learn(
            learned_prompts, tokenized_prompts
        ).float()

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

    regions = RegionAccumulator()
    gate_error_corr = PearsonAccumulator()
    correction_error_corr = PearsonAccumulator()
    sample_lists = {
        name: [] for name in ['gate', 'residual', 'correction', 'abs_correction']
    }
    generator = torch.Generator(device='cpu').manual_seed(args.sampling_seed)
    amp_enabled = args.amp and device.type == 'cuda'
    mechanism_counts = {
        'fn_total': 0,
        'fn_recovered': 0,
        'fn_positive_correction': 0,
        'fn_final_logit_ge_margin': 0,
        'fp_total': 0,
        'fp_suppressed': 0,
        'fp_negative_correction': 0,
        'fp_final_logit_le_negative_margin': 0,
    }

    for items in tqdm(loader, desc=f'diagnose {args.split_name}'):
        clip_image = items['img'].to(device, non_blocking=True)
        structural_image = items['structural_img'].to(device, non_blocking=True)
        target = items['native_mask'].to(device, non_blocking=True).unsqueeze(1).float()
        with torch.no_grad():
            image_features, patch_features = model.encode_image(
                clip_image, args.features_list, DPAM_layer=20
            )
            _, visual_map, visual_patch_feature = visual.forward_with_features(
                image_features, patch_features, textual.static_text_features
            )
            _, textual_map = textual.compute_global_local_score(
                image_features, patch_features, learned_text_features
            )
            smoothed_row0 = smooth_row0_probability(
                visual_map, textual_map, sigma=args.sigma
            )
            row0_probability = resize_row0_probability(
                smoothed_row0, args.metric_resolution, device=device
            )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            output = bridge_model(
                visual_patch_feature, row0_probability, structural_image
            )

        gate = output['gate'].float()
        residual = output['residual'].float()
        correction = output['gated_residual'].float()
        final_logits = output['mask_logits'].float()
        abs_correction = correction.abs()
        error = (target - row0_probability.float()).abs()
        row0_error = error >= 0.5
        false_positive_like = (~target.bool()) & (row0_probability >= 0.5)
        false_negative_like = target.bool() & (row0_probability < 0.5)
        true_negative_like = (~target.bool()) & (row0_probability < 0.5)
        true_positive_like = target.bool() & (row0_probability >= 0.5)
        mechanism_counts['fn_total'] += int(false_negative_like.sum())
        mechanism_counts['fn_recovered'] += int(
            (false_negative_like & (final_logits >= 0.0)).sum()
        )
        mechanism_counts['fn_positive_correction'] += int(
            (false_negative_like & (correction > 0.0)).sum()
        )
        mechanism_counts['fn_final_logit_ge_margin'] += int(
            (false_negative_like & (final_logits >= args.margin)).sum()
        )
        mechanism_counts['fp_total'] += int(false_positive_like.sum())
        mechanism_counts['fp_suppressed'] += int(
            (false_positive_like & (final_logits < 0.0)).sum()
        )
        mechanism_counts['fp_negative_correction'] += int(
            (false_positive_like & (correction < 0.0)).sum()
        )
        mechanism_counts['fp_final_logit_le_negative_margin'] += int(
            (false_positive_like & (final_logits <= -args.margin)).sum()
        )
        normal_images = (items['anomaly'].to(device) == 0)[:, None, None, None]
        normal_pixels = normal_images.expand_as(target)
        defect_image_pixels = ~normal_pixels
        crack = _crack_masks(
            list(items['img_path']), target.shape[-2:]
        ).to(device).unsqueeze(1)

        all_pixels = torch.ones_like(target, dtype=torch.bool)
        region_masks = {
            'all_pixels': all_pixels,
            'gt_positive': target.bool(),
            'gt_background': ~target.bool(),
            'row0_error_E_ge_0.5': row0_error,
            'row0_correct_E_lt_0.5': ~row0_error,
            'false_positive_like_Y0_P0_ge_0.5': false_positive_like,
            'false_negative_like_Y1_P0_lt_0.5': false_negative_like,
            'true_negative_like_Y0_P0_lt_0.5': true_negative_like,
            'true_positive_like_Y1_P0_ge_0.5': true_positive_like,
            'crack_pixels': crack,
            'normal_images_all_pixels': normal_pixels,
            'defect_images_all_pixels': defect_image_pixels,
        }
        for name, mask in region_masks.items():
            regions.update(name, mask, gate, residual, correction, error)

        gate_error_corr.update(gate, error)
        correction_error_corr.update(abs_correction, error)
        _sample_maps(
            sample_lists,
            {
                'gate': gate,
                'residual': residual,
                'correction': correction,
                'abs_correction': abs_correction,
            },
            args.samples_per_image,
            generator,
        )

    distribution = {}
    for name, chunks in sample_lists.items():
        values = torch.cat(chunks).numpy()
        distribution[name] = {
            'sampling': 'deterministic_uniform_with_replacement_per_image',
            'sample_count': int(values.size),
            'mean_sampled': float(values.mean()),
            'median_approx': float(np.quantile(values, 0.5)),
            'p90_approx': float(np.quantile(values, 0.9)),
            'p99_approx': float(np.quantile(values, 0.99)),
            'min_sampled': float(values.min()),
            'max_sampled': float(values.max()),
        }

    region_report = regions.finalize()
    error_region = region_report.get('row0_error_E_ge_0.5')
    correct_region = region_report.get('row0_correct_E_lt_0.5')
    fn_total = max(mechanism_counts['fn_total'], 1)
    fp_total = max(mechanism_counts['fp_total'], 1)
    report = {
        'protocol': {
            'diagnostic': (
                f'{args.checkpoint_state_key} gated residual spatial statistics'
            ),
            'training_performed': False,
            'split': args.split_name,
            'decision_split': args.decision_split,
            'checkpoint_path': args.checkpoint_path,
            'checkpoint_epoch': checkpoint['epoch'],
            'checkpoint_state_key': args.checkpoint_state_key,
            'row0_checkpoint_path': args.row0_checkpoint_path,
            'row0_checkpoint_sha256': actual_row0_sha,
            'row0_error': 'E = abs(Y - P0)',
            'row0_error_region': 'E >= 0.5',
            'quantile_sampling_seed': args.sampling_seed,
            'samples_per_image': args.samples_per_image,
        },
        'all_pixel_distributions': distribution,
        'regions': region_report,
        'correlations_all_pixels': {
            'pearson_gate_vs_row0_error': gate_error_corr.finalize(),
            'pearson_abs_correction_vs_row0_error': correction_error_corr.finalize(),
        },
        'region_contrast': {
            'gate_error_over_correct_ratio': (
                error_region['mean_gate'] / correct_region['mean_gate']
                if error_region and correct_region else None
            ),
            'abs_correction_error_over_correct_ratio': (
                error_region['mean_abs_correction'] / correct_region['mean_abs_correction']
                if error_region and correct_region else None
            ),
        },
        'margin_mechanism': {
            'margin': args.margin,
            'fn_like_pixel_count': mechanism_counts['fn_total'],
            'fp_like_pixel_count': mechanism_counts['fp_total'],
            'fn_recovery_rate': mechanism_counts['fn_recovered'] / fn_total,
            'fp_suppression_rate': mechanism_counts['fp_suppressed'] / fp_total,
            'fn_like_positive_correction_ratio': (
                mechanism_counts['fn_positive_correction'] / fn_total
            ),
            'fp_like_negative_correction_ratio': (
                mechanism_counts['fp_negative_correction'] / fp_total
            ),
            'fn_like_final_logit_ge_margin_ratio': (
                mechanism_counts['fn_final_logit_ge_margin'] / fn_total
            ),
            'fp_like_final_logit_le_negative_margin_ratio': (
                mechanism_counts['fp_final_logit_le_negative_margin'] / fp_total
            ),
            'fn_like_mean_correction': (
                region_report['false_negative_like_Y1_P0_lt_0.5']['mean_correction']
            ),
            'fp_like_mean_correction': (
                region_report['false_positive_like_Y0_P0_ge_0.5']['mean_correction']
            ),
        },
    }
    output_path = os.path.join(args.save_path, f'{args.split_name}_residual_diagnostics.json')
    with open(output_path, 'w', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))


def build_parser():
    parser = argparse.ArgumentParser('BridgeAdaptCLIP-v1.1 residual diagnosis')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--split_name', required=True, choices=['val', 'test'])
    parser.add_argument('--decision_split', default='val')
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--checkpoint_state_key', default='bridgeadaptclip_v11')
    parser.add_argument('--row0_checkpoint_path', required=True)
    parser.add_argument('--save_path', required=True)
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
    parser.add_argument('--sigma', type=float, default=4.0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--samples_per_image', type=int, default=4096)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--sampling_seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--amp', action='store_true')
    return parser


if __name__ == '__main__':
    parsed_args = build_parser().parse_args()
    setup_seed(parsed_args.seed)
    diagnose(parsed_args)
