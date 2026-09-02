"""Audit v1.5 hard-positive composition on a non-Test split."""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import BridgeAdaptCLIPV12, TextualAdapter, VisualAdapter
from dataset import BridgeDualResolutionDataset
from tools import get_transform, setup_seed
from tools.bridge_masks import DEFECT_NAMES, decode_bridge_class_masks
from tools.bridge_row0 import resize_row0_probability, smooth_row0_probability


def _freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def audit(args):
    if os.path.basename(os.path.normpath(args.data_path)) == 'test':
        raise ValueError('This development audit must not run on Test.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=20)
    _freeze(model)

    textual = TextualAdapter(model.to('cpu'), args.model_input_size, args.n_ctx)
    visual = VisualAdapter(args.model_input_size, 14, input_dim=768, reduction=4)
    row0 = torch.load(args.row0_checkpoint_path, map_location='cpu')
    textual.load_state_dict(row0['textual_learner'])
    visual.load_state_dict(row0['visual_learner'])
    _freeze(textual)
    _freeze(visual)

    bridge = BridgeAdaptCLIPV12(
        semantic_channels=768, fusion_channels=128, structural_channels=128,
        strip_kernel=5, structural_input_size=1024, probability_epsilon=1e-6,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    bridge.load_state_dict(checkpoint[args.checkpoint_state_key])
    model.to(device)
    textual.to(device)
    visual.to(device)
    bridge.to(device).eval()
    textual.prepare_static_text_feature(model)
    with torch.no_grad():
        prompts, tokenized = textual()
        learned_text = model.encode_text_learn(prompts, tokenized).float()

    transform, _ = get_transform(image_size=args.model_input_size)
    dataset = BridgeDualResolutionDataset(
        args.data_path, transform, structural_input_size=1024
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    class_stats = {
        name: {
            'available_pixels': 0,
            'selected_pixels': 0,
            'selected_logit_sum': 0.0,
            'images_present': 0,
            'images_with_selected_pixel': 0,
        }
        for name in DEFECT_NAMES
    }
    defect_images = 0
    total_selected = 0
    crack_images = 0
    crack_images_hit = 0
    mixed_crack_images = 0
    mixed_crack_selected = 0
    mixed_total_selected = 0
    amp_enabled = args.amp and device.type == 'cuda'

    for items in tqdm(loader, desc='audit v1.5 hard positives'):
        clip_image = items['img'].to(device, non_blocking=True)
        structural = items['structural_img'].to(device, non_blocking=True)
        with torch.no_grad():
            image_features, patch_features = model.encode_image(
                clip_image, [6, 12, 18, 24], DPAM_layer=20
            )
            _, visual_map, visual_patch = visual.forward_with_features(
                image_features, patch_features, textual.static_text_features
            )
            _, textual_map = textual.compute_global_local_score(
                image_features, patch_features, learned_text
            )
            row0_probability = resize_row0_probability(
                smooth_row0_probability(visual_map, textual_map, sigma=4.0),
                metric_resolution=1024, device=device,
            )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = bridge(visual_patch, row0_probability, structural)['mask_logits']
        logits = logits[:, 0].float().cpu().numpy()

        for batch_index, image_path in enumerate(items['img_path']):
            if int(items['anomaly'][batch_index]) == 0:
                continue
            class_masks, binary_mask = decode_bridge_class_masks(image_path)
            flat_positive = np.flatnonzero(binary_mask.reshape(-1))
            count = min(args.hard_positive_count, flat_positive.size)
            flat_logits = logits[batch_index].reshape(-1)
            local = np.argpartition(flat_logits[flat_positive], count - 1)[:count]
            selected = flat_positive[local]
            defect_images += 1
            total_selected += count

            present_names = [name for name in DEFECT_NAMES if class_masks[name].any()]
            selected_crack_count = 0
            for name in DEFECT_NAMES:
                flat_class = class_masks[name].reshape(-1)
                available = int(flat_class.sum())
                selected_for_class = selected[flat_class[selected]]
                selected_count = int(selected_for_class.size)
                stats = class_stats[name]
                stats['available_pixels'] += available
                stats['selected_pixels'] += selected_count
                if available:
                    stats['images_present'] += 1
                if selected_count:
                    stats['images_with_selected_pixel'] += 1
                    stats['selected_logit_sum'] += float(
                        flat_logits[selected_for_class].sum()
                    )
                if name == 'Crack':
                    selected_crack_count = selected_count

            if 'Crack' in present_names:
                crack_images += 1
                crack_images_hit += int(selected_crack_count > 0)
                if len(present_names) > 1:
                    mixed_crack_images += 1
                    mixed_crack_selected += selected_crack_count
                    mixed_total_selected += count

    for stats in class_stats.values():
        selected = stats['selected_pixels']
        stats['selected_fraction'] = selected / max(total_selected, 1)
        stats['available_fraction'] = stats['available_pixels'] / max(
            sum(s['available_pixels'] for s in class_stats.values()), 1
        )
        stats['mean_selected_logit'] = (
            stats.pop('selected_logit_sum') / selected if selected else None
        )
        stats['image_hit_rate'] = (
            stats['images_with_selected_pixel'] / stats['images_present']
            if stats['images_present'] else None
        )

    report = {
        'protocol': {
            'split': os.path.basename(os.path.normpath(args.data_path)),
            'checkpoint': os.path.abspath(args.checkpoint_path),
            'hard_positive_count_per_image': args.hard_positive_count,
            'test_forbidden': True,
        },
        'defect_images': defect_images,
        'total_selected_positive_pixels': total_selected,
        'class_composition': class_stats,
        'crack_image_coverage': {
            'crack_images': crack_images,
            'crack_images_with_selected_crack': crack_images_hit,
            'coverage': crack_images_hit / max(crack_images, 1),
        },
        'mixed_crack_images': {
            'image_count': mixed_crack_images,
            'selected_crack_pixels': mixed_crack_selected,
            'total_selected_slots': mixed_total_selected,
            'crack_slot_fraction': mixed_crack_selected / max(mixed_total_selected, 1),
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))


def build_parser():
    parser = argparse.ArgumentParser('Audit v1.5 hard-positive composition')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--row0_checkpoint_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint_state_key', default='bridgeadaptclip_v15')
    parser.add_argument('--pretrained_model', default='ViT-L/14@336px')
    parser.add_argument('--model_input_size', type=int, default=518)
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--hard_positive_count', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--amp', action='store_true')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    setup_seed(args.seed)
    audit(args)
