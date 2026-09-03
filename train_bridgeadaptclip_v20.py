"""Train the v2.0 broad calibration head on a frozen v1.3 fine base."""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import (
    BridgeAdaptCLIPV12, BridgeAdaptCLIPV20, BridgeAdaptCLIPV21Fine,
    TextualAdapter, VisualAdapter,
)
from dataset import BridgeDualResolutionDataset
from tools import get_logger, get_transform, setup_seed
from tools.bridge_row0 import file_sha256, resize_row0_probability, smooth_row0_probability
from tools.bridgeadaptclip_losses import BinaryDiceLossWithLogits, BinaryFocalLossWithLogits
from tools.bridgeadaptclip_v20_losses import (
    broad_gate_and_positive_preservation_losses,
    negative_only_broad_ranking_loss,
)


def _freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def train(args):
    if args.physical_batch_size * args.gradient_accumulation_steps != args.effective_batch_size:
        raise ValueError('physical batch times accumulation must equal effective batch')
    os.makedirs(args.save_path, exist_ok=True)
    logger = get_logger(args.save_path, 'bridge2893_10seed_0shot_bridgeadaptclipv20_train_log.txt')
    logger.info(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clip_model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    clip_model.visual.DAPM_replace(DPAM_layer=20)
    _freeze(clip_model)
    textual = TextualAdapter(clip_model.to('cpu'), args.model_input_size, args.n_ctx)
    visual = VisualAdapter(args.model_input_size, 14, input_dim=768, reduction=args.vl_reduction)
    row0_checkpoint = torch.load(args.row0_checkpoint_path, map_location='cpu')
    textual.load_state_dict(row0_checkpoint['textual_learner'])
    visual.load_state_dict(row0_checkpoint['visual_learner'])
    _freeze(textual)
    _freeze(visual)

    fine_class = (
        BridgeAdaptCLIPV21Fine
        if args.fine_checkpoint_state_key == 'bridgeadaptclip_v21_fine'
        else BridgeAdaptCLIPV12
    )
    fine_model = fine_class(
        semantic_channels=768, fusion_channels=args.fusion_channels,
        structural_channels=args.structural_channels, strip_kernel=args.strip_kernel,
        structural_input_size=args.structural_input_size,
        probability_epsilon=args.probability_epsilon,
    )
    fine_checkpoint = torch.load(args.fine_checkpoint_path, map_location='cpu')
    fine_model.load_state_dict(fine_checkpoint[args.fine_checkpoint_state_key])
    _freeze(fine_model)
    broad_model = BridgeAdaptCLIPV20(
        joint_channels=args.fusion_channels, broad_channels=args.broad_channels,
        output_size=args.structural_input_size,
    ).to(device).train()
    clip_model.to(device); textual.to(device); visual.to(device); fine_model.to(device)
    textual.prepare_static_text_feature(clip_model)
    with torch.no_grad():
        prompts, tokens = textual()
        learned_text = clip_model.encode_text_learn(prompts, tokens).float()

    optimizer = torch.optim.Adam(
        [{'params': broad_model.parameters(), 'lr': args.learning_rate}],
        betas=(0.5, 0.999),
    )
    focal = BinaryFocalLossWithLogits(alpha=args.focal_alpha, gamma=args.focal_gamma)
    dice = BinaryDiceLossWithLogits()
    clip_transform, _ = get_transform(image_size=args.model_input_size)
    dataset = BridgeDualResolutionDataset(
        args.train_data_path, clip_transform=clip_transform,
        structural_input_size=args.structural_input_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.physical_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    if len(loader) % args.gradient_accumulation_steps:
        raise ValueError('loader length must be divisible by accumulation steps')
    amp_enabled = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    row0_sha = file_sha256(args.row0_checkpoint_path)
    fine_sha = file_sha256(args.fine_checkpoint_path)
    logger.info(
        'frozen fine=%d; trainable broad=%d; row0_sha256=%s; fine_sha256=%s',
        sum(p.numel() for p in fine_model.parameters()),
        sum(p.numel() for p in broad_model.parameters()), row0_sha, fine_sha,
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        records = {key: [] for key in (
            'total', 'focal', 'dice', 'broad_gate', 'positive_preserve',
            'broad_rank', 'weighted_gate', 'weighted_preserve', 'weighted_rank',
            'mean_gate', 'mean_magnitude', 'mean_abs_correction',
            'background_correction', 'positive_correction', 'hard_positive',
            'hard_negative', 'ranking_valid_images',
        )}
        optimizer.zero_grad(set_to_none=True)
        for batch_index, items in enumerate(tqdm(loader, desc=f'epoch {epoch}/{args.epochs}')):
            clip_image = items['img'].to(device, non_blocking=True)
            structural = items['structural_img'].to(device, non_blocking=True)
            target = items['native_mask'].to(device, non_blocking=True).unsqueeze(1)
            with torch.no_grad():
                image_features, patch_features = clip_model.encode_image(
                    clip_image, args.features_list, DPAM_layer=20
                )
                _, visual_map, visual_patch = visual.forward_with_features(
                    image_features, patch_features, textual.static_text_features
                )
                _, textual_map = textual.compute_global_local_score(
                    image_features, patch_features, learned_text
                )
                row0_probability = resize_row0_probability(
                    smooth_row0_probability(visual_map, textual_map, sigma=args.sigma),
                    metric_resolution=args.structural_input_size, device=device,
                )
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    if args.fine_checkpoint_state_key == 'bridgeadaptclip_v21_fine':
                        fine_output = fine_model(
                            visual_patch, patch_features, row0_probability, structural
                        )
                    else:
                        fine_output = fine_model(visual_patch, row0_probability, structural)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = broad_model(
                    fine_output['joint_feature'], fine_output['mask_logits'], row0_probability
                )
                focal_loss = focal(output['mask_logits'], target)
                dice_loss = dice(output['mask_logits'], target)
                fp_target, gate_loss, preserve_loss = (
                    broad_gate_and_positive_preservation_losses(
                        output['broad_gate_logits'], output['broad_correction'],
                        target, output['fine_probability'],
                    )
                )
                rank_loss, hard_positive, hard_negative, valid_images = (
                    negative_only_broad_ranking_loss(
                        output['mask_logits'], output['fine_logits'], target,
                        args.hard_positive_count, args.hard_negative_count,
                    )
                )
                weighted_gate = args.broad_gate_loss_weight * gate_loss
                weighted_preserve = args.positive_preserve_loss_weight * preserve_loss
                weighted_rank = args.broad_ranking_loss_weight * rank_loss
                total = focal_loss + dice_loss + weighted_gate + weighted_preserve + weighted_rank
            scaler.scale(total / args.gradient_accumulation_steps).backward()
            if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                global_step += 1

            correction = output['broad_correction'].float()
            positive = target > 0.5
            background = ~positive
            values = {
                'total': total, 'focal': focal_loss, 'dice': dice_loss,
                'broad_gate': gate_loss, 'positive_preserve': preserve_loss,
                'broad_rank': rank_loss, 'weighted_gate': weighted_gate,
                'weighted_preserve': weighted_preserve, 'weighted_rank': weighted_rank,
                'mean_gate': output['broad_gate'].float().mean(),
                'mean_magnitude': output['broad_magnitude'].float().mean(),
                'mean_abs_correction': correction.abs().mean(),
                'background_correction': correction[background].mean(),
                'positive_correction': correction[positive].mean() if positive.any() else correction.sum()*0,
                'hard_positive': hard_positive, 'hard_negative': hard_negative,
            }
            for key, value in values.items(): records[key].append(float(value.detach()))
            records['ranking_valid_images'].append(valid_images)
            if args.max_train_steps and global_step >= args.max_train_steps: break

        summary = {key: float(np.mean(value)) for key, value in records.items()}
        logger.info('epoch [%d/%d] %s', epoch, args.epochs, json.dumps(summary, sort_keys=True))
        checkpoint = {
            'epoch': epoch, 'config': vars(args), 'row0_checkpoint_sha256': row0_sha,
            'fine_checkpoint_sha256': fine_sha, 'fine_checkpoint_epoch': fine_checkpoint.get('epoch'),
            'architecture': {
                'model_name': args.model_name,
                'fine_base': f'frozen {args.fine_checkpoint_state_key}',
                'fusion': 'Z_final = Z_fine - sigmoid(A_b)*softplus(R_b)',
                'broad_feature_size': args.structural_input_size // 8,
                'broad_correction_constraint': 'non-positive',
            },
            args.checkpoint_state_key: broad_model.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.save_path, f'epoch_{epoch}.pth'))
        if args.max_train_steps and global_step >= args.max_train_steps: break

    with open(os.path.join(args.save_path, 'training_metadata.json'), 'w', encoding='utf-8') as handle:
        json.dump({
            'model_name': args.model_name, 'optimizer': 'Adam',
            'optimizer_betas': [0.5, 0.999], 'learning_rate': args.learning_rate,
            'fine_base_frozen': True, 'row0_checkpoint_sha256': row0_sha,
            'fine_checkpoint_sha256': fine_sha, 'loss_weights': {
                'focal': 1.0, 'dice': 1.0,
                'broad_fp_gate': args.broad_gate_loss_weight,
                'positive_preserve': args.positive_preserve_loss_weight,
                'negative_only_ranking': args.broad_ranking_loss_weight,
            },
        }, handle, indent=2)


def build_parser():
    parser = argparse.ArgumentParser('BridgeAdaptCLIP-v2.0 training')
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--row0_checkpoint_path', required=True)
    parser.add_argument('--fine_checkpoint_path', required=True)
    parser.add_argument('--model_name', default='BridgeAdaptCLIP-v2.0')
    parser.add_argument('--checkpoint_state_key', default='bridgeadaptclip_v20')
    parser.add_argument('--fine_checkpoint_state_key', default='bridgeadaptclip_v13')
    parser.add_argument('--pretrained_model', default='ViT-L/14@336px')
    parser.add_argument('--features_list', type=int, nargs='+', default=[6,12,18,24])
    parser.add_argument('--model_input_size', type=int, default=518)
    parser.add_argument('--structural_input_size', type=int, default=1024)
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--vl_reduction', type=int, default=4)
    parser.add_argument('--fusion_channels', type=int, default=128)
    parser.add_argument('--structural_channels', type=int, default=128)
    parser.add_argument('--broad_channels', type=int, default=128)
    parser.add_argument('--strip_kernel', type=int, default=5)
    parser.add_argument('--probability_epsilon', type=float, default=1e-6)
    parser.add_argument('--sigma', type=float, default=4.0)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--physical_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--broad_gate_loss_weight', type=float, default=0.1)
    parser.add_argument('--positive_preserve_loss_weight', type=float, default=0.05)
    parser.add_argument('--broad_ranking_loss_weight', type=float, default=0.01)
    parser.add_argument('--hard_positive_count', type=int, default=256)
    parser.add_argument('--hard_negative_count', type=int, default=256)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--max_train_steps', type=int, default=0)
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args(); setup_seed(args.seed); train(args)
