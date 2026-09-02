"""Train BridgeAdaptCLIP-v1.1 on a frozen Protocol-v2 Row-0 semantic base."""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import BridgeAdaptCLIPV11, BridgeAdaptCLIPV12, TextualAdapter, VisualAdapter
from dataset import BridgeDualResolutionDataset
from tools import get_logger, get_transform, setup_seed
from tools.bridge_row0 import file_sha256, resize_row0_probability, smooth_row0_probability
from tools.bridgeadaptclip_losses import BinaryDiceLossWithLogits, BinaryFocalLossWithLogits
from tools.bridgeadaptclip_v12_losses import error_aware_gate_losses
from tools.bridgeadaptclip_v13_losses import signed_error_correction_loss
from tools.bridgeadaptclip_v14_losses import final_logit_margin_loss
from tools.bridgeadaptclip_v15_losses import hard_pixel_ranking_loss
from tools.bridgeadaptclip_v16_losses import skeleton_balanced_hard_pixel_ranking_loss


def _freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def train(args):
    if args.physical_batch_size * args.gradient_accumulation_steps != args.effective_batch_size:
        raise ValueError(
            'physical_batch_size * gradient_accumulation_steps must equal effective_batch_size'
        )
    os.makedirs(args.save_path, exist_ok=True)
    model_slug = args.model_name.lower().replace('-', '').replace('.', '')
    logger = get_logger(
        args.save_path,
        f'bridge2893_{args.seed}seed_0shot_{model_slug}_train_log.txt',
    )
    logger.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=20)
    _freeze(model)

    # No bridge anchors: construct the exact Original AdaptCLIP prompt ensemble.
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
    ).to(device).train()
    model.to(device)
    textual_learner.to(device)
    visual_learner.to(device)
    textual_learner.prepare_static_text_feature(model)

    with torch.no_grad():
        learned_prompts, tokenized_prompts = textual_learner()
        learned_text_features = model.encode_text_learn(
            learned_prompts, tokenized_prompts
        ).float()

    trainable_parameters = list(bridge_model.parameters())
    optimizer = torch.optim.Adam(
        [{'params': trainable_parameters, 'lr': args.new_module_learning_rate}],
        betas=(0.5, 0.999),
    )
    focal_loss = BinaryFocalLossWithLogits(
        alpha=args.focal_alpha, gamma=args.focal_gamma
    )
    dice_loss = BinaryDiceLossWithLogits()

    clip_transform, _ = get_transform(image_size=args.model_input_size)
    train_data = BridgeDualResolutionDataset(
        args.train_data_path,
        clip_transform=clip_transform,
        structural_input_size=args.structural_input_size,
        return_native_skeleton=args.skeleton_balanced_ranking,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.physical_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if len(train_loader) % args.gradient_accumulation_steps:
        raise ValueError(
            'Training loader length must be divisible by gradient_accumulation_steps.'
        )

    amp_enabled = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    row0_sha256 = file_sha256(args.row0_checkpoint_path)
    logger.info(
        'frozen_params clip=%d visual=%d textual=%d; trainable_v11=%d; amp=%s; '
        'row0_sha256=%s',
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in visual_learner.parameters()),
        sum(p.numel() for p in textual_learner.parameters()),
        sum(p.numel() for p in trainable_parameters),
        amp_enabled,
        row0_sha256,
    )

    global_step = 0
    completed_epoch = 0
    for epoch in range(1, args.epochs + 1):
        loss_records = {
            'total': [], 'pixel_focal': [], 'pixel_dice': [],
            'gate_loss': [], 'preserve_loss': [],
            'weighted_gate_loss': [], 'weighted_preserve_loss': [],
            'signed_correction_loss': [], 'signed_positive_loss': [],
            'signed_negative_loss': [], 'weighted_signed_correction_loss': [],
            'margin_loss': [], 'fn_margin_loss': [], 'fp_margin_loss': [],
            'weighted_margin_loss': [],
            'ranking_loss': [], 'weighted_ranking_loss': [],
            'global_ranking_loss': [], 'thin_ranking_loss': [],
            'hard_positive_logit': [], 'hard_negative_logit': [],
            'thin_positive_logit': [], 'selected_skeleton_count': [],
            'ranking_valid_images': [],
            'residual_l1': [], 'mean_gate': [],
            'mean_abs_residual': [], 'mean_abs_correction': [],
        }
        optimizer.zero_grad(set_to_none=True)

        for batch_index, items in enumerate(tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}')):
            clip_image = items['img'].to(device, non_blocking=True)
            structural_image = items['structural_img'].to(device, non_blocking=True)
            target_mask = items['native_mask'].to(device, non_blocking=True).unsqueeze(1)
            skeleton_mask = None
            if args.skeleton_balanced_ranking:
                skeleton_mask = items['native_skeleton'].to(
                    device, non_blocking=True
                ).unsqueeze(1)

            # Keep the frozen semantic base on the exact Row-0 inference path.
            # AMP is applied only to the newly trained structural module.
            with torch.no_grad():
                image_features, patch_features = model.encode_image(
                    clip_image, args.features_list, DPAM_layer=20
                )
                _, visual_map, visual_patch_feature = visual_learner.forward_with_features(
                    image_features, patch_features, textual_learner.static_text_features
                )
                _, textual_map = textual_learner.compute_global_local_score(
                    image_features, patch_features, learned_text_features
                )
                smoothed_row0 = smooth_row0_probability(
                    visual_map, textual_map, sigma=args.sigma
                )
                row0_probability = resize_row0_probability(
                    smoothed_row0,
                    metric_resolution=args.structural_input_size,
                    device=device,
                )

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = bridge_model(
                    visual_patch_feature.detach(), row0_probability, structural_image
                )
                pixel_focal = focal_loss(output['mask_logits'], target_mask)
                pixel_dice = dice_loss(output['mask_logits'], target_mask)
                gate_target, gate_loss, preserve_loss = error_aware_gate_losses(
                    output['gate_logits'], output['gated_residual'],
                    target_mask, row0_probability,
                )
                signed_loss, signed_positive, signed_negative = (
                    signed_error_correction_loss(
                        output['gated_residual'], target_mask, gate_target
                    )
                )
                margin_loss, fn_margin_loss, fp_margin_loss = (
                    final_logit_margin_loss(
                        output['mask_logits'], target_mask, gate_target,
                        margin=args.margin,
                    )
                )
                if args.skeleton_balanced_ranking:
                    (
                        ranking_loss,
                        global_ranking_loss,
                        thin_ranking_loss,
                        hard_positive_logit,
                        thin_positive_logit,
                        hard_negative_logit,
                        selected_skeleton_count,
                        ranking_valid_images,
                    ) = skeleton_balanced_hard_pixel_ranking_loss(
                        output['mask_logits'], target_mask, skeleton_mask,
                        global_positive_count=args.global_positive_count,
                        skeleton_positive_count=args.skeleton_positive_count,
                        hard_negative_count=args.hard_negative_count,
                    )
                else:
                    (
                        ranking_loss,
                        hard_positive_logit,
                        hard_negative_logit,
                        ranking_valid_images,
                    ) = hard_pixel_ranking_loss(
                        output['mask_logits'],
                        target_mask,
                        hard_positive_count=args.hard_positive_count,
                        hard_negative_count=args.hard_negative_count,
                    )
                    global_ranking_loss = ranking_loss
                    thin_ranking_loss = ranking_loss
                    thin_positive_logit = hard_positive_logit
                    selected_skeleton_count = ranking_loss.detach() * 0.0
                weighted_gate_loss = args.gate_loss_weight * gate_loss
                weighted_preserve_loss = args.preserve_loss_weight * preserve_loss
                weighted_signed_loss = args.signed_correction_loss_weight * signed_loss
                weighted_margin_loss = args.margin_loss_weight * margin_loss
                weighted_ranking_loss = args.ranking_loss_weight * ranking_loss
                residual_l1 = output['gated_residual'].float().abs().mean()
                total_loss = (
                    pixel_focal + pixel_dice
                    + args.residual_l1_weight * residual_l1
                    + weighted_gate_loss + weighted_preserve_loss
                    + weighted_signed_loss
                    + weighted_margin_loss
                    + weighted_ranking_loss
                )

            scaler.scale(total_loss / args.gradient_accumulation_steps).backward()
            if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            loss_records['total'].append(float(total_loss.detach()))
            loss_records['pixel_focal'].append(float(pixel_focal.detach()))
            loss_records['pixel_dice'].append(float(pixel_dice.detach()))
            loss_records['gate_loss'].append(float(gate_loss.detach()))
            loss_records['preserve_loss'].append(float(preserve_loss.detach()))
            loss_records['weighted_gate_loss'].append(float(weighted_gate_loss.detach()))
            loss_records['weighted_preserve_loss'].append(
                float(weighted_preserve_loss.detach())
            )
            loss_records['signed_correction_loss'].append(float(signed_loss.detach()))
            loss_records['signed_positive_loss'].append(float(signed_positive.detach()))
            loss_records['signed_negative_loss'].append(float(signed_negative.detach()))
            loss_records['weighted_signed_correction_loss'].append(
                float(weighted_signed_loss.detach())
            )
            loss_records['margin_loss'].append(float(margin_loss.detach()))
            loss_records['fn_margin_loss'].append(float(fn_margin_loss.detach()))
            loss_records['fp_margin_loss'].append(float(fp_margin_loss.detach()))
            loss_records['weighted_margin_loss'].append(
                float(weighted_margin_loss.detach())
            )
            loss_records['ranking_loss'].append(float(ranking_loss.detach()))
            loss_records['global_ranking_loss'].append(
                float(global_ranking_loss.detach())
            )
            loss_records['thin_ranking_loss'].append(
                float(thin_ranking_loss.detach())
            )
            loss_records['weighted_ranking_loss'].append(
                float(weighted_ranking_loss.detach())
            )
            loss_records['hard_positive_logit'].append(
                float(hard_positive_logit.detach())
            )
            loss_records['hard_negative_logit'].append(
                float(hard_negative_logit.detach())
            )
            loss_records['thin_positive_logit'].append(
                float(thin_positive_logit.detach())
            )
            loss_records['selected_skeleton_count'].append(
                float(selected_skeleton_count.detach())
            )
            loss_records['ranking_valid_images'].append(ranking_valid_images)
            loss_records['residual_l1'].append(float(residual_l1.detach()))
            loss_records['mean_gate'].append(float(output['gate'].float().mean().detach()))
            loss_records['mean_abs_residual'].append(
                float(output['residual'].float().abs().mean().detach())
            )
            loss_records['mean_abs_correction'].append(
                float(output['gated_residual'].float().abs().mean().detach())
            )
            if args.max_train_steps and global_step >= args.max_train_steps:
                break

        completed_epoch = epoch
        logger.info(
            'epoch [%d/%d], total=%.6f, pixel_focal=%.6f, pixel_dice=%.6f, '
            'gate_loss=%.6f, preserve_loss=%.6f, weighted_gate_loss=%.6f, '
            'weighted_preserve_loss=%.6f, signed_loss=%.6f, '
            'signed_positive=%.6f, signed_negative=%.6f, '
            'weighted_signed_loss=%.6f, margin_loss=%.6f, '
            'fn_margin_loss=%.6f, fp_margin_loss=%.6f, '
            'weighted_margin_loss=%.6f, mean_gate=%.6f, '
            'ranking_loss=%.6f, weighted_ranking_loss=%.6f, '
            'global_ranking_loss=%.6f, thin_ranking_loss=%.6f, '
            'hard_positive_logit=%.6f, thin_positive_logit=%.6f, '
            'hard_negative_logit=%.6f, selected_skeleton_count=%.3f, '
            'ranking_valid_images=%.3f, mean_abs_residual=%.6f, '
            'mean_abs_correction=%.6f',
            epoch, args.epochs,
            np.mean(loss_records['total']),
            np.mean(loss_records['pixel_focal']),
            np.mean(loss_records['pixel_dice']),
            np.mean(loss_records['gate_loss']),
            np.mean(loss_records['preserve_loss']),
            np.mean(loss_records['weighted_gate_loss']),
            np.mean(loss_records['weighted_preserve_loss']),
            np.mean(loss_records['signed_correction_loss']),
            np.mean(loss_records['signed_positive_loss']),
            np.mean(loss_records['signed_negative_loss']),
            np.mean(loss_records['weighted_signed_correction_loss']),
            np.mean(loss_records['margin_loss']),
            np.mean(loss_records['fn_margin_loss']),
            np.mean(loss_records['fp_margin_loss']),
            np.mean(loss_records['weighted_margin_loss']),
            np.mean(loss_records['mean_gate']),
            np.mean(loss_records['ranking_loss']),
            np.mean(loss_records['weighted_ranking_loss']),
            np.mean(loss_records['global_ranking_loss']),
            np.mean(loss_records['thin_ranking_loss']),
            np.mean(loss_records['hard_positive_logit']),
            np.mean(loss_records['thin_positive_logit']),
            np.mean(loss_records['hard_negative_logit']),
            np.mean(loss_records['selected_skeleton_count']),
            np.mean(loss_records['ranking_valid_images']),
            np.mean(loss_records['mean_abs_residual']),
            np.mean(loss_records['mean_abs_correction']),
        )
        checkpoint = {
            'epoch': epoch,
            'row0_checkpoint_path': os.path.abspath(args.row0_checkpoint_path),
            'row0_checkpoint_sha256': row0_sha256,
            'config': vars(args),
            'architecture': {
                'model_name': args.model_name,
                'semantic_base': 'Original AdaptCLIP Protocol-v2 Row0',
                'semantic_base_frozen': True,
                'static_prompts': 'original_adaptclip_prompt_ensemble',
                'prompt_query_adapter_used': False,
                'fusion_formula': 'Z_final = Z_row0 + sigmoid(G) * R',
                'gate_inputs': ['F_high', 'F_sem', 'P_row0'],
                'residual_direction': 'bidirectional_unbounded',
                'residual_initialization': 'weight=0,bias=0',
                'gate_initialization': 'weight=0,bias=-4',
                'gate_target': 'stopgrad(abs(Y - P_row0))',
                'gate_loss_weight': args.gate_loss_weight,
                'preserve_loss_weight': args.preserve_loss_weight,
                'signed_correction_loss': (
                    '0.5 * balanced[YE*softplus(-C), (1-Y)E*softplus(C)]'
                ),
                'signed_correction_loss_weight': args.signed_correction_loss_weight,
                'margin_loss': (
                    '0.5*mean[YE*softplus(m-Z_final)] + '
                    '0.5*mean[(1-Y)E*softplus(m+Z_final)]'
                ),
                'margin_loss_supervision': 'final output[mask_logits]',
                'margin': args.margin,
                'margin_loss_weight': args.margin_loss_weight,
                'hard_pixel_ranking_loss': (
                    'mean_image mean_pair softplus(Z_hard_negative - '
                    'Z_hard_positive) on final output[mask_logits]'
                ),
                'hard_positive_count_per_image': args.hard_positive_count,
                'hard_negative_count_per_image': args.hard_negative_count,
                'ranking_loss_weight': args.ranking_loss_weight,
                'skeleton_balanced_ranking': args.skeleton_balanced_ranking,
                'skeletonization': 'skimage.morphology.skeletonize(binary_GT)',
                'global_positive_count_per_image': args.global_positive_count,
                'skeleton_positive_count_per_image': args.skeleton_positive_count,
            },
        }
        checkpoint[args.checkpoint_state_key] = bridge_model.state_dict()
        torch.save(checkpoint, os.path.join(args.save_path, f'epoch_{epoch}.pth'))
        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    metadata = {
        'completed_epochs': completed_epoch,
        'optimizer': 'Adam',
        'optimizer_betas': [0.5, 0.999],
        'weight_decay': 0.0,
        'new_module_learning_rate': args.new_module_learning_rate,
        'physical_batch_size': args.physical_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.effective_batch_size,
        'loss_weights': {
            'pixel_focal': 1.0,
            'pixel_dice': 1.0,
            'gated_residual_l1': args.residual_l1_weight,
            'error_aware_gate_bce': args.gate_loss_weight,
            'semantic_preservation': args.preserve_loss_weight,
            'signed_error_correction': args.signed_correction_loss_weight,
            'final_logit_margin': args.margin_loss_weight,
            'hard_pixel_ranking': args.ranking_loss_weight,
            'image_ce': 0.0,
        },
        'native_pixel_losses_fp32': True,
        'margin': args.margin,
        'margin_supervision': 'final output[mask_logits]',
        'hard_pixel_ranking_supervision': 'final output[mask_logits]',
        'hard_positive_count_per_image': args.hard_positive_count,
        'hard_negative_count_per_image': args.hard_negative_count,
        'skeleton_balanced_ranking': args.skeleton_balanced_ranking,
        'skeletonization': 'skimage.morphology.skeletonize(binary_GT)',
        'global_positive_count_per_image': args.global_positive_count,
        'skeleton_positive_count_per_image': args.skeleton_positive_count,
        'clip_backbone_frozen': True,
        'visual_adapter_frozen': True,
        'textual_adapter_frozen': True,
        'row0_checkpoint_path': os.path.abspath(args.row0_checkpoint_path),
        'row0_checkpoint_sha256': row0_sha256,
        'row0_prompt_strategy': 'original_adaptclip_prompt_ensemble',
        'row0_probability_pipeline': 'average_visual_textual_then_gaussian_sigma4_then_bilinear1024',
        'image_head_policy': 'exact_frozen_row0',
    }
    with open(os.path.join(args.save_path, 'training_metadata.json'), 'w', encoding='utf-8') as output:
        json.dump(metadata, output, indent=2)


def build_parser():
    parser = argparse.ArgumentParser('BridgeAdaptCLIP-v1.1 training')
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--row0_checkpoint_path', required=True)
    parser.add_argument('--model_name', default='BridgeAdaptCLIP-v1.1')
    parser.add_argument('--checkpoint_state_key', default='bridgeadaptclip_v11')
    parser.add_argument('--pretrained_model', default='ViT-L/14@336px')
    parser.add_argument('--features_list', type=int, nargs='+', default=[6, 12, 18, 24])
    parser.add_argument('--model_input_size', type=int, default=518)
    parser.add_argument('--structural_input_size', type=int, default=1024)
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--vl_reduction', type=int, default=4)
    parser.add_argument('--fusion_channels', type=int, default=128)
    parser.add_argument('--structural_channels', type=int, default=128)
    parser.add_argument('--strip_kernel', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--physical_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=8)
    parser.add_argument('--new_module_learning_rate', type=float, default=3e-4)
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--residual_l1_weight', type=float, default=0.0)
    parser.add_argument('--gate_loss_weight', type=float, default=0.0)
    parser.add_argument('--preserve_loss_weight', type=float, default=0.0)
    parser.add_argument('--signed_correction_loss_weight', type=float, default=0.0)
    parser.add_argument('--margin_loss_weight', type=float, default=0.0)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--ranking_loss_weight', type=float, default=0.0)
    parser.add_argument('--hard_positive_count', type=int, default=256)
    parser.add_argument('--hard_negative_count', type=int, default=256)
    parser.add_argument('--skeleton_balanced_ranking', action='store_true')
    parser.add_argument('--global_positive_count', type=int, default=128)
    parser.add_argument('--skeleton_positive_count', type=int, default=128)
    parser.add_argument('--probability_epsilon', type=float, default=1e-6)
    parser.add_argument('--sigma', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--max_train_steps', type=int, default=0)
    return parser


if __name__ == '__main__':
    parsed_args = build_parser().parse_args()
    setup_seed(parsed_args.seed)
    train(parsed_args)
