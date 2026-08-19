"""Train the full zero-reference BridgeAdaptCLIP-v1 model."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
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
from tools import get_logger, get_transform, setup_seed
from tools.bridgeadaptclip_losses import (
    BinaryDiceLossWithLogits,
    BinaryFocalLossWithLogits,
)

def train(args):
    if args.physical_batch_size * args.gradient_accumulation_steps != args.effective_batch_size:
        raise ValueError(
            'physical_batch_size * gradient_accumulation_steps must equal effective_batch_size'
        )

    os.makedirs(args.save_path, exist_ok=True)
    logger = get_logger(
        args.save_path,
        f'bridge2893_{args.seed}seed_0shot_bridgeadaptclip_v1_train_log.txt',
    )
    logger.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=20)
    dpam_layer = 20
    patch_size = 14
    semantic_channels = 768

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    textual_learner = TextualAdapter(
        model.to('cpu'), args.model_input_size, args.n_ctx,
        static_normal_descriptions=BRIDGE_NORMAL_ANCHORS,
        static_anomaly_descriptions=BRIDGE_ANOMALY_ANCHORS,
    )
    visual_learner = VisualAdapter(
        args.model_input_size, patch_size,
        input_dim=semantic_channels, reduction=args.vl_reduction,
    )
    bridge_model = BridgeAdaptCLIPV1(
        semantic_channels=semantic_channels,
        fusion_channels=args.fusion_channels,
        structural_channels=args.structural_channels,
        strip_kernel=args.strip_kernel,
        structural_input_size=args.structural_input_size,
    )

    model.to(device)
    textual_learner.to(device).train()
    visual_learner.to(device).train()
    bridge_model.to(device).train()
    textual_learner.prepare_static_text_feature(model)

    adapter_parameters = list(textual_learner.parameters()) + list(visual_learner.parameters())
    new_module_parameters = list(bridge_model.parameters())
    optimizer = torch.optim.Adam([
        {'params': adapter_parameters, 'lr': args.adapter_learning_rate},
        {'params': new_module_parameters, 'lr': args.new_module_learning_rate},
    ], betas=(0.5, 0.999))

    focal_loss = BinaryFocalLossWithLogits(alpha=args.focal_alpha, gamma=args.focal_gamma)
    dice_loss = BinaryDiceLossWithLogits()

    clip_transform, _ = get_transform(image_size=args.model_input_size)
    train_data = BridgeDualResolutionDataset(
        args.train_data_path,
        clip_transform=clip_transform,
        structural_input_size=args.structural_input_size,
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
            'Training loader length must be divisible by gradient_accumulation_steps '
            'to preserve the declared effective batch size.'
        )

    amp_enabled = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    logger.info(
        'trainable_params adapters=%d new_modules=%d total=%d; amp=%s',
        sum(p.numel() for p in adapter_parameters),
        sum(p.numel() for p in new_module_parameters),
        sum(p.numel() for p in adapter_parameters + new_module_parameters),
        amp_enabled,
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        loss_records = {'total': [], 'image_ce': [], 'pixel_focal': [], 'pixel_dice': []}
        optimizer.zero_grad(set_to_none=True)

        for batch_index, items in enumerate(tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}')):
            clip_image = items['img'].to(device, non_blocking=True)
            structural_image = items['structural_img'].to(device, non_blocking=True)
            target_mask = items['native_mask'].to(device, non_blocking=True).unsqueeze(1)
            image_target = items['anomaly'].to(device, non_blocking=True).long()

            with torch.no_grad():
                image_features, patch_features = model.encode_image(
                    clip_image, args.features_list, DPAM_layer=dpam_layer
                )

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                global_visual_logits, visual_map, visual_patch_feature = (
                    visual_learner.forward_with_features(
                        image_features, patch_features,
                        textual_learner.static_text_features,
                    )
                )
                learned_prompts, tokenized_prompts = textual_learner()
                learned_text_features = model.encode_text_learn(
                    learned_prompts, tokenized_prompts
                ).float()
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
                mask_logits = bridge_output['mask_logits']

                image_ce = 0.5 * (
                    F.cross_entropy(global_visual_logits, image_target)
                    + F.cross_entropy(global_textual_logits, image_target)
                )
                pixel_focal = focal_loss(mask_logits, target_mask)
                pixel_dice = dice_loss(mask_logits, target_mask)
                total_loss = image_ce + pixel_focal + pixel_dice

            scaler.scale(total_loss / args.gradient_accumulation_steps).backward()
            if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            loss_records['total'].append(float(total_loss.detach()))
            loss_records['image_ce'].append(float(image_ce.detach()))
            loss_records['pixel_focal'].append(float(pixel_focal.detach()))
            loss_records['pixel_dice'].append(float(pixel_dice.detach()))

            if args.max_train_steps and global_step >= args.max_train_steps:
                break

        logger.info(
            'epoch [%d/%d], total=%.5f, image_ce=%.5f, pixel_focal=%.5f, pixel_dice=%.5f',
            epoch, args.epochs,
            np.mean(loss_records['total']),
            np.mean(loss_records['image_ce']),
            np.mean(loss_records['pixel_focal']),
            np.mean(loss_records['pixel_dice']),
        )

        checkpoint = {
            'epoch': epoch,
            'textual_learner': textual_learner.state_dict(),
            'visual_learner': visual_learner.state_dict(),
            'bridgeadaptclip_v1': bridge_model.state_dict(),
            'config': vars(args),
            'architecture': {
                'model_name': 'BridgeAdaptCLIP-v1',
                'reference_mode': '0-reference',
                'semantic_grid': [37, 37],
                'degconv_lite': True,
                'edge_gate': True,
                'direction_embedding': False,
                'srf_residual_formula': '(1 + alpha_s) * F0_up',
                'srf_attention_initialization': 'weight=0,bias=-4',
                'normal_anchor': BRIDGE_NORMAL_ANCHORS[0],
                'anomaly_anchor': BRIDGE_ANOMALY_ANCHORS[0],
            },
        }
        torch.save(checkpoint, os.path.join(args.save_path, f'epoch_{epoch}.pth'))

        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    metadata = {
        'completed_epochs': epoch,
        'optimizer': 'Adam',
        'optimizer_betas': [0.5, 0.999],
        'weight_decay': 0.0,
        'physical_batch_size': args.physical_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.effective_batch_size,
        'loss_weights': {'image_ce': 1.0, 'pixel_focal': 1.0, 'pixel_dice': 1.0},
        'focal_alpha_positive': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'native_pixel_losses_fp32': True,
        'batch_norm_observation': (
            'VisualAdapter BatchNorm statistics use the physical batch; '
            'gradient accumulation only preserves the effective gradient batch.'
        ),
        'clip_backbone_frozen': True,
        'prompt_query_adapter_used': False,
    }
    with open(os.path.join(args.save_path, 'training_metadata.json'), 'w', encoding='utf-8') as output:
        json.dump(metadata, output, indent=2)


def build_parser():
    parser = argparse.ArgumentParser('BridgeAdaptCLIP-v1 training')
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--save_path', required=True)
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
    parser.add_argument('--adapter_learning_rate', type=float, default=1e-3)
    parser.add_argument('--new_module_learning_rate', type=float, default=1e-3)
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--max_train_steps', type=int, default=0, help='smoke-test stop condition')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    setup_seed(args.seed)
    train(args)
