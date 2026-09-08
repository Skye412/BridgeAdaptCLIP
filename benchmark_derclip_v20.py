"""Reproducible parameter, memory, and throughput audit for DeRCLIP v2.0."""

import argparse
import json
import os
import time

import torch

import adaptcliplib
from adaptcliplib import BridgeAdaptCLIPV12, BridgeAdaptCLIPV20, TextualAdapter, VisualAdapter
from dataset import BridgeDualResolutionDataset
from tools import get_transform, setup_seed
from tools.bridge_row0 import resize_row0_probability, smooth_row0_probability
from tools.bridgeadaptclip_losses import BinaryDiceLossWithLogits, BinaryFocalLossWithLogits
from tools.bridgeadaptclip_v20_losses import (
    broad_gate_and_positive_preservation_losses,
    negative_only_broad_ranking_loss,
)


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def load_pipeline(args, device):
    clip_model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    clip_model.visual.DAPM_replace(DPAM_layer=20)
    freeze(clip_model)
    textual = TextualAdapter(clip_model.to('cpu'), args.model_input_size, args.n_ctx)
    visual = VisualAdapter(args.model_input_size, 14, input_dim=768, reduction=args.vl_reduction)
    row0 = torch.load(args.row0_checkpoint_path, map_location='cpu')
    textual.load_state_dict(row0['textual_learner'])
    visual.load_state_dict(row0['visual_learner'])
    freeze(textual)
    freeze(visual)
    fine = BridgeAdaptCLIPV12(
        semantic_channels=768,
        fusion_channels=args.fusion_channels,
        structural_channels=args.structural_channels,
        strip_kernel=args.strip_kernel,
        structural_input_size=args.structural_input_size,
        probability_epsilon=args.probability_epsilon,
    )
    fine_state = torch.load(args.fine_checkpoint_path, map_location='cpu')
    fine.load_state_dict(fine_state[args.fine_checkpoint_state_key])
    freeze(fine)
    broad = BridgeAdaptCLIPV20(
        joint_channels=args.fusion_channels,
        broad_channels=args.broad_channels,
        output_size=args.structural_input_size,
    )
    broad_state = torch.load(args.checkpoint_path, map_location='cpu')
    broad.load_state_dict(broad_state[args.checkpoint_state_key])
    clip_model.to(device)
    textual.to(device)
    visual.to(device)
    fine.to(device)
    broad.to(device)
    textual.prepare_static_text_feature(clip_model)
    with torch.no_grad():
        prompts, tokens = textual()
        learned_text = clip_model.encode_text_learn(prompts, tokens).float()
    return clip_model, textual, visual, fine, broad, learned_text


def forward_pipeline(modules, learned_text, batch, args, device, train_broad=False):
    clip_model, textual, visual, fine, broad = modules
    clip_image = batch['img'].to(device, non_blocking=True)
    structural = batch['structural_img'].to(device, non_blocking=True)
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
            metric_resolution=args.structural_input_size,
            device=device,
        )
        with torch.amp.autocast('cuda', enabled=args.amp):
            fine_output = fine(visual_patch, row0_probability, structural)
    context = torch.enable_grad() if train_broad else torch.no_grad()
    with context:
        with torch.amp.autocast('cuda', enabled=args.amp):
            output = broad(
                fine_output['joint_feature'], fine_output['mask_logits'], row0_probability
            )
    return output, row0_probability


def inference_profile(modules, learned_text, dataset, args, device):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    first = next(iter(loader))
    forward_pipeline(modules, learned_text, first, args, device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    seconds = 0.0
    images = 0
    for batch in loader:
        batch['img'] = batch['img'].to(device, non_blocking=True)
        batch['structural_img'] = batch['structural_img'].to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        forward_pipeline(modules, learned_text, batch, args, device)
        torch.cuda.synchronize(device)
        seconds += time.perf_counter() - started
        images += len(batch['sample_id'])
    return {
        'batch_size': 1,
        'images': images,
        'pipeline_forward_seconds': seconds,
        'images_per_second': images / seconds,
        'milliseconds_per_image': 1000.0 * seconds / images,
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(device),
        'includes': 'CLIP, adapters, Gaussian smoothing, Fine, Broad, sigmoid-ready logits',
        'excludes': 'data loading, host-to-device transfer, metric calculation, disk output',
    }


def training_memory_profile(modules, learned_text, dataset, args, device):
    broad = modules[-1]
    broad.train()
    for parameter in broad.parameters():
        parameter.requires_grad_(True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.physical_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    iterator = iter(loader)
    optimizer = torch.optim.Adam(broad.parameters(), lr=3e-4, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    focal = BinaryFocalLossWithLogits(alpha=0.75, gamma=2)
    dice = BinaryDiceLossWithLogits()

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        output, row0_probability = forward_pipeline(
            modules, learned_text, batch, args, device, train_broad=True
        )
        target = batch['native_mask'].to(device, non_blocking=True).unsqueeze(1)
        focal_loss = focal(output['mask_logits'], target)
        dice_loss = dice(output['mask_logits'], target)
        _, gate_loss, preserve_loss = broad_gate_and_positive_preservation_losses(
            output['broad_gate_logits'], output['broad_correction'],
            target, output['fine_probability'],
        )
        ranking_loss, _, _, _ = negative_only_broad_ranking_loss(
            output['mask_logits'], output['fine_logits'], target, 256, 256
        )
        loss = focal_loss + dice_loss + 0.1 * gate_loss + 0.05 * preserve_loss
        loss = loss + 0.01 * ranking_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    step(next(iterator))  # allocate optimizer state and CUDA workspaces
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    step(next(iterator))
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    broad.eval()
    for parameter in broad.parameters():
        parameter.requires_grad_(False)
    return {
        'physical_batch_size': args.physical_batch_size,
        'gradient_accumulation_steps': 2,
        'effective_batch_size': 8,
        'amp': args.amp,
        'steady_state_peak_cuda_memory_bytes': peak,
        'scope': 'formal v2.0 final-stage training; frozen CLIP/adapters/Fine, trainable Broad',
    }


def main(args):
    setup_seed(10)
    device = torch.device('cuda')
    loaded = load_pipeline(args, device)
    modules, learned_text = loaded[:5], loaded[5]
    names = ('clip', 'textual_adapter', 'visual_adapter', 'fine_module', 'broad_module')
    component_parameters = {
        name: parameter_count(module) for name, module in zip(names, modules)
    }
    derclip_added = (
        component_parameters['fine_module'] + component_parameters['broad_module']
    )
    task_specific = sum(
        component_parameters[name]
        for name in ('textual_adapter', 'visual_adapter', 'fine_module', 'broad_module')
    )
    clip_transform, _ = get_transform(image_size=args.model_input_size)
    test_data = BridgeDualResolutionDataset(
        args.test_data_path, clip_transform=clip_transform,
        structural_input_size=args.structural_input_size,
    )
    train_data = BridgeDualResolutionDataset(
        args.train_data_path, clip_transform=clip_transform,
        structural_input_size=args.structural_input_size,
    )
    report = {
        'model': 'DeRCLIP v2.0',
        'checkpoint': args.checkpoint_path,
        'hardware': torch.cuda.get_device_name(device),
        'parameters': {
            'components': component_parameters,
            'total_inference_parameters': sum(component_parameters.values()),
            'task_specific_parameters_including_row0_adapters': task_specific,
            'derclip_added_parameters_fine_plus_broad': derclip_added,
            'maximum_simultaneously_trainable_derclip_parameters': max(
                component_parameters['fine_module'], component_parameters['broad_module']
            ),
            'trainable_parameters_final_v2_stage': component_parameters['broad_module'],
            'frozen_parameters_final_v2_stage': (
                sum(component_parameters.values()) - component_parameters['broad_module']
            ),
        },
        'inference': inference_profile(modules, learned_text, test_data, args, device),
        'training_memory': training_memory_profile(
            modules, learned_text, train_data, args, device
        ),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--test_data_path', required=True)
    p.add_argument('--train_data_path', required=True)
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--row0_checkpoint_path', required=True)
    p.add_argument('--fine_checkpoint_path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint_state_key', default='bridgeadaptclip_v20')
    p.add_argument('--fine_checkpoint_state_key', default='bridgeadaptclip_v13')
    p.add_argument('--pretrained_model', default='ViT-L/14@336px')
    p.add_argument('--features_list', type=int, nargs='+', default=[6, 12, 18, 24])
    p.add_argument('--model_input_size', type=int, default=518)
    p.add_argument('--structural_input_size', type=int, default=1024)
    p.add_argument('--n_ctx', type=int, default=12)
    p.add_argument('--vl_reduction', type=int, default=4)
    p.add_argument('--fusion_channels', type=int, default=128)
    p.add_argument('--structural_channels', type=int, default=128)
    p.add_argument('--broad_channels', type=int, default=128)
    p.add_argument('--strip_kernel', type=int, default=5)
    p.add_argument('--probability_epsilon', type=float, default=1e-6)
    p.add_argument('--sigma', type=float, default=4.0)
    p.add_argument('--physical_batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
