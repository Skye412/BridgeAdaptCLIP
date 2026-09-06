"""Domain-supervised AnomalyCLIP comparison under Bridge2893 Protocol v2.

The model and prompt learner come from the pinned official AnomalyCLIP source.
CLIP/pixel training follows the official loss path at 518; validation/test
predictions are bilinearly lifted to the frozen native 1024 GT evaluator.
"""

import argparse
import importlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BridgeDualResolutionDataset
from tools import get_transform, setup_seed
from tools.supervised_protocol import BinaryProtocolMetrics


def load_upstream(path):
    sys.path.insert(0, os.path.abspath(path))
    library = importlib.import_module('AnomalyCLIP_lib')
    prompts = importlib.import_module('prompt_ensemble')
    losses = importlib.import_module('loss')
    return (
        library, prompts.AnomalyCLIP_PromptLearner,
        losses.FocalLoss, losses.BinaryDiceLoss,
    )


def features_and_maps(library, model, prompt_learner, images, features_list, image_size):
    image_features, patch_features = model.encode_image(
        images, features_list, DPAM_layer=20
    )
    image_features = F.normalize(image_features, dim=-1)
    prompts, tokenized, compound = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized, compound).float()
    text_features = torch.stack(torch.chunk(text_features, 2, dim=0), dim=1)
    text_features = F.normalize(text_features, dim=-1)
    image_logits = (image_features.unsqueeze(1) @ text_features.permute(0, 2, 1))
    image_logits = image_logits[:, 0] / 0.07
    maps = []
    for patch in patch_features:
        patch = F.normalize(patch, dim=-1)
        similarity, _ = library.compute_similarity(patch, text_features[0])
        similarity_map = library.get_similarity_map(
            similarity[:, 1:, :], image_size
        ).permute(0, 3, 1, 2)
        maps.append(similarity_map)
    return image_logits, maps


@torch.no_grad()
def validate(library, model, prompt_learner, loader, args, device):
    prompt_learner.eval()
    metrics = BinaryProtocolMetrics(args.pixel_thresholds)
    for batch in tqdm(loader, desc='validation', leave=False):
        images = batch['img'].to(device, non_blocking=True)
        _, maps = features_and_maps(
            library, model, prompt_learner, images,
            args.features_list, args.image_size,
        )
        probability = torch.stack([
            torch.from_numpy(gaussian_filter(
                score.detach().float().cpu().numpy(), sigma=args.sigma
            )) for score in torch.stack([
                (item[:, 1] + 1.0 - item[:, 0]) / 2.0 for item in maps
            ]).mean(0)
        ]).to(device).unsqueeze(1)
        probability = F.interpolate(
            probability, size=(1024, 1024), mode='bilinear', align_corners=False
        ).clamp(0, 1)
        target = batch['native_mask'].to(device).unsqueeze(1)
        metrics.update(probability, target, batch['anomaly'])
    return metrics.compute()


def train(args):
    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    library, PromptLearner, FocalLoss, BinaryDiceLoss = load_upstream(
        args.upstream_root
    )
    device = torch.device('cuda')
    details = {
        'Prompt_length': args.n_ctx,
        'learnabel_text_embedding_depth': args.depth,
        'learnabel_text_embedding_length': args.t_n_ctx,
    }
    model, _ = library.load(
        'ViT-L/14@336px', device=device, design_details=details,
        download_root=args.clip_cache,
    )
    model.visual.DAPM_replace(DPAM_layer=20)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prompt_learner = PromptLearner(model.to('cpu'), details).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        prompt_learner.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    clip_transform, _ = get_transform(args.image_size)
    train_data = BridgeDualResolutionDataset(
        args.train_data_path, clip_transform, structural_input_size=1024
    )
    val_data = BridgeDualResolutionDataset(
        args.val_data_path, clip_transform, structural_input_size=1024
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    focal = FocalLoss()
    dice = BinaryDiceLoss()
    curve, best_ap, best_epoch = [], -1.0, None
    start_epoch = 1
    latest = os.path.join(args.output_dir, 'latest.pth')
    if args.resume and os.path.exists(latest):
        state = torch.load(latest, map_location='cpu')
        prompt_learner.load_state_dict(state['prompt_learner'])
        optimizer.load_state_dict(state['optimizer'])
        curve, best_ap, best_epoch = (
            state['validation_curve'], state['best_ap'], state['best_epoch']
        )
        start_epoch = state['epoch'] + 1
    wall_start = time.time()
    peak_memory = 0
    for epoch in range(start_epoch, args.epochs + 1):
        prompt_learner.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'AnomalyCLIP-BD {epoch}/{args.epochs}'):
            images = batch['img'].to(device, non_blocking=True)
            labels = batch['anomaly'].long().to(device)
            target = F.interpolate(
                batch['native_mask'].to(device).unsqueeze(1),
                size=(args.image_size, args.image_size), mode='nearest',
            )
            with torch.no_grad():
                image_features, patch_features = model.encode_image(
                    images, args.features_list, DPAM_layer=20
                )
                image_features = F.normalize(image_features, dim=-1)
            prompts, tokenized, compound = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(
                prompts, tokenized, compound
            ).float()
            text_features = torch.stack(torch.chunk(text_features, 2, dim=0), dim=1)
            text_features = F.normalize(text_features, dim=-1)
            image_logits = (
                image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            )[:, 0] / 0.07
            loss = F.cross_entropy(image_logits, labels)
            for patch in patch_features:
                patch = F.normalize(patch, dim=-1)
                similarity, _ = library.compute_similarity(patch, text_features[0])
                logits = library.get_similarity_map(
                    similarity[:, 1:, :], args.image_size
                ).permute(0, 3, 1, 2)
                loss = loss + 4.0 * (
                    focal(logits, target)
                    + dice(logits[:, 1], target[:, 0])
                    + dice(logits[:, 0], 1-target[:, 0])
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
        val_metrics = validate(
            library, model, prompt_learner, val_loader, args, device
        )
        row = {'epoch': epoch, 'train_loss': sum(losses)/len(losses), **val_metrics}
        curve.append(row)
        if val_metrics['P-AP'] > best_ap:
            best_ap, best_epoch = val_metrics['P-AP'], epoch
            torch.save({
                'prompt_learner': prompt_learner.state_dict(),
                'epoch': epoch, 'validation_metrics': val_metrics,
                'model_name': 'anomalyclip_bd',
                'upstream_commit': args.upstream_commit,
            }, os.path.join(args.output_dir, 'best.pth'))
        torch.save({
            'prompt_learner': prompt_learner.state_dict(),
            'optimizer': optimizer.state_dict(), 'epoch': epoch,
            'validation_curve': curve, 'best_ap': best_ap,
            'best_epoch': best_epoch,
        }, latest)
        with open(os.path.join(args.output_dir, 'validation_curve.json'), 'w') as stream:
            json.dump(curve, stream, indent=2)
    metadata = {
        'model': 'AnomalyCLIP-BD', 'upstream_commit': args.upstream_commit,
        'upstream_url': 'https://github.com/zqhang/AnomalyCLIP',
        'training_protocol': 'official prompt learner/loss path adapted to Bridge2893',
        'image_size': args.image_size, 'native_metric_resolution': 1024,
        'epochs': args.epochs, 'seed': args.seed, 'batch_size': args.batch_size,
        'optimizer': 'Adam lr=1e-3 betas=(0.5,0.999)',
        'best_epoch': best_epoch, 'best_validation_P-AP': best_ap,
        'wall_time_seconds': time.time()-wall_start,
        'peak_gpu_memory_bytes': peak_memory,
        'trainable_parameters': sum(p.numel() for p in prompt_learner.parameters()),
    }
    with open(os.path.join(args.output_dir, 'training_metadata.json'), 'w') as stream:
        json.dump(metadata, stream, indent=2)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream_root', required=True)
    p.add_argument('--clip_cache', default='/home/skye/.cache/clip')
    p.add_argument('--upstream_commit', default='3911738c0867544f545a076ad78f3f11d9ecbfdf')
    p.add_argument('--train_data_path', required=True)
    p.add_argument('--val_data_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--epochs', type=int, default=45)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--image_size', type=int, default=518)
    p.add_argument('--features_list', type=int, nargs='+', default=[6,12,18,24])
    p.add_argument('--depth', type=int, default=9)
    p.add_argument('--n_ctx', type=int, default=12)
    p.add_argument('--t_n_ctx', type=int, default=4)
    p.add_argument('--sigma', type=float, default=4)
    p.add_argument('--pixel_thresholds', type=int, default=2048)
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    train(parser().parse_args())
