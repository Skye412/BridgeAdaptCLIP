"""Train same-supervision 1024 segmentation baselines on Bridge2893."""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from adaptcliplib.supervised_baselines import build_supervised_baseline
from dataset import BridgeSupervisedDataset
from tools import setup_seed
from tools.bridgeadaptclip_losses import BinaryDiceLossWithLogits
from tools.supervised_protocol import BinaryProtocolMetrics


def configure_optimizer(model, name, epochs, steps_per_epoch):
    if name == 'deeplabv3plus_r50':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01
        )
    total_steps = max(epochs * steps_per_epoch, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(1.0 - step / total_steps, 0.0) ** 0.9
    )
    return optimizer, scheduler


@torch.no_grad()
def validate(model, loader, device, bins, amp):
    model.eval()
    metrics = BinaryProtocolMetrics(bins=bins)
    for batch in tqdm(loader, desc='validation', leave=False):
        images = batch['img'].to(device, non_blocking=True)
        targets = batch['native_mask'].to(device, non_blocking=True).unsqueeze(1)
        with torch.amp.autocast('cuda', enabled=amp):
            logits = model(images)
        metrics.update(torch.sigmoid(logits.float()), targets, batch['anomaly'])
    return metrics.compute()


def train(args):
    if args.physical_batch_size * args.gradient_accumulation_steps != args.effective_batch_size:
        raise ValueError('physical batch times accumulation must equal effective batch')
    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = bool(args.amp and device.type == 'cuda')
    model = build_supervised_baseline(args.model, pretrained=True).to(device)
    train_data = BridgeSupervisedDataset(args.train_data_path, training=True)
    val_data = BridgeSupervisedDataset(args.val_data_path, training=False)
    train_loader = DataLoader(
        train_data, batch_size=args.physical_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    optimizer, scheduler = configure_optimizer(
        model, args.model, args.epochs,
        len(train_loader) // args.gradient_accumulation_steps,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    dice = BinaryDiceLossWithLogits()
    curve = []
    best_ap = -1.0
    best_epoch = None
    global_step = 0
    start_epoch = 1
    wall_start = time.time()
    latest_path = os.path.join(args.output_dir, 'latest.pth')
    if args.resume and os.path.exists(latest_path):
        state = torch.load(latest_path, map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        curve = state['validation_curve']
        best_ap = state['best_ap']
        best_epoch = state['best_epoch']
        global_step = state['global_step']
        start_epoch = state['epoch'] + 1

    peak_memory = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for batch_index, batch in enumerate(
            tqdm(train_loader, desc=f'{args.model} epoch {epoch}/{args.epochs}')
        ):
            images = batch['img'].to(device, non_blocking=True)
            targets = batch['native_mask'].to(device, non_blocking=True).unsqueeze(1)
            with torch.amp.autocast('cuda', enabled=amp):
                logits = model(images)
                bce = F.binary_cross_entropy_with_logits(
                    logits.float(), targets.float()
                )
                dice_loss = dice(logits, targets)
                loss = bce + dice_loss
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            loss_sum += float(loss.detach())
            if device.type == 'cuda':
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

        val_metrics = validate(model, val_loader, device, args.pixel_thresholds, amp)
        row = {
            'epoch': epoch,
            'train_loss': loss_sum / max(len(train_loader), 1),
            'learning_rate': optimizer.param_groups[0]['lr'],
            **val_metrics,
        }
        curve.append(row)
        if val_metrics['P-AP'] > best_ap:
            best_ap = val_metrics['P-AP']
            best_epoch = epoch
            torch.save({
                'model': model.state_dict(), 'model_name': args.model,
                'epoch': epoch, 'validation_metrics': val_metrics,
                'pretraining': (
                    'torchvision ResNet50 IMAGENET1K_V2'
                    if args.model == 'deeplabv3plus_r50' else 'nvidia/mit-b1 ImageNet'
                ),
            }, os.path.join(args.output_dir, 'best.pth'))
        state = {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'epoch': epoch, 'validation_curve': curve, 'best_ap': best_ap,
            'best_epoch': best_epoch, 'global_step': global_step,
        }
        torch.save(state, latest_path)
        with open(os.path.join(args.output_dir, 'validation_curve.json'), 'w') as stream:
            json.dump(curve, stream, indent=2)

    metadata = {
        'model': args.model,
        'task': 'Bridge2893 four-defect-union binary segmentation',
        'input_resolution': 1024,
        'gt_resolution': 1024,
        'epochs': args.epochs,
        'seed': args.seed,
        'physical_batch_size': args.physical_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.effective_batch_size,
        'best_epoch': best_epoch,
        'best_validation_P-AP': best_ap,
        'optimizer': str(optimizer),
        'loss': 'BCEWithLogits + Dice',
        'wall_time_seconds': time.time() - wall_start,
        'peak_gpu_memory_bytes': peak_memory,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with open(os.path.join(args.output_dir, 'training_metadata.json'), 'w') as stream:
        json.dump(metadata, stream, indent=2)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=('deeplabv3plus_r50', 'segformer_b1'), required=True)
    p.add_argument('--train_data_path', required=True)
    p.add_argument('--val_data_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--epochs', type=int, default=45)
    p.add_argument('--physical_batch_size', type=int, default=1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=8)
    p.add_argument('--effective_batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--pixel_thresholds', type=int, default=2048)
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    train(parser().parse_args())
