"""Exact frozen Row-0 zero-reference inference helpers."""

import hashlib

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def smooth_row0_probability(visual_map, textual_map, sigma=4.0):
    """Reproduce Row 0 fusion and Gaussian smoothing at model resolution."""
    semantic_probability = 0.5 * (
        visual_map[:, 1].float() + textual_map[:, 1].float()
    )
    return torch.stack([
        torch.from_numpy(gaussian_filter(score.cpu().numpy(), sigma=sigma))
        for score in semantic_probability
    ])


def resize_row0_probability(smoothed_probability, metric_resolution=1024, device=None):
    probability = F.interpolate(
        smoothed_probability[:, None].float(),
        size=(metric_resolution, metric_resolution),
        mode='bilinear',
        align_corners=False,
    )
    return probability.to(device) if device is not None else probability


def row0_image_score(global_visual_logits, global_textual_logits, smoothed_probability):
    map_max = smoothed_probability.to(global_visual_logits.device).flatten(1).max(dim=1).values
    return (
        global_visual_logits.float().softmax(dim=-1)[:, 1]
        + global_textual_logits.float().softmax(dim=-1)[:, 1]
        + map_max
    ) / 3.0
