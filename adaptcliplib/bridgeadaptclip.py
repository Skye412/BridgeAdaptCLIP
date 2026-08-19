"""BridgeAdaptCLIP-v1 high-resolution structural refinement modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


BRIDGE_NORMAL_ANCHORS = ('a photo of a normal concrete bridge surface',)
BRIDGE_ANOMALY_ANCHORS = ('a photo of a damaged concrete bridge surface',)


def _group_norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False,
            ),
            _group_norm(out_channels),
            nn.GELU(),
        )


class DEGConvLite(nn.Module):
    """Lightweight gated horizontal/vertical strip enhancement."""

    def __init__(self, channels=128, strip_kernel=5):
        super().__init__()
        if strip_kernel % 2 != 1:
            raise ValueError('strip_kernel must be odd')
        pad = strip_kernel // 2
        self.input_projection = ConvNormAct(channels, channels, kernel_size=1)
        self.horizontal = nn.Conv2d(
            channels, channels, kernel_size=(1, strip_kernel),
            padding=(0, pad), groups=channels, bias=False,
        )
        self.vertical = nn.Conv2d(
            channels, channels, kernel_size=(strip_kernel, 1),
            padding=(pad, 0), groups=channels, bias=False,
        )
        self.direction_projection = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            _group_norm(channels),
            nn.GELU(),
        )
        self.edge_gate = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, feature):
        projected = self.input_projection(feature)
        directional = self.direction_projection(torch.cat([
            self.horizontal(projected), self.vertical(projected)
        ], dim=1))
        gate = torch.sigmoid(self.edge_gate(directional))
        return feature + directional * gate


class BridgeAdaptCLIPV1(nn.Module):
    """Native-1024 structural branch with residual spatial refinement."""

    def __init__(
        self,
        semantic_channels=768,
        fusion_channels=128,
        structural_channels=128,
        strip_kernel=5,
        structural_input_size=1024,
    ):
        super().__init__()
        if structural_input_size % 4:
            raise ValueError('structural_input_size must be divisible by 4')
        self.semantic_channels = semantic_channels
        self.fusion_channels = fusion_channels
        self.structural_channels = structural_channels
        self.strip_kernel = strip_kernel
        self.structural_input_size = structural_input_size
        self.fusion_size = structural_input_size // 4

        self.semantic_projection = nn.Sequential(
            nn.Conv2d(semantic_channels + 2, fusion_channels, kernel_size=1, bias=False),
            _group_norm(fusion_channels),
            nn.GELU(),
        )

        self.structural_stem = nn.Sequential(
            ConvNormAct(3, 32, kernel_size=3, stride=2),
            ConvNormAct(32, 64, kernel_size=3, stride=2),
            ConvNormAct(64, 64, kernel_size=3, stride=1),
            ConvNormAct(64, structural_channels, kernel_size=3, stride=1),
        )
        self.degconv_lite = DEGConvLite(structural_channels, strip_kernel)

        self.spatial_attention = nn.Conv2d(structural_channels, 1, kernel_size=1)
        self.fusion_projection = ConvNormAct(
            fusion_channels + structural_channels, fusion_channels, kernel_size=3
        )
        self.decoder_512 = ConvNormAct(fusion_channels, 64, kernel_size=3)
        self.decoder_1024 = ConvNormAct(64, 32, kernel_size=3)
        self.mask_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(
        self,
        visual_patch_feature,
        visual_anomaly_map,
        textual_anomaly_map,
        structural_image,
    ):
        if structural_image.shape[-2:] != (
            self.structural_input_size, self.structural_input_size
        ):
            raise ValueError(
                f'Expected structural input {self.structural_input_size}x'
                f'{self.structural_input_size}, got {tuple(structural_image.shape[-2:])}'
            )

        semantic_grid = visual_patch_feature.shape[-2:]
        visual_cue = F.interpolate(
            visual_anomaly_map, size=semantic_grid,
            mode='bilinear', align_corners=False,
        )
        textual_cue = F.interpolate(
            textual_anomaly_map, size=semantic_grid,
            mode='bilinear', align_corners=False,
        )
        semantic_feature = self.semantic_projection(torch.cat([
            visual_patch_feature, visual_cue, textual_cue
        ], dim=1))

        structural_feature = self.degconv_lite(
            self.structural_stem(structural_image)
        )
        if structural_feature.shape[-2:] != (self.fusion_size, self.fusion_size):
            raise RuntimeError('Structural stem produced an unexpected spatial size.')

        semantic_up = F.interpolate(
            semantic_feature,
            size=structural_feature.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        spatial_attention = torch.sigmoid(
            self.spatial_attention(structural_feature)
        )
        refined_semantic = (1.0 + spatial_attention) * semantic_up
        fused = self.fusion_projection(torch.cat([
            refined_semantic, structural_feature
        ], dim=1))

        decoded = F.interpolate(
            fused, scale_factor=2, mode='bilinear', align_corners=False
        )
        decoded = self.decoder_512(decoded)
        decoded = F.interpolate(
            decoded, scale_factor=2, mode='bilinear', align_corners=False
        )
        decoded = self.decoder_1024(decoded)
        mask_logits = self.mask_head(decoded)

        return {
            'mask_logits': mask_logits,
            'spatial_attention': spatial_attention,
            'semantic_feature': semantic_feature,
            'semantic_up': semantic_up,
            'structural_feature': structural_feature,
            'refined_semantic': refined_semantic,
        }
