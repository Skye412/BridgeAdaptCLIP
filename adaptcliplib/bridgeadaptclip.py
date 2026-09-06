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


class SquareConvReplacement(nn.Module):
    """Controlled SGE ablation using two independent 3x3 depthwise paths.

    Projection, concatenation, output projection, gating, channels, and the
    residual connection match :class:`DEGConvLite`; only strip geometry is
    replaced by square kernels.
    """

    def __init__(self, channels=128):
        super().__init__()
        self.input_projection = ConvNormAct(channels, channels, kernel_size=1)
        self.horizontal = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False,
        )
        self.vertical = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False,
        )
        self.direction_projection = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            _group_norm(channels),
            nn.GELU(),
        )
        self.edge_gate = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, feature):
        projected = self.input_projection(feature)
        enhanced = self.direction_projection(torch.cat([
            self.horizontal(projected), self.vertical(projected)
        ], dim=1))
        gate = torch.sigmoid(self.edge_gate(enhanced))
        return feature + enhanced * gate


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
        nn.init.zeros_(self.spatial_attention.weight)
        nn.init.constant_(self.spatial_attention.bias, -4.0)
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


class BridgeAdaptCLIPV11(nn.Module):
    """Frozen-semantic base with a spatially gated structural logit residual.

    The module never predicts a replacement anomaly mask. It receives the
    frozen Row-0 native-resolution probability map and applies a trainable,
    bidirectional correction in logit space::

        final_logits = row0_logits + sigmoid(gate_logits) * residual

    Both heads observe structural features, frozen semantic features, and the
    Row-0 probability. Zero initialization makes the initial output exactly
    equal to Row 0.
    """

    def __init__(
        self,
        semantic_channels=768,
        fusion_channels=128,
        structural_channels=128,
        strip_kernel=5,
        structural_input_size=1024,
        probability_epsilon=1e-6,
        structural_variant='strip',
    ):
        super().__init__()
        if structural_input_size % 4:
            raise ValueError('structural_input_size must be divisible by 4')
        if not 0.0 < probability_epsilon < 0.5:
            raise ValueError('probability_epsilon must be in (0, 0.5)')
        self.semantic_channels = semantic_channels
        self.fusion_channels = fusion_channels
        self.structural_channels = structural_channels
        self.strip_kernel = strip_kernel
        self.structural_input_size = structural_input_size
        self.fusion_size = structural_input_size // 4
        self.probability_epsilon = probability_epsilon
        if structural_variant not in ('strip', 'square', 'semantic_only'):
            raise ValueError(
                'structural_variant must be strip, square, or semantic_only'
            )
        self.structural_variant = structural_variant

        self.semantic_projection = nn.Sequential(
            nn.Conv2d(semantic_channels, fusion_channels, kernel_size=1, bias=False),
            _group_norm(fusion_channels),
            nn.GELU(),
        )
        self.structural_stem = nn.Sequential(
            ConvNormAct(3, 32, kernel_size=3, stride=2),
            ConvNormAct(32, 64, kernel_size=3, stride=2),
            ConvNormAct(64, 64, kernel_size=3, stride=1),
            ConvNormAct(64, structural_channels, kernel_size=3, stride=1),
        )
        self.degconv_lite = (
            SquareConvReplacement(structural_channels)
            if structural_variant == 'square'
            else DEGConvLite(structural_channels, strip_kernel)
        )

        joint_channels = structural_channels + fusion_channels + 1
        self.joint_projection = ConvNormAct(
            joint_channels, fusion_channels, kernel_size=3
        )

        self.residual_decoder_512 = ConvNormAct(
            fusion_channels, 64, kernel_size=3
        )
        self.residual_decoder_1024 = ConvNormAct(64, 32, kernel_size=3)
        self.residual_head = nn.Conv2d(32, 1, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        self.gate_projection = ConvNormAct(
            fusion_channels, 64, kernel_size=3
        )
        self.gate_head = nn.Conv2d(64, 1, kernel_size=1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -4.0)

    def forward(
        self,
        visual_patch_feature,
        row0_probability,
        structural_image,
    ):
        expected_size = (self.structural_input_size, self.structural_input_size)
        if structural_image.shape[-2:] != expected_size:
            raise ValueError(
                f'Expected structural input {expected_size}, got '
                f'{tuple(structural_image.shape[-2:])}'
            )
        if row0_probability.shape[-2:] != expected_size:
            raise ValueError(
                f'Expected Row-0 probability {expected_size}, got '
                f'{tuple(row0_probability.shape[-2:])}'
            )
        if row0_probability.shape[1] != 1:
            raise ValueError('row0_probability must have one channel')

        semantic_feature = self.semantic_projection(visual_patch_feature)
        if self.structural_variant == 'semantic_only':
            structural_feature = torch.zeros(
                structural_image.shape[0], self.structural_channels,
                self.fusion_size, self.fusion_size,
                device=structural_image.device, dtype=structural_image.dtype,
            )
        else:
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
        row0_fusion = F.interpolate(
            row0_probability.float(),
            size=structural_feature.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        joint_feature = self.joint_projection(torch.cat([
            structural_feature, semantic_up, row0_fusion
        ], dim=1))

        residual_feature = F.interpolate(
            joint_feature, scale_factor=2, mode='bilinear', align_corners=False
        )
        residual_feature = self.residual_decoder_512(residual_feature)
        residual_feature = F.interpolate(
            residual_feature, scale_factor=2, mode='bilinear', align_corners=False
        )
        residual_feature = self.residual_decoder_1024(residual_feature)
        residual = self.residual_head(residual_feature)

        gate_logits = self.gate_head(self.gate_projection(joint_feature))
        gate_logits = F.interpolate(
            gate_logits, size=expected_size, mode='bilinear', align_corners=False
        )
        gate = torch.sigmoid(gate_logits)

        safe_probability = row0_probability.float().clamp(
            self.probability_epsilon, 1.0 - self.probability_epsilon
        )
        row0_logits = torch.logit(safe_probability)
        gated_residual = gate * residual
        final_logits = row0_logits + gated_residual

        return {
            'mask_logits': final_logits,
            'row0_logits': row0_logits,
            'row0_probability': row0_probability,
            'residual': residual,
            'gate': gate,
            'gate_logits': gate_logits,
            'gated_residual': gated_residual,
            'semantic_feature': semantic_feature,
            'semantic_up': semantic_up,
            'structural_feature': structural_feature,
            'joint_feature': joint_feature,
        }


class BridgeAdaptCLIPV12(BridgeAdaptCLIPV11):
    """v1.1 architecture trained with explicit Row-0 error-aware gate losses."""


class BridgeAdaptCLIPV20(nn.Module):
    """Low-resolution non-positive calibration applied after a frozen fine base."""

    def __init__(self, joint_channels=128, broad_channels=128, output_size=1024):
        super().__init__()
        if output_size % 8:
            raise ValueError('output_size must be divisible by 8')
        self.output_size = output_size
        self.broad_size = output_size // 8
        self.broad_block = nn.Sequential(
            nn.Conv2d(joint_channels + 2, broad_channels, 3, padding=1, bias=False),
            _group_norm(broad_channels),
            nn.GELU(),
            nn.Conv2d(
                broad_channels, broad_channels, 3, padding=2, dilation=2, bias=False
            ),
            _group_norm(broad_channels),
            nn.GELU(),
        )
        self.gate_head = nn.Conv2d(broad_channels, 1, 1)
        self.magnitude_head = nn.Conv2d(broad_channels, 1, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -4.0)
        nn.init.zeros_(self.magnitude_head.weight)
        nn.init.constant_(self.magnitude_head.bias, -4.0)

    def forward(self, fine_joint_feature, fine_logits, row0_probability):
        expected = (self.output_size, self.output_size)
        if fine_logits.shape[-2:] != expected or row0_probability.shape[-2:] != expected:
            raise ValueError('fine_logits and row0_probability must match output_size')
        broad_joint = F.adaptive_avg_pool2d(
            fine_joint_feature.detach(), (self.broad_size, self.broad_size)
        )
        fine_probability = torch.sigmoid(fine_logits.detach().float())
        fine_low = F.interpolate(
            fine_probability, size=broad_joint.shape[-2:], mode='bilinear',
            align_corners=False,
        )
        row0_low = F.interpolate(
            row0_probability.detach().float(), size=broad_joint.shape[-2:],
            mode='bilinear', align_corners=False,
        )
        broad_feature = self.broad_block(torch.cat([broad_joint, fine_low, row0_low], 1))
        gate_logits_low = self.gate_head(broad_feature)
        magnitude_logits_low = self.magnitude_head(broad_feature)
        gate_logits = F.interpolate(
            gate_logits_low, size=expected, mode='bilinear', align_corners=False
        )
        magnitude_logits = F.interpolate(
            magnitude_logits_low, size=expected, mode='bilinear', align_corners=False
        )
        broad_gate = torch.sigmoid(gate_logits)
        broad_magnitude = F.softplus(magnitude_logits.float())
        broad_correction = -broad_gate * broad_magnitude
        final_logits = fine_logits.detach().float() + broad_correction
        return {
            'mask_logits': final_logits,
            'fine_logits': fine_logits.detach(),
            'fine_probability': fine_probability,
            'broad_feature': broad_feature,
            'broad_gate_logits': gate_logits,
            'broad_gate': broad_gate,
            'broad_magnitude': broad_magnitude,
            'broad_correction': broad_correction,
        }


class MultiLevelSemanticGuidance(nn.Module):
    """Zero-initialized shallow CLIP residuals added to the adapted deep feature."""

    def __init__(self, input_channels=768, output_channels=128, shallow_levels=3):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.shallow_levels = shallow_levels
        self.branches = nn.ModuleList()
        for _ in range(shallow_levels):
            branch = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, bias=False),
                _group_norm(output_channels),
                nn.GELU(),
                nn.Conv2d(output_channels, output_channels, 1),
            )
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
            self.branches.append(branch)

    @staticmethod
    def tokens_to_grid(tokens):
        if tokens.ndim != 3:
            raise ValueError('CLIP patch tokens must have shape B x N x C')
        spatial = tokens[:, 1:, :]
        side = int(spatial.shape[1] ** 0.5)
        if side * side != spatial.shape[1]:
            raise ValueError('CLIP spatial token count must form a square grid')
        return spatial.permute(0, 2, 1).reshape(
            spatial.shape[0], spatial.shape[2], side, side
        )

    def forward(self, base_feature, patch_features, active_branch_indices=None):
        if len(patch_features) != self.shallow_levels + 1:
            raise ValueError(
                f'Expected {self.shallow_levels + 1} CLIP levels, got {len(patch_features)}'
            )
        if active_branch_indices is None:
            active_branch_indices = set(range(self.shallow_levels))
        else:
            active_branch_indices = set(active_branch_indices)
        invalid = active_branch_indices.difference(range(self.shallow_levels))
        if invalid:
            raise ValueError(f'Invalid shallow branch indices: {sorted(invalid)}')
        fused = base_feature
        residuals = []
        expected_grid = base_feature.shape[-2:]
        for branch_index, (branch, tokens) in enumerate(
            zip(self.branches, patch_features[:-1])
        ):
            grid = self.tokens_to_grid(tokens.detach().float())
            if grid.shape[1] != self.input_channels or grid.shape[-2:] != expected_grid:
                raise ValueError('All CLIP levels must match the deep 37x37 feature grid')
            residual = branch(F.normalize(grid, dim=1))
            if branch_index in active_branch_indices:
                fused = fused + residual
            residuals.append(residual)
        return fused, residuals


class BridgeAdaptCLIPV21Fine(BridgeAdaptCLIPV12):
    """v1.3 fine architecture guided by frozen CLIP levels 6, 12, 18, and 24."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_feature_levels = (6, 12, 18, 24)
        self.multi_level_guidance = MultiLevelSemanticGuidance(
            input_channels=self.semantic_channels,
            output_channels=self.fusion_channels,
            shallow_levels=3,
        )

    def forward(
        self, visual_patch_feature, patch_features, row0_probability,
        structural_image, active_shallow_levels=None,
    ):
        expected_size = (self.structural_input_size, self.structural_input_size)
        if structural_image.shape[-2:] != expected_size:
            raise ValueError('Structural image has an unexpected spatial size')
        if row0_probability.shape[-2:] != expected_size or row0_probability.shape[1] != 1:
            raise ValueError('Row-0 probability has an unexpected shape')
        base_semantic = self.semantic_projection(visual_patch_feature)
        if active_shallow_levels is None:
            active_branch_indices = None
        else:
            shallow_levels = self.clip_feature_levels[:-1]
            unknown = set(active_shallow_levels).difference(shallow_levels)
            if unknown:
                raise ValueError(f'Unknown shallow CLIP levels: {sorted(unknown)}')
            active_branch_indices = [
                index for index, level in enumerate(shallow_levels)
                if level in active_shallow_levels
            ]
        semantic_feature, level_residuals = self.multi_level_guidance(
            base_semantic, patch_features,
            active_branch_indices=active_branch_indices,
        )
        structural_feature = self.degconv_lite(self.structural_stem(structural_image))
        semantic_up = F.interpolate(
            semantic_feature, size=structural_feature.shape[-2:], mode='bilinear',
            align_corners=False,
        )
        row0_fusion = F.interpolate(
            row0_probability.float(), size=structural_feature.shape[-2:],
            mode='bilinear', align_corners=False,
        )
        joint_feature = self.joint_projection(torch.cat([
            structural_feature, semantic_up, row0_fusion
        ], dim=1))
        residual_feature = F.interpolate(
            joint_feature, scale_factor=2, mode='bilinear', align_corners=False
        )
        residual_feature = self.residual_decoder_512(residual_feature)
        residual_feature = F.interpolate(
            residual_feature, scale_factor=2, mode='bilinear', align_corners=False
        )
        residual_feature = self.residual_decoder_1024(residual_feature)
        residual = self.residual_head(residual_feature)
        gate_logits = self.gate_head(self.gate_projection(joint_feature))
        gate_logits = F.interpolate(
            gate_logits, size=expected_size, mode='bilinear', align_corners=False
        )
        gate = torch.sigmoid(gate_logits)
        safe_probability = row0_probability.float().clamp(
            self.probability_epsilon, 1.0 - self.probability_epsilon
        )
        row0_logits = torch.logit(safe_probability)
        gated_residual = gate * residual
        return {
            'mask_logits': row0_logits + gated_residual,
            'row0_logits': row0_logits,
            'row0_probability': row0_probability,
            'residual': residual,
            'gate': gate,
            'gate_logits': gate_logits,
            'gated_residual': gated_residual,
            'semantic_feature': semantic_feature,
            'base_semantic_feature': base_semantic,
            'multi_level_residuals': level_residuals,
            'active_shallow_levels': (
                self.clip_feature_levels[:-1]
                if active_shallow_levels is None else tuple(active_shallow_levels)
            ),
            'semantic_up': semantic_up,
            'structural_feature': structural_feature,
            'joint_feature': joint_feature,
        }
