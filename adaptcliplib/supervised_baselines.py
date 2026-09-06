"""Paper comparison segmentation models with native 1024 outputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


def _group_norm(channels):
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()
        branches = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                _group_norm(out_channels), nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            branches.append(nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 3, padding=rate,
                    dilation=rate, bias=False,
                ),
                _group_norm(out_channels), nn.ReLU(inplace=True),
            ))
        self.branches = nn.ModuleList(branches)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _group_norm(out_channels), nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            _group_norm(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )

    def forward(self, feature):
        spatial = feature.shape[-2:]
        pooled = F.interpolate(
            self.pool(feature), size=spatial, mode='bilinear', align_corners=False
        )
        return self.project(torch.cat(
            [branch(feature) for branch in self.branches] + [pooled], dim=1
        ))


class DeepLabV3PlusResNet50(nn.Module):
    """DeepLabv3+ with ImageNet ResNet-50, ASPP, and low-level decoder."""

    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
            replace_stride_with_dilation=(False, False, True),
        )
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = _ASPP(2048, 256)
        self.low_projection = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            _group_norm(48), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            _group_norm(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _group_norm(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, image):
        output_size = image.shape[-2:]
        x = self.stem(image)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)
        x = self.decoder(torch.cat([x, self.low_projection(low)], dim=1))
        return F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)


class SegFormerB1(nn.Module):
    """Hugging Face SegFormer-B1 with ImageNet-pretrained MiT backbone."""

    model_id = 'nvidia/mit-b1'

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as error:
            raise ImportError(
                'SegFormer-B1 requires transformers; install requirements-comparisons.txt'
            ) from error
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_id, num_labels=1, ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig(
                num_labels=1, depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512],
                num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1],
                patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                decoder_hidden_size=256,
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, image):
        logits = self.model(pixel_values=image).logits
        return F.interpolate(
            logits, size=image.shape[-2:], mode='bilinear', align_corners=False
        )


def build_supervised_baseline(name, pretrained=True):
    if name == 'deeplabv3plus_r50':
        return DeepLabV3PlusResNet50(pretrained=pretrained)
    if name == 'segformer_b1':
        return SegFormerB1(pretrained=pretrained)
    raise ValueError(f'Unknown supervised baseline: {name}')
