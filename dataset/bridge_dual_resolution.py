"""Dual-resolution Bridge2893 dataset for BridgeAdaptCLIP-v1."""

import json
import os

import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import functional as TF

from tools.bridge_masks import decode_bridge_class_masks


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BridgeDualResolutionDataset(data.Dataset):
    def __init__(self, root, clip_transform, structural_input_size=1024):
        self.root = root
        self.clip_transform = clip_transform
        self.structural_input_size = structural_input_size

        with open(os.path.join(root, 'meta.json'), 'r', encoding='utf-8') as meta_file:
            meta = json.load(meta_file)
        split_data = meta.get('train') or meta.get('test')
        if not split_data:
            raise ValueError(f'No train/test samples found in {root}/meta.json')

        self.cls_names = list(split_data)
        self.obj_list = self.cls_names
        self.data_all = []
        for cls_name in self.cls_names:
            self.data_all.extend(split_data[cls_name])

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        sample = self.data_all[index]
        relative_path = sample['img_path']
        image_path = os.path.join(self.root, relative_path)
        image = Image.open(image_path).convert('RGB')
        expected_size = (self.structural_input_size, self.structural_input_size)
        if image.size != expected_size:
            raise ValueError(
                f'{image_path} has size {image.size}; expected native {expected_size}.'
            )

        clip_image = self.clip_transform(image)
        structural_image = TF.pil_to_tensor(image).float().div_(255.0)
        structural_image = TF.normalize(
            structural_image, mean=IMAGENET_MEAN, std=IMAGENET_STD
        )

        anomaly = int(sample['anomaly'])
        if anomaly:
            _, native_mask = decode_bridge_class_masks(image_path)
            if native_mask.shape != expected_size:
                raise ValueError(
                    f'{image_path} GT has shape {native_mask.shape}; expected {expected_size}.'
                )
            native_mask = torch.from_numpy(native_mask.copy()).float()
        else:
            native_mask = torch.zeros(expected_size, dtype=torch.float32)

        return {
            'img': clip_image,
            'structural_img': structural_image,
            'native_mask': native_mask,
            'anomaly': anomaly,
            'cls_name': sample['cls_name'],
            'sample_id': os.path.splitext(os.path.basename(relative_path))[0],
            'img_path': image_path,
        }
