"""Native-resolution supervised Bridge2893 dataset for comparison models."""

import json
import os
import random

import torch
from PIL import Image, ImageEnhance
from torch.utils import data
from torchvision.transforms import functional as TF

from tools.bridge_masks import decode_bridge_class_masks


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BridgeSupervisedDataset(data.Dataset):
    """Return native 1024 RGB and the four-defect union mask.

    Geometric transforms are applied identically to image and mask. The mild
    photometric jitter is training-only and does not introduce external data.
    """

    def __init__(self, root, image_size=1024, training=False):
        self.root = root
        self.image_size = image_size
        self.training = training
        with open(os.path.join(root, 'meta.json'), encoding='utf-8') as stream:
            meta = json.load(stream)
        split_data = meta.get('train') or meta.get('test')
        if not split_data:
            raise ValueError(f'No train/test samples found in {root}/meta.json')
        self.obj_list = list(split_data)
        self.data_all = []
        for samples in split_data.values():
            self.data_all.extend(samples)

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        sample = self.data_all[index]
        image_path = os.path.join(self.root, sample['img_path'])
        image = Image.open(image_path).convert('RGB')
        expected = (self.image_size, self.image_size)
        if image.size != expected:
            raise ValueError(f'{image_path} has size {image.size}, expected {expected}')
        anomaly = int(sample['anomaly'])
        if anomaly:
            _, union_mask = decode_bridge_class_masks(image_path)
            mask = torch.from_numpy(union_mask.copy()).float()
        else:
            mask = torch.zeros(expected, dtype=torch.float32)

        if self.training:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() < 0.8:
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(0.9, 1.1)
                )
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(0.9, 1.1)
                )

        image_tensor = TF.pil_to_tensor(image).float().div_(255.0)
        image_tensor = TF.normalize(
            image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD
        )
        return {
            'img': image_tensor,
            'native_mask': mask,
            'anomaly': anomaly,
            'cls_name': sample['cls_name'],
            'sample_id': os.path.splitext(os.path.basename(image_path))[0],
            'img_path': image_path,
        }
