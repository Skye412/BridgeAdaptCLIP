"""Frozen external crack-dataset manifests and morphology metrics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import skeletonize


GEOMETRY_PROTOCOLS = (
    "current_top_left_pad",
    "symmetric_pad_native_scale",
    "fit_long_side_1024",
)


def prepare_geometry_canvas(image, protocol, canvas_size=1024):
    """Map one sub-canvas image to a frozen square sensitivity canvas."""
    if protocol not in GEOMETRY_PROTOCOLS:
        raise ValueError(f"Unknown geometry protocol: {protocol}")
    source = ImageOps.exif_transpose(image).convert("RGB")
    source_width, source_height = source.size
    if protocol == "fit_long_side_1024":
        scale = canvas_size / max(source_width, source_height)
        content_width = min(canvas_size, int(round(source_width * scale)))
        content_height = min(canvas_size, int(round(source_height * scale)))
        content = source.resize(
            (content_width, content_height), resample=Image.Resampling.BICUBIC
        )
    else:
        if source_width > canvas_size or source_height > canvas_size:
            raise ValueError(
                f"{protocol} requires source dimensions <= {canvas_size}, "
                f"got {(source_width, source_height)}"
            )
        content_width, content_height = source_width, source_height
        content = source

    pad_width, pad_height = canvas_size - content_width, canvas_size - content_height
    if protocol == "current_top_left_pad":
        left = top = 0
    else:
        left, top = pad_width // 2, pad_height // 2
    right, bottom = pad_width - left, pad_height - top
    array = np.asarray(content, dtype=np.uint8)
    canvas = np.pad(
        array, ((top, bottom), (left, right), (0, 0)), mode="edge"
    )
    metadata = {
        "protocol": protocol,
        "source_width": source_width,
        "source_height": source_height,
        "content_width": content_width,
        "content_height": content_height,
        "canvas_size": canvas_size,
        "content_left": left,
        "content_top": top,
        "content_right": left + content_width,
        "content_bottom": top + content_height,
        "valid_content_fraction": (
            content_width * content_height / float(canvas_size * canvas_size)
        ),
    }
    return Image.fromarray(canvas), metadata


def restore_geometry_map(canvas_map, metadata):
    """Crop a canvas output to valid content and map it to source H/W."""
    values = np.asarray(canvas_map, dtype=np.float32)
    canvas_size = metadata["canvas_size"]
    if values.shape != (canvas_size, canvas_size):
        raise ValueError(
            f"Expected {(canvas_size, canvas_size)} canvas output, got {values.shape}"
        )
    content = values[
        metadata["content_top"]:metadata["content_bottom"],
        metadata["content_left"]:metadata["content_right"],
    ]
    source_shape = (metadata["source_height"], metadata["source_width"])
    if content.shape != source_shape:
        content = cv2.resize(
            content,
            (metadata["source_width"], metadata["source_height"]),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.asarray(content, dtype=np.float32)


def build_crack_test_manifest(dataset_root, dataset_name):
    root = Path(dataset_root)
    image_dir, mask_dir = root / "test_img", root / "test_lab"
    if dataset_name == "CamCrack789":
        image_paths = sorted(image_dir.glob("image-*.png"))
        pairs = [(path, mask_dir / path.name.replace("image-", "target-")) for path in image_paths]
        expected_count, mask_rule = 157, "uint8>=128"
    elif dataset_name == "Crack500":
        image_paths = sorted(image_dir.glob("*.jpg"))
        pairs = [(path, mask_dir / f"{path.stem}.png") for path in image_paths]
        expected_count, mask_rule = 675, "binary>0"
    else:
        raise ValueError(f"Unsupported crack dataset: {dataset_name}")
    if len(pairs) != expected_count:
        raise ValueError(f"Expected {expected_count} {dataset_name} test images, found {len(pairs)}")
    records = []
    for image_path, mask_path in pairs:
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing crack mask: {mask_path}")
        with Image.open(image_path) as image:
            width, height = ImageOps.exif_transpose(image).size
        with Image.open(mask_path) as mask:
            mask_size = ImageOps.exif_transpose(mask).size
        if (width, height) != mask_size:
            raise ValueError(f"Image/mask size mismatch: {image_path}")
        records.append({
            "sample_id": image_path.stem,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "width": width,
            "height": height,
            "mask_rule": mask_rule,
        })
    return records


def load_crack_mask(record):
    with Image.open(record["mask_path"]) as source:
        array = np.asarray(ImageOps.exif_transpose(source))
    if array.ndim == 3:
        array = array[..., 0]
    if record["mask_rule"] == "uint8>=128":
        return array >= 128
    if record["mask_rule"] == "binary>0":
        return array > 0
    raise ValueError(f"Unknown mask rule: {record['mask_rule']}")


class CrackMorphologyMetrics:
    """Micro-aggregated fixed-threshold crack morphology metrics."""

    def __init__(self, threshold=0.5, tolerance=3, min_component_pixels=10):
        self.threshold = threshold
        self.tolerance = tolerance
        self.min_component_pixels = min_component_pixels
        self.boundary_pred = 0
        self.boundary_gt = 0
        self.boundary_pred_match = 0
        self.boundary_gt_match = 0
        self.pred_skeleton = 0
        self.gt_skeleton = 0
        self.pred_skeleton_in_gt = 0
        self.gt_skeleton_in_pred = 0
        self.gt_skeleton_tolerant_match = 0
        self.components = 0
        self.recalled_components = 0

    def update(self, probability, target):
        prediction = np.asarray(probability >= self.threshold, dtype=bool)
        target = np.asarray(target, dtype=bool)
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        kernel3 = np.ones((3, 3), dtype=np.uint8)
        tolerance_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * self.tolerance + 1, 2 * self.tolerance + 1)
        )
        pred_u8, gt_u8 = prediction.astype(np.uint8), target.astype(np.uint8)
        pred_boundary = prediction & ~cv2.erode(pred_u8, kernel3, iterations=1).astype(bool)
        gt_boundary = target & ~cv2.erode(gt_u8, kernel3, iterations=1).astype(bool)
        dilated_pred_boundary = cv2.dilate(
            pred_boundary.astype(np.uint8), tolerance_kernel, iterations=1
        ).astype(bool)
        dilated_gt_boundary = cv2.dilate(
            gt_boundary.astype(np.uint8), tolerance_kernel, iterations=1
        ).astype(bool)
        self.boundary_pred += int(pred_boundary.sum())
        self.boundary_gt += int(gt_boundary.sum())
        self.boundary_pred_match += int((pred_boundary & dilated_gt_boundary).sum())
        self.boundary_gt_match += int((gt_boundary & dilated_pred_boundary).sum())

        pred_skeleton = skeletonize(prediction)
        gt_skeleton = skeletonize(target)
        self.pred_skeleton += int(pred_skeleton.sum())
        self.gt_skeleton += int(gt_skeleton.sum())
        self.pred_skeleton_in_gt += int((pred_skeleton & target).sum())
        self.gt_skeleton_in_pred += int((gt_skeleton & prediction).sum())
        dilated_prediction = cv2.dilate(
            pred_u8, tolerance_kernel, iterations=1
        ).astype(bool)
        self.gt_skeleton_tolerant_match += int((gt_skeleton & dilated_prediction).sum())

        count, labels, stats, _ = cv2.connectedComponentsWithStats(gt_u8, connectivity=8)
        for component in range(1, count):
            if int(stats[component, cv2.CC_STAT_AREA]) < self.min_component_pixels:
                continue
            self.components += 1
            if np.any(dilated_prediction[labels == component]):
                self.recalled_components += 1

    def report(self):
        boundary_precision = self.boundary_pred_match / max(self.boundary_pred, 1)
        boundary_recall = self.boundary_gt_match / max(self.boundary_gt, 1)
        boundary_f1 = (
            2 * boundary_precision * boundary_recall
            / max(boundary_precision + boundary_recall, 1e-15)
        )
        topology_precision = self.pred_skeleton_in_gt / max(self.pred_skeleton, 1)
        topology_sensitivity = self.gt_skeleton_in_pred / max(self.gt_skeleton, 1)
        cldice = (
            2 * topology_precision * topology_sensitivity
            / max(topology_precision + topology_sensitivity, 1e-15)
        )
        return {
            "Boundary-F1": boundary_f1,
            "Boundary-Precision": boundary_precision,
            "Boundary-Recall": boundary_recall,
            "clDice": cldice,
            "Skeleton-Recall": self.gt_skeleton_tolerant_match / max(self.gt_skeleton, 1),
            "Connected-Component-Recall": self.recalled_components / max(self.components, 1),
            "support": {
                "gt_boundary_pixels": self.boundary_gt,
                "gt_skeleton_pixels": self.gt_skeleton,
                "gt_components_at_least_min_size": self.components,
            },
        }
