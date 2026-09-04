"""Frozen DACL10K-v2 external anomaly-evaluation utilities.

The official validation annotations are multi-label polygons.  This module
keeps that representation until it constructs the eight binary anomaly tasks
defined by DACL10K External Evaluation Protocol v1.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


DAMAGE_LABELS = (
    "alligator crack", "cavity", "crack", "efflorescence",
    "exposed rebars", "graffiti", "hollowareas", "restformwork",
    "rockpocket", "rust", "spalling", "weathering", "wetspot",
)
OBJECT_LABELS = (
    "bearing", "drainage", "expansion joint", "joint tape",
    "protective equipment", "washouts/concrete corrosion",
)
BRIDGE4_LABELS = (
    "alligator crack", "crack", "efflorescence", "rust", "spalling",
)
UNSEEN_LABELS = tuple(label for label in DAMAGE_LABELS if label not in BRIDGE4_LABELS)
TASK_LABELS = {
    "Bridge4": BRIDGE4_LABELS,
    "Crack": ("alligator crack", "crack"),
    "Spalling": ("spalling",),
    "Corrosion": ("rust",),
    "Efflorescence": ("efflorescence",),
    "Seen-Damage": BRIDGE4_LABELS,
    "Unseen-Damage": UNSEEN_LABELS,
    "All-Damage": DAMAGE_LABELS,
}


def tile_starts(length: int, tile_size: int = 1024, stride: int = 768) -> list[int]:
    """Return deterministic starts, forcing complete right/bottom coverage."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def hann_weight(tile_size: int = 1024, minimum: float = 1e-3) -> np.ndarray:
    """Make the frozen non-periodic 2-D Hann window."""
    one_d = torch.hann_window(tile_size, periodic=False, dtype=torch.float32)
    return torch.outer(one_d, one_d).clamp_min(minimum).numpy()


def build_validation_manifest(dataset_root: str | os.PathLike) -> list[dict]:
    """Validate and return all 975 official-validation image/JSON pairs."""
    root = Path(dataset_root)
    image_dir, annotation_dir = root / "val" / "img", root / "val" / "ann"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Expected val/img and val/ann under {root}")
    records = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        annotation_path = annotation_dir / f"{image_path.name}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        if "size" not in annotation or "objects" not in annotation:
            raise ValueError(f"Invalid annotation structure: {annotation_path}")
        with Image.open(image_path) as image:
            width, height = ImageOps.exif_transpose(image).size
        expected = (int(annotation["size"]["width"]), int(annotation["size"]["height"]))
        if (width, height) != expected:
            raise ValueError(f"Image/annotation size mismatch: {image_path}")
        records.append({
            "sample_id": image_path.stem,
            "image_path": str(image_path),
            "annotation_path": str(annotation_path),
            "width": width,
            "height": height,
            "object_count": len(annotation["objects"]),
        })
    if len(records) != 975:
        raise ValueError(f"Expected 975 official-validation images, found {len(records)}")
    return records


def _polygon_mask(obj: dict, height: int, width: int) -> np.ndarray:
    points = obj.get("points") or {}
    exterior = np.asarray(points.get("exterior", []), dtype=np.float64)
    if exterior.ndim != 2 or exterior.shape[0] < 3 or exterior.shape[1] != 2:
        raise ValueError(f"Invalid polygon exterior for object {obj.get('id')}")
    exterior = np.rint(exterior).astype(np.int32)
    exterior[:, 0] = np.clip(exterior[:, 0], 0, width - 1)
    exterior[:, 1] = np.clip(exterior[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [exterior], 1)
    for interior in points.get("interior", []):
        hole = np.asarray(interior, dtype=np.float64)
        if hole.ndim == 2 and hole.shape[0] >= 3 and hole.shape[1] == 2:
            hole = np.rint(hole).astype(np.int32)
            hole[:, 0] = np.clip(hole[:, 0], 0, width - 1)
            hole[:, 1] = np.clip(hole[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [hole], 0)
    return mask.astype(bool)


def rasterize_damage_labels(annotation_path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Rasterize the 13 damage labels while preserving overlapping labels."""
    with open(annotation_path, "r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    height = int(annotation["size"]["height"])
    width = int(annotation["size"]["width"])
    masks = {label: np.zeros((height, width), dtype=bool) for label in DAMAGE_LABELS}
    known = set(DAMAGE_LABELS) | set(OBJECT_LABELS)
    for obj in annotation["objects"]:
        label = str(obj.get("classTitle", "")).strip().lower()
        if label not in known:
            raise ValueError(f"Unknown DACL10K class {label!r} in {annotation_path}")
        if label in masks:
            masks[label] |= _polygon_mask(obj, height, width)
    return masks


def build_protocol_masks(damage_masks: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Build target/ignore masks with target-positive precedence."""
    missing = set(DAMAGE_LABELS).difference(damage_masks)
    if missing:
        raise ValueError(f"Missing damage masks: {sorted(missing)}")
    shape = next(iter(damage_masks.values())).shape
    all_damage = np.zeros(shape, dtype=bool)
    for label in DAMAGE_LABELS:
        all_damage |= damage_masks[label]
    result = {}
    for task, labels in TASK_LABELS.items():
        positive = np.zeros(shape, dtype=bool)
        for label in labels:
            positive |= damage_masks[label]
        if task == "All-Damage":
            ignore = np.zeros(shape, dtype=bool)
        else:
            ignore = all_damage & ~positive
        result[task] = {"positive": positive, "ignore": ignore}
    return result


@dataclass
class StreamingBinaryHistogram:
    """Fixed-bin, mergeable pixel AP/AUROC/F1max accumulator."""

    bins: int = 65536

    def __post_init__(self):
        self.positive_hist = np.zeros(self.bins, dtype=np.int64)
        self.negative_hist = np.zeros(self.bins, dtype=np.int64)
        self.positive_images = 0
        self.valid_pixels = 0
        self.ignored_pixels = 0

    def update(self, scores: np.ndarray, positive: np.ndarray, ignore: np.ndarray) -> None:
        if scores.shape != positive.shape or scores.shape != ignore.shape:
            raise ValueError("score, positive, and ignore shapes must match")
        safe_scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        index_dtype = np.uint16 if self.bins <= 65536 else np.uint32
        indices = np.floor(
            np.clip(safe_scores, 0.0, 1.0) * (self.bins - 1)
        ).astype(index_dtype)
        self.update_binned(indices, positive, ignore)

    def update_binned(
        self, indices: np.ndarray, positive: np.ndarray, ignore: np.ndarray
    ) -> None:
        if indices.shape != positive.shape or indices.shape != ignore.shape:
            raise ValueError("bin-index, positive, and ignore shapes must match")
        self.positive_images += int(positive.any())
        self.ignored_pixels += int(ignore.sum(dtype=np.int64))
        valid = ~ignore
        self.valid_pixels += int(valid.sum(dtype=np.int64))
        if not valid.any():
            return
        valid_positive = positive[valid]
        valid_indices = indices[valid]
        self.positive_hist += np.bincount(
            valid_indices[valid_positive], minlength=self.bins
        )
        self.negative_hist += np.bincount(
            valid_indices[~valid_positive], minlength=self.bins
        )

    def metrics(self) -> dict:
        positives = int(self.positive_hist.sum())
        negatives = int(self.negative_hist.sum())
        support = {
            "positive_image_count": self.positive_images,
            "positive_pixel_count": positives,
            "negative_pixel_count": negatives,
            "valid_pixel_count": self.valid_pixels,
            "ignored_pixel_count": self.ignored_pixels,
            "positive_prevalence": positives / max(self.valid_pixels, 1),
        }
        if positives == 0 or negatives == 0:
            return {"P-AP": None, "P-AUROC": None, "P-F1max": None, "support": support}
        tp = np.cumsum(self.positive_hist[::-1], dtype=np.float64)
        fp = np.cumsum(self.negative_hist[::-1], dtype=np.float64)
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / positives
        recall_delta = np.diff(np.concatenate(([0.0], recall)))
        average_precision = float(np.sum(recall_delta * precision))
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-15)
        tpr = np.concatenate(([0.0], recall))
        fpr = np.concatenate(([0.0], fp / negatives))
        auroc = float(np.trapz(tpr, fpr))
        return {
            "P-AP": average_precision,
            "P-AUROC": auroc,
            "P-F1max": float(f1.max(initial=0.0)),
            "support": support,
        }

    def state_dict(self) -> dict:
        return {
            "positive_hist": self.positive_hist,
            "negative_hist": self.negative_hist,
            "positive_images": np.asarray(self.positive_images, dtype=np.int64),
            "valid_pixels": np.asarray(self.valid_pixels, dtype=np.int64),
            "ignored_pixels": np.asarray(self.ignored_pixels, dtype=np.int64),
        }

    def load_state_dict(self, state: dict) -> None:
        self.positive_hist = np.asarray(state["positive_hist"], dtype=np.int64)
        self.negative_hist = np.asarray(state["negative_hist"], dtype=np.int64)
        if len(self.positive_hist) != self.bins:
            raise ValueError("Histogram bin count does not match resume state")
        self.positive_images = int(state["positive_images"])
        self.valid_pixels = int(state["valid_pixels"])
        self.ignored_pixels = int(state["ignored_pixels"])


class ProtocolAccumulator:
    def __init__(self, bins: int = 65536):
        self.bins = bins
        self.tasks = {name: StreamingBinaryHistogram(bins) for name in TASK_LABELS}

    def update(self, scores: np.ndarray, masks: dict[str, dict[str, np.ndarray]]) -> None:
        safe_scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        index_dtype = np.uint16 if self.bins <= 65536 else np.uint32
        indices = np.floor(
            np.clip(safe_scores, 0.0, 1.0) * (self.bins - 1)
        ).astype(index_dtype)
        for name, accumulator in self.tasks.items():
            accumulator.update_binned(
                indices, masks[name]["positive"], masks[name]["ignore"]
            )

    def report(self) -> dict:
        report = {name: accumulator.metrics() for name, accumulator in self.tasks.items()}
        report["Bridge4-Macro"] = {
            "P-AP": (
                float(np.mean([
                    report[name]["P-AP"]
                    for name in ("Crack", "Spalling", "Corrosion", "Efflorescence")
                ]))
                if all(report[name]["P-AP"] is not None for name in (
                    "Crack", "Spalling", "Corrosion", "Efflorescence"
                )) else None
            )
        }
        return report

    def save(self, path: str | os.PathLike, completed_images: int) -> None:
        payload = {"completed_images": np.asarray(completed_images, dtype=np.int64)}
        for task, accumulator in self.tasks.items():
            slug = task.lower().replace("-", "_")
            for key, value in accumulator.state_dict().items():
                payload[f"{slug}__{key}"] = value
        temporary = str(path) + ".tmp.npz"
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)

    def load(self, path: str | os.PathLike) -> int:
        with np.load(path) as state:
            completed = int(state["completed_images"])
            for task, accumulator in self.tasks.items():
                slug = task.lower().replace("-", "_")
                accumulator.load_state_dict({
                    key: state[f"{slug}__{key}"]
                    for key in (
                        "positive_hist", "negative_hist", "positive_images",
                        "valid_pixels", "ignored_pixels",
                    )
                })
        return completed


def sliding_window_probability(
    image: Image.Image,
    predict_tiles,
    tile_size: int = 1024,
    stride: int = 768,
    tile_batch_size: int = 1,
) -> np.ndarray:
    """Predict and Hann-stitch one native-resolution RGB image."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    pad_bottom, pad_right = max(0, tile_size - height), max(0, tile_size - width)
    if pad_bottom or pad_right:
        rgb = np.pad(rgb, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
    padded_height, padded_width = rgb.shape[:2]
    starts = [
        (top, left)
        for top in tile_starts(padded_height, tile_size, stride)
        for left in tile_starts(padded_width, tile_size, stride)
    ]
    weight = hann_weight(tile_size)
    weighted_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    for offset in range(0, len(starts), tile_batch_size):
        batch_starts = starts[offset:offset + tile_batch_size]
        tiles = [
            Image.fromarray(rgb[top:top + tile_size, left:left + tile_size])
            for top, left in batch_starts
        ]
        predictions = np.asarray(predict_tiles(tiles), dtype=np.float32)
        if predictions.shape != (len(tiles), tile_size, tile_size):
            raise ValueError(f"Unexpected tile prediction shape {predictions.shape}")
        for prediction, (top, left) in zip(predictions, batch_starts):
            weighted_sum[top:top + tile_size, left:left + tile_size] += prediction * weight
            weight_sum[top:top + tile_size, left:left + tile_size] += weight
    if np.any(weight_sum <= 0):
        raise RuntimeError("Sliding-window stitching left uncovered pixels")
    return (weighted_sum / weight_sum)[:height, :width]
