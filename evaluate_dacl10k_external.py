"""Evaluate frozen AdaptCLIP variants on DACL10K-v2 official validation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import BridgeAdaptCLIPV12, BridgeAdaptCLIPV20, BridgeAdaptCLIPV21Fine
from adaptcliplib import TextualAdapter, VisualAdapter
from dataset.bridge_dual_resolution import IMAGENET_MEAN, IMAGENET_STD
from tools import get_transform, setup_seed
from tools.bridge_row0 import resize_row0_probability, smooth_row0_probability
from tools.dacl10k_external import (
    ProtocolAccumulator,
    build_protocol_masks,
    build_validation_manifest,
    rasterize_damage_labels,
    sliding_window_probability,
)


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


class FrozenTilePredictor:
    """One shared frozen semantic base with optional v2.0/v2.1 heads."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_enabled = args.amp and self.device.type == "cuda"
        self.clip_transform, _ = get_transform(args.model_input_size)

        self.clip_model, _ = adaptcliplib.load(args.pretrained_model, device=self.device)
        self.clip_model.visual.DAPM_replace(DPAM_layer=20)
        freeze(self.clip_model)
        self.textual = TextualAdapter(
            self.clip_model.to("cpu"), args.model_input_size, args.n_ctx
        )
        self.visual = VisualAdapter(
            args.model_input_size, 14, input_dim=768, reduction=args.vl_reduction
        )
        row0 = torch.load(args.row0_checkpoint, map_location="cpu")
        self.textual.load_state_dict(row0["textual_learner"])
        self.visual.load_state_dict(row0["visual_learner"])
        freeze(self.textual)
        freeze(self.visual)

        self.fine = None
        self.broad = None
        if args.model != "row0":
            fine_class = (
                BridgeAdaptCLIPV21Fine
                if args.model in ("fine21", "v21") else BridgeAdaptCLIPV12
            )
            self.fine = fine_class(
                semantic_channels=768,
                fusion_channels=args.fusion_channels,
                structural_channels=args.structural_channels,
                strip_kernel=args.strip_kernel,
                structural_input_size=args.tile_size,
                probability_epsilon=args.probability_epsilon,
            )
            fine_checkpoint = torch.load(args.fine_checkpoint, map_location="cpu")
            self.fine.load_state_dict(fine_checkpoint[args.fine_state_key])
            freeze(self.fine)
            if args.model in ("v20", "v21"):
                self.broad = BridgeAdaptCLIPV20(
                    joint_channels=args.fusion_channels,
                    broad_channels=args.broad_channels,
                    output_size=args.tile_size,
                )
                broad_checkpoint = torch.load(args.broad_checkpoint, map_location="cpu")
                self.broad.load_state_dict(broad_checkpoint[args.broad_state_key])
                freeze(self.broad)

        self.clip_model.to(self.device)
        self.textual.to(self.device)
        self.visual.to(self.device)
        if self.fine is not None:
            self.fine.to(self.device)
            if self.broad is not None:
                self.broad.to(self.device)
        self.textual.prepare_static_text_feature(self.clip_model)
        with torch.no_grad():
            prompts, tokens = self.textual()
            self.learned_text = self.clip_model.encode_text_learn(prompts, tokens).float()

    def _structural_tensor(self, image):
        tensor = TF.pil_to_tensor(image).float().div_(255.0)
        return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def predict_outputs(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        clip_images = torch.stack([self.clip_transform(image) for image in images]).to(
            self.device, non_blocking=True
        )
        with torch.inference_mode():
            image_features, patch_features = self.clip_model.encode_image(
                clip_images, self.args.features_list, DPAM_layer=20
            )
            _, visual_map, visual_patch = self.visual.forward_with_features(
                image_features, patch_features, self.textual.static_text_features
            )
            _, textual_map = self.textual.compute_global_local_score(
                image_features, patch_features, self.learned_text
            )
            smoothed = smooth_row0_probability(
                visual_map, textual_map, sigma=self.args.sigma
            )
            row0_probability = resize_row0_probability(
                smoothed, metric_resolution=self.args.tile_size, device=self.device
            )
            if self.args.model == "row0":
                probability = row0_probability[:, 0]
                outputs = {"probability": probability}
            else:
                structural = torch.stack([
                    self._structural_tensor(image) for image in images
                ]).to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    if self.args.model in ("fine21", "v21"):
                        fine_output = self.fine(
                            visual_patch, patch_features, row0_probability, structural
                        )
                    else:
                        fine_output = self.fine(
                            visual_patch, row0_probability, structural
                        )
                    if self.broad is not None:
                        broad_output = self.broad(
                            fine_output["joint_feature"],
                            fine_output["mask_logits"],
                            row0_probability,
                        )
                        logits = broad_output["mask_logits"]
                    else:
                        logits = fine_output["mask_logits"]
                probability = torch.sigmoid(logits.float())[:, 0]
                outputs = {
                    "probability": probability,
                    "row0_probability": row0_probability[:, 0],
                    "fine_probability": torch.sigmoid(
                        fine_output["mask_logits"].float()
                    )[:, 0],
                    "fine_correction": (
                        fine_output["mask_logits"].float()
                        - fine_output["row0_logits"].float()
                    )[:, 0],
                }
                if self.broad is not None:
                    outputs["broad_correction"] = broad_output[
                        "broad_correction"
                    ].float()[:, 0]
        return {
            name: torch.nan_to_num(
                value.float(), nan=0.0,
                posinf=(1.0 if "probability" in name else 0.0),
                neginf=0.0,
            ).cpu().numpy()
            for name, value in outputs.items()
        }

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        return self.predict_outputs(images)["probability"]


def write_json(path, payload):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def evaluate(args):
    setup_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("dacl10k-external")
    manifest = build_validation_manifest(args.dataset_root)
    write_json(output_dir / "validation_manifest.json", {
        "split": "DACL10K-v2 official validation",
        "usage": "external evaluation only",
        "count": len(manifest),
        "records": manifest,
    })
    if args.max_images is not None:
        manifest = manifest[:args.max_images]

    state_path = output_dir / "streaming_state.npz"
    accumulator = ProtocolAccumulator(args.histogram_bins)
    completed = 0
    if args.resume and state_path.is_file():
        completed = accumulator.load(state_path)
        if completed > len(manifest):
            raise ValueError("Resume state exceeds current manifest length")
        logger.info("Resuming after %d images", completed)

    predictor = FrozenTilePredictor(args)
    start_time = time.time()
    processed_tiles = 0
    progress = tqdm(manifest[completed:], initial=completed, total=len(manifest))
    for index, record in enumerate(progress, start=completed):
        with Image.open(record["image_path"]) as source_image:
            image = ImageOps.exif_transpose(source_image)
            score = sliding_window_probability(
                image,
                predictor,
                tile_size=args.tile_size,
                stride=args.stride,
                tile_batch_size=args.tile_batch_size,
            )
        damage_masks = rasterize_damage_labels(record["annotation_path"])
        protocol_masks = build_protocol_masks(damage_masks)
        accumulator.update(score, protocol_masks)
        rows = 1 if record["height"] <= args.tile_size else int(
            np.ceil((record["height"] - args.tile_size) / args.stride) + 1
        )
        cols = 1 if record["width"] <= args.tile_size else int(
            np.ceil((record["width"] - args.tile_size) / args.stride) + 1
        )
        processed_tiles += rows * cols
        completed_now = index + 1
        if completed_now % args.save_every == 0 or completed_now == len(manifest):
            accumulator.save(state_path, completed_now)
            write_json(output_dir / "progress.json", {
                "model": args.model,
                "completed_images": completed_now,
                "total_images": len(manifest),
                "tiles_since_process_start": processed_tiles,
                "elapsed_seconds_since_process_start": time.time() - start_time,
            })

    raw_metrics = accumulator.report()
    percent_metrics = {}
    for task, values in raw_metrics.items():
        percent_metrics[task] = {
            key: (100.0 * value if key.startswith("P-") and value is not None else value)
            for key, value in values.items()
        }
    report = {
        "protocol": {
            "protocol_id": "dacl10k-external-eval-v1",
            "dataset": "DACL10K-v2 official validation",
            "target_domain_training": False,
            "target_domain_model_selection": False,
            "model": args.model,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "padding": "right-bottom replicate",
            "stitching": "non-periodic 2D Hann weighted continuous probability average",
            "histogram_bins": args.histogram_bins,
            "ap_integration": "Average Precision step integral",
            "image_level_metrics": "not reported",
        },
        "checkpoints": {
            "row0": args.row0_checkpoint,
            "fine": args.fine_checkpoint,
            "broad": args.broad_checkpoint,
        },
        "images_evaluated": len(manifest),
        "metrics": raw_metrics,
        "metrics_percent": percent_metrics,
    }
    write_json(output_dir / "metrics.json", report)
    logger.info("Completed %s on %d images", args.model, len(manifest))
    logger.info("Metrics percent: %s", percent_metrics)


def build_parser():
    parser = argparse.ArgumentParser("DACL10K external anomaly evaluation")
    parser.add_argument(
        "--model", choices=("row0", "fine13", "v20", "fine21", "v21"),
        required=True,
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--row0_checkpoint", required=True)
    parser.add_argument("--fine_checkpoint")
    parser.add_argument("--broad_checkpoint")
    parser.add_argument("--fine_state_key", default="bridgeadaptclip_v13")
    parser.add_argument("--broad_state_key", default="bridgeadaptclip_v20")
    parser.add_argument("--pretrained_model", default="ViT-L/14@336px")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--model_input_size", type=int, default=518)
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--tile_batch_size", type=int, default=1)
    parser.add_argument("--histogram_bins", type=int, default=65536)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--max_images", type=int)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--vl_reduction", type=int, default=4)
    parser.add_argument("--fusion_channels", type=int, default=128)
    parser.add_argument("--structural_channels", type=int, default=128)
    parser.add_argument("--broad_channels", type=int, default=128)
    parser.add_argument("--strip_kernel", type=int, default=5)
    parser.add_argument("--probability_epsilon", type=float, default=1e-6)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.model != "row0" and not args.fine_checkpoint:
        parser.error("Fine and final models require --fine_checkpoint")
    if args.model in ("v20", "v21") and not args.broad_checkpoint:
        parser.error("v20/v21 require --broad_checkpoint")
    if args.model in ("fine21", "v21") and args.fine_state_key == "bridgeadaptclip_v13":
        args.fine_state_key = "bridgeadaptclip_v21_fine"
    return args


if __name__ == "__main__":
    evaluate(build_parser())
