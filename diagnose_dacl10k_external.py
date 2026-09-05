"""Frozen DACL10K sliding-window and structural-correction diagnostics."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from evaluate_dacl10k_external import FrozenTilePredictor
from tools import setup_seed
from tools.dacl10k_external import (
    ProtocolAccumulator,
    StreamingBinaryHistogram,
    build_protocol_masks,
    build_validation_manifest,
    rasterize_damage_labels,
    sliding_window_outputs,
)


REGION_LABELS = {
    "Crack": ("crack", "alligator crack"),
    "Spalling": ("spalling",),
    "Corrosion": ("rust",),
    "Efflorescence": ("efflorescence",),
}


def union_masks(damage_masks, labels):
    result = np.zeros(next(iter(damage_masks.values())).shape, dtype=bool)
    for label in labels:
        result |= damage_masks[label]
    return result


def update_moments(state, name, values, mask):
    count = int(mask.sum(dtype=np.int64))
    if not count:
        return
    selected = values[mask]
    record = state.setdefault(name, {"count": 0, "sum": 0.0, "abs_sum": 0.0})
    record["count"] += count
    record["sum"] += float(selected.sum(dtype=np.float64))
    record["abs_sum"] += float(np.abs(selected).sum(dtype=np.float64))


def write_json(path, payload):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def diagnose(args):
    setup_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(output_dir / "run.log"), logging.StreamHandler()],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("dacl10k-diagnostics")
    manifest = build_validation_manifest(args.dataset_root)
    if args.max_images is not None:
        manifest = manifest[:args.max_images]
    predictor = FrozenTilePredictor(args)
    correction_names = []
    if predictor.fine is not None:
        correction_names.append("fine_correction")
    if predictor.broad is not None:
        correction_names.append("broad_correction")
    geometry_histograms = {
        name: StreamingBinaryHistogram(args.histogram_bins)
        for name in ("edge_dominated", "center_dominated", "overlap", "non_overlap")
    }
    pad_groups = {
        "padded_images": ProtocolAccumulator(args.histogram_bins),
        "non_padded_images": ProtocolAccumulator(args.histogram_bins),
    }
    pad_image_counts = {name: 0 for name in pad_groups}
    correction_moments = {name: {} for name in correction_names}
    start = time.time()

    for index, record in enumerate(tqdm(manifest), start=1):
        with Image.open(record["image_path"]) as source:
            image = ImageOps.exif_transpose(source)

            def diagnostic_outputs(tiles):
                outputs = predictor.predict_outputs(tiles)
                return {
                    key: outputs[key]
                    for key in ("probability", *correction_names)
                }

            outputs, geometry = sliding_window_outputs(
                image, diagnostic_outputs, tile_size=args.tile_size,
                stride=args.stride, tile_batch_size=args.tile_batch_size,
                edge_width=args.edge_width,
            )
        damage = rasterize_damage_labels(record["annotation_path"])
        protocol = build_protocol_masks(damage)
        score = outputs["probability"]
        bridge4 = protocol["Bridge4"]
        for name, histogram in geometry_histograms.items():
            region = geometry[name]
            histogram.update(score, bridge4["positive"], bridge4["ignore"] | ~region)
        pad_group = "padded_images" if bool(geometry["padded_image"]) else "non_padded_images"
        pad_groups[pad_group].update(score, protocol)
        pad_image_counts[pad_group] += 1

        all_damage = protocol["All-Damage"]["positive"]
        bridge_positive = protocol["Bridge4"]["positive"]
        regions = {
            "background": ~all_damage,
            "all_damage": all_damage,
            "Bridge4": bridge_positive,
            "unseen_damage": protocol["Unseen-Damage"]["positive"],
        }
        for name, labels in REGION_LABELS.items():
            regions[name] = union_masks(damage, labels)
        for correction_name in correction_moments:
            values = outputs[correction_name]
            for region_name, mask in regions.items():
                update_moments(
                    correction_moments[correction_name], region_name, values, mask
                )
        if index % 5 == 0 or index == len(manifest):
            write_json(output_dir / "progress.json", {
                "model": args.model,
                "completed_images": index,
                "total_images": len(manifest),
                "elapsed_seconds": time.time() - start,
            })

    correction_report = {}
    for correction_name, regions in correction_moments.items():
        correction_report[correction_name] = {
            name: {
                "pixel_count": record["count"],
                "mean_signed_correction": record["sum"] / record["count"],
                "mean_abs_correction": record["abs_sum"] / record["count"],
            }
            for name, record in regions.items()
        }
    report = {
        "protocol": {
            "protocol_id": "dacl10k-external-diagnostics-v1",
            "model": args.model,
            "edge_definition": (
                f"final pixels where Hann contribution from each tile's outer "
                f"{args.edge_width}px exceeds center contribution"
            ),
            "overlap_definition": "pixels covered by more than one 1024/768 tile",
        },
        "geometry_bridge4_metrics": {
            name: histogram.metrics() for name, histogram in geometry_histograms.items()
        },
        "pad_group_image_counts": pad_image_counts,
        "pad_group_metrics": {
            name: accumulator.report() for name, accumulator in pad_groups.items()
        },
        "correction_moments": correction_report,
    }
    write_json(output_dir / "diagnostics.json", report)
    logger.info("Diagnostics completed: %s", args.model)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("row0", "v20", "v21"), required=True)
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
    parser.add_argument("--edge_width", type=int, default=128)
    parser.add_argument("--tile_batch_size", type=int, default=1)
    parser.add_argument("--histogram_bins", type=int, default=65536)
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
    args = parser.parse_args()
    if args.model != "row0" and (not args.fine_checkpoint or not args.broad_checkpoint):
        parser.error("v20/v21 require --fine_checkpoint and --broad_checkpoint")
    if args.model == "v21":
        args.fine_state_key = "bridgeadaptclip_v21_fine"
        args.broad_state_key = "bridgeadaptclip_v21"
    return args


if __name__ == "__main__":
    diagnose(build_parser())
