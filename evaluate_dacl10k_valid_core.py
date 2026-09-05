"""Frozen DACL10K evaluation using 1024 input tiles and 768 valid cores."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

from evaluate_dacl10k_external import FrozenTilePredictor
from tools import setup_seed
from tools.dacl10k_external import (
    ProtocolAccumulator,
    build_protocol_masks,
    build_validation_manifest,
    rasterize_damage_labels,
    valid_core_window_outputs,
)


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
        handlers=[logging.FileHandler(output_dir / "run.log"), logging.StreamHandler()],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("dacl10k-valid-core")
    manifest = build_validation_manifest(args.dataset_root)
    write_json(output_dir / "validation_manifest.json", {
        "split": "DACL10K-v2 official validation",
        "usage": "frozen-model geometry sensitivity analysis only",
        "count": len(manifest),
        "records": manifest,
    })
    if args.max_images is not None:
        manifest = manifest[:args.max_images]

    predictor = FrozenTilePredictor(args)
    accumulator = ProtocolAccumulator(args.histogram_bins)
    total_tiles = 0
    tile_counts = []
    start = time.time()
    for index, record in enumerate(tqdm(manifest), start=1):
        with Image.open(record["image_path"]) as source:
            image = ImageOps.exif_transpose(source)

            def probability_only(tiles):
                return {"probability": predictor.predict_outputs(tiles)["probability"]}

            outputs, geometry = valid_core_window_outputs(
                image,
                probability_only,
                tile_size=args.tile_size,
                halo=args.halo,
                tile_batch_size=args.tile_batch_size,
            )
        damage = rasterize_damage_labels(record["annotation_path"])
        protocol = build_protocol_masks(damage)
        accumulator.update(outputs["probability"], protocol)
        total_tiles += geometry["tile_count"]
        tile_counts.append(geometry["tile_count"])
        if index % args.save_every == 0 or index == len(manifest):
            write_json(output_dir / "progress.json", {
                "model": args.model,
                "geometry_protocol": "valid_core_128_halo",
                "completed_images": index,
                "total_images": len(manifest),
                "processed_tiles": total_tiles,
                "elapsed_seconds": time.time() - start,
            })

    raw_metrics = accumulator.report()
    percent_metrics = {
        task: {
            key: 100.0 * value if key.startswith("P-") and value is not None else value
            for key, value in values.items()
        }
        for task, values in raw_metrics.items()
    }
    core_size = args.tile_size - 2 * args.halo
    report = {
        "protocol": {
            "protocol_id": "external-geometry-sensitivity-v1",
            "dataset": "DACL10K-v2 official validation",
            "model": args.model,
            "geometry_protocol": "valid_core_128_halo",
            "target_domain_training": False,
            "target_domain_model_selection": False,
            "checkpoint_selection": "frozen Bridge2893 checkpoint",
            "input_tile_size": args.tile_size,
            "halo": args.halo,
            "valid_core_size": core_size,
            "output_stride": core_size,
            "padding": "symmetric replicate context halo",
            "stitching": "non-periodic 2-D Hann weighted valid-core average",
            "ground_truth": "original annotation resolution; never resized",
        },
        "images_evaluated": len(manifest),
        "tile_statistics": {
            "minimum": min(tile_counts),
            "mean": total_tiles / len(tile_counts),
            "maximum": max(tile_counts),
            "total": total_tiles,
        },
        "metrics": raw_metrics,
        "metrics_percent": percent_metrics,
    }
    write_json(output_dir / "metrics.json", report)
    logger.info("Completed DACL10K valid-core %s on %d images", args.model, len(manifest))


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
    parser.add_argument("--halo", type=int, default=128)
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
    args = parser.parse_args()
    if args.model != "row0" and (not args.fine_checkpoint or not args.broad_checkpoint):
        parser.error("v20/v21 require --fine_checkpoint and --broad_checkpoint")
    if args.model == "v21":
        args.fine_state_key = "bridgeadaptclip_v21_fine"
        args.broad_state_key = "bridgeadaptclip_v21"
    return args


if __name__ == "__main__":
    evaluate(build_parser())
