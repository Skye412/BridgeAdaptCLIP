"""Zero-training external evaluation on CamCrack789 and Crack500 test sets."""

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
from tools.crack_external import (
    CrackMorphologyMetrics, build_crack_test_manifest, load_crack_mask,
)
from tools.dacl10k_external import StreamingBinaryHistogram, sliding_window_probability


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
    logger = logging.getLogger("crack-external")
    manifest = build_crack_test_manifest(args.dataset_root, args.dataset_name)
    write_json(output_dir / "test_manifest.json", {
        "dataset": args.dataset_name, "split": "official/local test",
        "usage": "external evaluation only", "records": manifest,
    })
    if args.max_images is not None:
        manifest = manifest[:args.max_images]
    predictor = FrozenTilePredictor(args)
    histogram = StreamingBinaryHistogram(args.histogram_bins)
    morphology = CrackMorphologyMetrics(
        threshold=args.binary_threshold,
        tolerance=args.tolerance,
        min_component_pixels=args.min_component_pixels,
    )
    start = time.time()
    for index, record in enumerate(tqdm(manifest), start=1):
        with Image.open(record["image_path"]) as source:
            image = ImageOps.exif_transpose(source)
            probability = sliding_window_probability(
                image, predictor, tile_size=args.tile_size, stride=args.stride,
                tile_batch_size=args.tile_batch_size,
            )
        target = load_crack_mask(record)
        histogram.update(probability, target, np.zeros_like(target, dtype=bool))
        morphology.update(probability, target)
        if index % 10 == 0 or index == len(manifest):
            write_json(output_dir / "progress.json", {
                "dataset": args.dataset_name, "model": args.model,
                "completed_images": index, "total_images": len(manifest),
                "elapsed_seconds": time.time() - start,
            })
    pixel = histogram.metrics()
    shape = morphology.report()
    report = {
        "protocol": {
            "protocol_id": "external-crack-eval-v1",
            "dataset": args.dataset_name,
            "split": "test",
            "model": args.model,
            "target_domain_training": False,
            "target_domain_model_selection": False,
            "binary_threshold": args.binary_threshold,
            "boundary_and_skeleton_tolerance_pixels": args.tolerance,
            "min_component_pixels": args.min_component_pixels,
            "tile_size": args.tile_size,
            "padding": "right-bottom replicate",
        },
        "images_evaluated": len(manifest),
        "pixel_metrics": pixel,
        "pixel_metrics_percent": {
            key: 100.0 * value if key.startswith("P-") and value is not None else value
            for key, value in pixel.items()
        },
        "morphology_metrics": shape,
        "morphology_metrics_percent": {
            key: 100.0 * value if key != "support" else value
            for key, value in shape.items()
        },
    }
    write_json(output_dir / "metrics.json", report)
    logger.info("Completed %s/%s: %s", args.dataset_name, args.model, report)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=("CamCrack789", "Crack500"), required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument(
        "--model", choices=("row0", "fine13", "v20", "v21"), required=True
    )
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
    parser.add_argument("--binary_threshold", type=float, default=0.5)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--min_component_pixels", type=int, default=10)
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
    if args.model != "row0" and not args.fine_checkpoint:
        parser.error("fine13/v20/v21 require --fine_checkpoint")
    if args.model in ("v20", "v21") and not args.broad_checkpoint:
        parser.error("v20/v21 require --broad_checkpoint")
    if args.model == "v21":
        args.fine_state_key = "bridgeadaptclip_v21_fine"
        args.broad_state_key = "bridgeadaptclip_v21"
    return args


if __name__ == "__main__":
    evaluate(build_parser())
