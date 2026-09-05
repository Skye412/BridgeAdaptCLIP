"""Frozen-model geometry sensitivity analysis for external crack datasets."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from evaluate_dacl10k_external import FrozenTilePredictor
from tools import setup_seed
from tools.crack_external import (
    GEOMETRY_PROTOCOLS,
    CrackMorphologyMetrics,
    build_crack_test_manifest,
    load_crack_mask,
    prepare_geometry_canvas,
    restore_geometry_map,
)
from tools.dacl10k_external import StreamingBinaryHistogram


def write_json(path, payload):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def update_correction_moments(state, name, values, target):
    for region, mask in (
        ("all_valid", np.ones_like(target, dtype=bool)),
        ("crack", target),
        ("background", ~target),
    ):
        count = int(mask.sum(dtype=np.int64))
        if not count:
            continue
        selected = values[mask]
        record = state.setdefault(name, {}).setdefault(
            region, {"count": 0, "sum": 0.0, "abs_sum": 0.0}
        )
        record["count"] += count
        record["sum"] += float(selected.sum(dtype=np.float64))
        record["abs_sum"] += float(np.abs(selected).sum(dtype=np.float64))


def evaluate(args):
    setup_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(output_dir / "run.log"), logging.StreamHandler()],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("crack-geometry-sensitivity")
    manifest = build_crack_test_manifest(args.dataset_root, args.dataset_name)
    write_json(output_dir / "test_manifest.json", {
        "dataset": args.dataset_name,
        "split": "official/local test",
        "usage": "frozen-model geometry sensitivity analysis only",
        "records": manifest,
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
    correction_moments = {}
    content_fractions = []
    start = time.time()
    for index, record in enumerate(tqdm(manifest), start=1):
        with Image.open(record["image_path"]) as source:
            canvas, geometry = prepare_geometry_canvas(
                source, args.geometry_protocol, args.tile_size
            )
        outputs = predictor.predict_outputs([canvas])
        restored = {
            name: restore_geometry_map(values[0], geometry)
            for name, values in outputs.items()
        }
        target = load_crack_mask(record)
        probability = restored["probability"]
        histogram.update(probability, target, np.zeros_like(target, dtype=bool))
        morphology.update(probability, target)
        for name in ("fine_correction", "broad_correction"):
            if name in restored:
                update_correction_moments(
                    correction_moments, name, restored[name], target
                )
        content_fractions.append(geometry["valid_content_fraction"])
        if index % 10 == 0 or index == len(manifest):
            write_json(output_dir / "progress.json", {
                "dataset": args.dataset_name,
                "model": args.model,
                "geometry_protocol": args.geometry_protocol,
                "completed_images": index,
                "total_images": len(manifest),
                "elapsed_seconds": time.time() - start,
            })

    correction_report = {
        name: {
            region: {
                "pixel_count": values["count"],
                "mean_signed_correction": values["sum"] / values["count"],
                "mean_absolute_correction": values["abs_sum"] / values["count"],
            }
            for region, values in regions.items()
        }
        for name, regions in correction_moments.items()
    }
    pixel = histogram.metrics()
    shape = morphology.report()
    report = {
        "protocol": {
            "protocol_id": "external-geometry-sensitivity-v1",
            "dataset": args.dataset_name,
            "split": "test",
            "model": args.model,
            "geometry_protocol": args.geometry_protocol,
            "target_domain_training": False,
            "target_domain_model_selection": False,
            "checkpoint_selection": "frozen Bridge2893 checkpoint",
            "binary_threshold": args.binary_threshold,
            "tile_size": args.tile_size,
            "inverse_mapping": "crop valid content then bilinear resize to original H/W",
            "ground_truth": "original resolution; never resized",
        },
        "images_evaluated": len(manifest),
        "valid_content_fraction": {
            "minimum": float(np.min(content_fractions)),
            "mean": float(np.mean(content_fractions)),
            "maximum": float(np.max(content_fractions)),
        },
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
        "correction_moments": correction_report,
    }
    write_json(output_dir / "metrics.json", report)
    logger.info("Completed %s/%s/%s", args.dataset_name, args.model, args.geometry_protocol)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=("CamCrack789", "Crack500"), required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--model", choices=("row0", "fine13", "v20", "v21"), required=True)
    parser.add_argument("--geometry_protocol", choices=GEOMETRY_PROTOCOLS, required=True)
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
