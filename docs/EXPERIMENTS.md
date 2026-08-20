# Experiment versioning

Every Bridge2893 run uses one immutable directory under `results/`:

```text
results/<experiment_id>/
├── analysis/          # result interpretation and next actions
├── checkpoints/       # trained models or copied input checkpoints
├── evaluation/        # logs, metrics, predictions, and visualizations
├── provenance/        # Git, environment, hardware, and dataset records
└── metrics.json       # machine-readable final metrics
```

Before starting a run:

1. Commit and push all code and configuration changes.
2. Create a new experiment directory with `scripts/init_experiment.sh`.
3. Point training checkpoint and evaluation output paths into that directory.
4. Never overwrite a completed experiment directory.
5. Record validation-based model selection and keep the frozen test split out of tuning decisions.

Example:

```bash
bash scripts/init_experiment.sh bridge2893_adapter_v002_<short-commit>
```

Experiment artifacts and models stay on the server and are ignored by Git.
Source code, scripts, and configuration are tracked in GitHub.

## Original AdaptCLIP supervised baseline

The Bridge2893 supervised baseline trains only the three upstream AdaptCLIP
adapters. It adds no new model module. Run training and validation with:

```bash
EXPERIMENT_DIR=results/<experiment_id> bash scripts/train_bridge2893_original.sh
EXPERIMENT_DIR=results/<experiment_id> bash scripts/validate_bridge2893_original.sh
```

Validation uses one-shot normal prompts for seeds 10, 20, and 30. The selected
checkpoint maximizes mean validation pixel AP, with mean image AUROC as the
tie-breaker. The frozen test split is evaluated only after selection.

## Bridge2893 Evaluation Protocol v2

The publication Row 0 uses the following immutable protocol:

- AdaptCLIP model input is 518 x 518.
- Pixel metrics are computed at 1024 x 1024 against the original frozen PNG
  raster mask. Ground truth is never downsampled and restored.
- The 518 x 518 floating-point anomaly map is resized to 1024 x 1024 with
  bilinear interpolation and `align_corners=False`; it is not thresholded
  before pixel AUROC or AP.
- The Row 0 checkpoint maximizes zero-reference validation P-AP. Since no
  prompt is sampled, each checkpoint is evaluated once.
- Test is evaluated only once after validation selects the checkpoint.
- Few-reference experiments must use a versioned 35% Prompt-Pool and 65%
  Query-Pool normal-parent manifest. Prompt and Query parents must be disjoint,
  and zero- versus few-reference comparisons use the same Query-Pool. The pool
  manifests are a P1 artifact and must be frozen before publication runs.

The frozen dataset metadata is not edited in place. Each experiment snapshots
the evaluation protocol under `provenance/` so that dataset checksums remain
valid.

Run the native-resolution zero-reference re-evaluation with:

```bash
EXPERIMENT_DIR=results/<v005_id> \
SOURCE_EXPERIMENT_DIR=results/<v004_id> \
bash scripts/run_bridge2893_native1024_eval.sh
```

The earlier v004 experiment remains a valid historical result but is legacy:
its metric resolution is 518 and its checkpoint was selected with one-reference
validation. Its overlapping one-reference support/query result is diagnostic
only and must not be used in a publication table.
# BridgeAdaptCLIP-v1.1

The next formal run is **Frozen Semantic Base + Spatially Gated Structural
Residual**. See `docs/BRIDGEADAPTCLIP_V11.md` and
`configs/bridgeadaptclip_v11.json`. It must load the formal Row-0 Epoch 14
checkpoint, preserve the exact Row-0 image output, select by validation
native-1024 P-AP, and evaluate test once.
