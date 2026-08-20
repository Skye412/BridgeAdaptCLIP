# BridgeAdaptCLIP-v1 zero-cost diagnostics

These diagnostics use the frozen Protocol-v2 Row 0 Epoch 14 and Full-v1 Epoch
4 checkpoints. They do not train or update any parameter.

## Diagnostic A: Full-v1 semantic-only

The Full-v1 Visual/Textual adapters are evaluated without the structural
decoder. Their averaged semantic probability map is Gaussian-smoothed at the
518 model resolution, bilinearly resized once to 1024, and compared with the
original native raster mask. This isolates semantic-adapter/prompt drift from
decoder replacement.

## Diagnostic B: validation-only offline fusion

The Row 0 native-1024 probability `P0` and Full-v1 decoder probability `Ps` are
combined on validation using exactly 22 candidates:

```text
linear:        (1-lambda) * P0 + lambda * Ps
probability OR: 1 - (1-P0) * (1-lambda*Ps)
lambda:         0.0, 0.1, ..., 1.0
```

The candidate maximizing validation native-1024 P-AP is frozen. P-AUROC is the
tie-breaker, followed by deterministic candidate order. Test reads the frozen
selection JSON and does not scan or select any setting.

The fused diagnostic uses the unchanged Row 0 image score; fusion affects only
pixel localization.

## Diagnostic C: composition and source analysis

Validation and test reports include:

- Crack, Spalling, Corrosion, and Efflorescence one-vs-normal metrics;
- CODEBRIM and S2DS overall image/pixel metrics and image support;
- per-source normal/defect counts, class-positive image counts, defect pixels,
  and each class fraction of defect pixels.

Before accepting a report, the script requires the freshly inferred Row 0 and
Full-v1 decoder P-AP to reproduce the existing official value within 0.10
absolute point on both validation and test.

## Run

Initialize a new immutable diagnostic experiment directory, then run:

```bash
EXPERIMENT_DIR=results/<diagnostic_id> \
ROW0_CHECKPOINT=results/bridge2893_original_adaptclip_supervised_v004_82c2be3/checkpoints/epoch_14.pth \
FULL_CHECKPOINT=results/bridge2893_bridgeadaptclip_v1_full_native1024_d380912/checkpoints/epoch_4.pth \
bash scripts/run_bridgeadaptclip_v1_diagnostics.sh
```
