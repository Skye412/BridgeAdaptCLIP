# BridgeAdaptCLIP-v1 full experiment

## Research question

Can native-resolution structural refinement improve thin bridge crack
localization while preserving Original AdaptCLIP zero-reference anomaly
semantics?

This phase implements the full model only. It does not run component ablations.
All unmeasured BridgeAdaptCLIP-v1 results remain `TBD` until a completed server
experiment produces them.

## Locked comparison

Row 0 is Original AdaptCLIP under Bridge2893 Evaluation Protocol v2:

| Setting | P-AP | Crack P-AP | Spalling P-AP | Corrosion P-AP | Efflorescence P-AP |
|---|---:|---:|---:|---:|---:|
| Original AdaptCLIP Row 0 | 68.12294 | 6.15 | 74.92 | 61.11 | 39.68 |
| BridgeAdaptCLIP-v1 Full | TBD | TBD | TBD | TBD | TBD |

The implementation must not use test data for checkpoint or threshold
selection. Checkpoints are selected by zero-reference validation native-1024
P-AP, and test is run once afterward.

## Architecture

The semantic path keeps ViT-L/14 input at 518 x 518. CLIP image and text
encoders are frozen. VisualAdapter and TextualAdapter remain trainable; PQA is
not instantiated in the zero-reference pixel path.

VisualAdapter exposes its existing adapted 37 x 37 patch representation through
a backward-compatible method. It is concatenated with the Visual/Textual
anomaly cues and projected to 128 channels.

The structural path consumes the original 1024 x 1024 RGB patch with ImageNet
normalization:

```text
3x1024x1024
 -> Conv 3x3 s2, 32
 -> Conv 3x3 s2, 64
 -> Conv 3x3 s1, 64
 -> Conv 3x3 s1, 128
 -> DEGConv-lite(K=5, horizontal + vertical depthwise strips, channel gate)
 -> 128x256x256
```

SRF-inspired refinement is implemented exactly as:

```text
alpha_s = sigmoid(Conv1x1(F_high))
F0_refined = (1 + alpha_s) * bilinear_upsample(F0, 256, 256)
F_fusion = projection(concat(F0_refined, F_high))
```

The spatial-attention convolution starts with zero weights and bias -4, so the
initial refinement multiplier is approximately 1.018 instead of 1.5. This is a
near-identity initialization, not an additional module.

The lightweight decoder uses two bilinear x2 stages with convolutional
refinement and produces one 1024 x 1024 binary anomaly logit map.

## Training contract

- Train/val/test: frozen Bridge2893 seed-42 parent-isolated splits.
- Initialization: same frozen CLIP starting point as Row 0; do not initialize
  adapters from the trained Row 0 checkpoint.
- Effective batch size: 8 (physical 4 x gradient accumulation 2). Physical
  batch 8 was measured to exceed the 16 GB V100 memory limit. VisualAdapter
  BatchNorm therefore observes 4 samples per forward pass; accumulation does
  not make its running statistics identical to Row 0 batch 8.
- Optimizer: Row 0-matched Adam, betas (0.5, 0.999), adapter LR 1e-3,
  new-module LR 1e-3, no weight decay.
- Loss: mean Visual/Textual image CE + binary focal logits loss + binary Dice
  logits loss, weights 1:1:1.
- Focal positive alpha: 0.75; gamma: 2.
- Native-1024 Focal and Dice computations, especially spatial reductions, are
  forced to FP32 under AMP.
- Maximum epochs: 15; seed: 10; AMP enabled by the official script.

## Evaluation contract

- Image metrics use the unchanged Original AdaptCLIP zero-reference image
  score: Visual global score, Textual global score, and the maximum of the
  smoothed semantic anomaly map.
- Pixel metrics use the continuous sigmoid output of the 1024 decoder without
  pre-metric thresholding.
- GT is the original frozen 1024 raster mask and is never resized.
- Report overall image/pixel AUROC, AP, F1max; per-defect P-AP; Crack P-F1max;
  and macro diagnostic P-AP.

## Commands

```bash
EXPERIMENT_DIR=results/bridge2893_bridgeadaptclip_v1_full_native1024_<commit> \
bash scripts/run_bridgeadaptclip_v1.sh
```

The scripts train 15 epochs, evaluate every checkpoint on validation, select by
full-precision validation P-AP, and evaluate the selected checkpoint once on
test.

## Success interpretation

- Primary: Crack P-AP materially exceeds 6.15, ideally reaching at least 15.
- Overall: P-AP should preserve or exceed 68.12294.
- Safety: Spalling P-AP should not materially regress from 74.92.
- Diagnostic: macro P-AP should improve, so an overall gain cannot be attributed
  only to large-area defects.

No BridgeAdaptCLIP-v1 result is claimed in this document before execution.

The Full-v1 loss pathway is intentionally not identical to Row 0: Visual and
Textual local maps receive decoder-mediated pixel gradients, while the new
native-1024 decoder is supervised directly. Consequently, a Full-v1 gain tests
the complete high-resolution design but cannot be attributed to DEGConv-lite
or SRF alone; the planned controls/ablations remain required.
