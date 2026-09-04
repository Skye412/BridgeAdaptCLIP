# DACL10K External Evaluation Protocol v1

This protocol evaluates frozen Row 0, BridgeAdaptCLIP-v2.0, and
BridgeAdaptCLIP-v2.1 on the 975 labeled images in the DACL10K-v2 official
validation split. DACL10K data is used only for external evaluation: it is not
used for training, checkpoint selection, threshold selection, or model changes.

The 13 official damage labels are evaluated as All-Damage. Crack, Alligator
Crack, Spalling, Rust, and Efflorescence form the five source labels mapped to
the four Bridge2893-compatible categories. The remaining eight damage labels
are ignored in compatible evaluation. The six component labels are never
positive by themselves; component-only pixels are background. Target-positive
pixels take precedence over other-damage ignore pixels when labels overlap.

Native images are processed with 1024x1024 tiles at stride 768. Small images
receive right/bottom replicate padding. Continuous tile probabilities are
combined with a non-periodic 2-D Hann window clamped to 1e-3, then cropped to
the original image size. Ground-truth polygons remain at original resolution.

Pixel AP, AUROC, and F1max use a streaming 65,536-bin score histogram. AP uses
the Average Precision step integral, while AUROC uses trapezoidal integration.
Each metric includes positive-image, positive-pixel, valid-pixel, ignored-pixel,
and prevalence support. Image-level metrics are not reported.

Run unit tests and a smoke evaluation before a full run:

```bash
python -m unittest tests.test_dacl10k_external
MAX_IMAGES=1 bash scripts/run_dacl10k_external_v1.sh row0
bash scripts/run_dacl10k_external_v1.sh all
```
