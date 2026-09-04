# BridgeAdaptCLIP-v2.1 Zero-Training Diagnostics

All diagnostics use the frozen validation-selected v2.1-Fine Epoch 9
checkpoint. No parameters are updated. The Fine-only Test evaluation was run
once for diagnosis; level switching was evaluated on Validation only.

## Fine-only Test comparison

| Metric | v1.3 Fine | v2.1 Fine | Delta |
|---|---:|---:|---:|
| P-AUROC | 96.584974 | 93.600906 | -2.984068 |
| P-AP | 73.406685 | 72.420329 | -0.986356 |
| Macro P-AP | 56.627615 | 54.960123 | -1.667491 |
| Crack P-AP | 39.978542 | 39.425782 | -0.552760 |
| Spalling P-AP | 76.799846 | 76.536837 | -0.263010 |
| Corrosion P-AP | 62.038627 | 61.197706 | -0.840921 |
| Efflorescence P-AP | 47.693442 | 42.680169 | -5.013274 |

The Phase-1 Validation gains do not generalize to Test. The largest failure is
Efflorescence, while the other class changes are smaller but consistently
negative.

## Validation inference-only level ablation

| Active shallow levels | P-AUROC | P-AP | Macro | Crack | Spalling | Corrosion | Efflorescence |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6+12+18 | 93.6364 | **76.9007** | 54.2177 | 21.1655 | 76.8283 | 57.8217 | **61.0550** |
| 12+18 | 92.7916 | 76.6318 | 55.2197 | 23.0721 | **77.6418** | 59.5631 | 60.6020 |
| 6+18 | 92.8804 | 76.1756 | 55.2570 | 22.5951 | 76.8643 | 61.9865 | 59.5820 |
| 6+12 | **93.6472** | 75.7500 | 53.1217 | 16.7986 | 74.4788 | 62.0230 | 59.1863 |
| 6 only | 92.9375 | 74.8931 | 54.0136 | 17.9226 | 74.1563 | **66.3679** | 57.6075 |
| 12 only | 92.5464 | 75.0541 | 53.7189 | 18.0492 | 74.6841 | 63.9317 | 58.2108 |
| 18 only | 91.8916 | 75.5533 | **55.9386** | **23.8356** | 77.5047 | 63.7516 | 58.6626 |

All three levels are complementary for Validation overall P-AP. Layer 18 is
most useful for Crack and macro balance, while layer 6 is strongest for
Corrosion. Removing layer 6 has little overall cost and improves several class
diagnostics, but these Validation-only observations are not sufficient to tune
a new model after the Fine-only Test generalization failure.

The mean residual-to-base L2 ratios are 0.261871 (layer 6), 0.178148 (layer
12), and 0.295311 (layer 18). Layer 18 has the largest residual magnitude, but
magnitude alone does not predict overall P-AP contribution.

## Fine-to-Broad Test increments

| Metric | v1.3 Fine to v2.0 Final | v2.1 Fine to v2.1 Final |
|---|---:|---:|
| P-AUROC | -0.590044 | +1.703919 |
| P-AP | +1.404661 | +3.058447 |
| Macro P-AP | +0.875633 | +2.460354 |
| Crack P-AP | +0.083801 | +0.408423 |
| Spalling P-AP | +0.736841 | +2.550924 |
| Corrosion P-AP | +0.244711 | +0.643294 |
| Efflorescence P-AP | +2.437180 | +6.238775 |

The v2.1 Broad Head improves every reported class over its v2.1 Fine input.
Therefore the final class trade-off against v2.0 is not caused by Broad Head
erosion; it originates in the weaker Test generalization of the multi-level
Fine Expert. Broad calibration substantially compensates for that weakness.

## Decision

Stop expanding the frozen multi-level CLIP direction on seed42. Keep v2.1 as a
valid overall-P-AP variant and semantic ablation, but retain v2.0 as the balanced
default until an independent split or multiple training seeds can resolve the
small final differences. Do not select a reduced layer subset using these
already-inspected Validation/Test results.
