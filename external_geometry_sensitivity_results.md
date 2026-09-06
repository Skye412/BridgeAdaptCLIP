# External Geometry Sensitivity Analysis v1

## Status and provenance

- Status: complete
- Code commit used by the formal queue: `a52d6a8`
- Server result directory: `/home/skye/data/Skye/AdaptCLIP/results/external_geometry_sensitivity_v1_a52d6a8`
- Scope: inference-only geometry audit; no checkpoint was retrained or selected from external-test results
- Completed artifacts: 27 formal `metrics.json` files plus the DACL10K Row 0 edge/center diagnostic
- Validation: all 28 smoke components passed and 13 relevant unit tests passed

The three crack-image protocols are:

- **A — current/top-left pad:** the frozen external-evaluation path.
- **B — symmetric pad/native scale:** keep the source scale and center it in the 1024 canvas using replicate padding.
- **C — fit-long-side 1024:** bicubic-resize the long side to 1024, symmetric replicate-pad the remaining side, then bilinearly map continuous predictions back to the original GT size.

All protocols preserve the original-size GT. Row 0 and all structural models use the same source image, geometry operation, and inverse mapping within each protocol.

## Reproduction check

Protocol A reproduced the previously frozen external results to numerical precision. Across the reported P-AP, P-AUROC, and P-F1max values, the largest absolute discrepancy was below `9e-8`. The new evaluator therefore measures geometry changes rather than introducing an evaluation drift.

## CamCrack789

### Protocol A — current top-left padding

| Model | Valid content | P-AP | P-AUROC | P-F1max | Boundary-F1 | clDice | Skeleton recall | Component recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Row 0 | 0.292969 | 32.636850 | 93.730734 | 41.249214 | 17.624155 | 53.328402 | 66.012107 | 81.623932 |
| Fine-v1.3 | 0.292969 | 61.777409 | 96.075645 | 57.646859 | 59.320944 | 66.064914 | 87.789969 | 91.880342 |
| v2.0 | 0.292969 | **62.261040** | **96.742734** | **57.742288** | **62.693595** | **68.629925** | 86.929458 | 91.025641 |
| v2.1 | 0.292969 | 59.490870 | 96.316047 | 56.021782 | 58.725738 | 67.437352 | **90.194785** | 90.598291 |

### Protocol B — symmetric padding at native scale

| Model | Valid content | P-AP | P-AUROC | P-F1max | Boundary-F1 | clDice | Skeleton recall | Component recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Row 0 | 0.292969 | 28.842708 | 92.665216 | 38.292699 | 8.789631 | 42.109801 | 86.475134 | 91.452991 |
| Fine-v1.3 | 0.292969 | 53.334041 | 93.311378 | **52.226835** | 26.012518 | 33.681767 | 96.415525 | **96.153846** |
| v2.0 | 0.292969 | **53.341781** | **93.312873** | 52.224101 | **27.304035** | **34.316316** | 96.029119 | **96.153846** |
| v2.1 | 0.292969 | 11.650233 | 83.484158 | 16.459286 | 20.879869 | 26.547516 | **97.753439** | 97.863248 |

### Protocol C — fit long side to 1024

| Model | Valid content | P-AP | P-AUROC | P-F1max | Boundary-F1 | clDice | Skeleton recall | Component recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Row 0 | 0.750000 | 43.524884 | 95.625453 | 49.226572 | 28.927583 | **61.497721** | 66.916797 | 79.914530 |
| Fine-v1.3 | 0.750000 | 52.231617 | 95.432413 | 48.968917 | 57.708644 | 59.290029 | **87.646880** | **94.871795** |
| v2.0 | 0.750000 | **52.562464** | **95.667657** | **49.464189** | **60.570480** | 61.135846 | 86.132248 | 92.735043 |
| v2.1 | 0.750000 | 39.451491 | 94.083572 | 41.351279 | 59.293290 | 59.822773 | 85.949596 | 91.025641 |

### CamCrack789 interpretation

- Scaling the content up helps Row 0 strongly: P-AP rises from 32.64 to 43.52 (`+10.89`). The original under-filled 1024 canvas therefore penalizes the CLIP-only baseline.
- The same scale change does not help the structural models: v2.0 falls from 62.26 to 52.56 (`-9.70`), Fine-v1.3 falls by `-9.55`, and v2.1 falls by `-20.04`.
- The Row 0–v2.0 P-AP gap shrinks from `29.62` under A to `9.04` under C. Thus part of the striking relative gain under A comes from Row 0 under-fill. It is not the whole result: under C, v2.0 still retains much stronger morphology localization, especially Boundary-F1 (`60.57` versus `28.93`).
- Merely changing top-left placement to symmetric padding hurts ranking, and v2.1 is exceptionally sensitive: its P-AP drops from 59.49 to 11.65. Skeleton/component recall can increase while P-AP, Boundary-F1, and clDice collapse, indicating many poorly ranked high-recall responses rather than better calibrated localization.

## Crack500

### Protocol A — current top-left padding

| Model | Valid content | P-AP | P-AUROC | P-F1max | Boundary-F1 | clDice | Skeleton recall | Component recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Row 0 | 0.222078 | **23.464373** | **76.170924** | **29.465465** | 8.758552 | **29.599641** | 52.043728 | 68.585643 |
| Fine-v1.3 | 0.222078 | 19.605700 | 72.041908 | 23.122918 | 11.037448 | 22.291143 | **70.813972** | **90.689410** |
| v2.0 | 0.222078 | 19.603574 | 71.883768 | 23.125873 | **11.202881** | 22.445046 | 69.916719 | 88.841507 |
| v2.1 | 0.222078 | 9.805525 | 66.218603 | 17.562183 | 7.018371 | 18.780105 | 79.078777 | 90.547264 |

### Protocol B — symmetric padding at native scale

| Model | Valid content | P-AP | P-AUROC | P-F1max | Boundary-F1 | clDice | Skeleton recall | Component recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Row 0 | 0.222078 | **21.682526** | **73.374498** | **27.346153** | 4.921094 | **18.353364** | 81.503712 | 88.201848 |
| Fine-v1.3 | 0.222078 | 19.888496 | 70.233055 | 25.896771 | 5.926192 | 14.334823 | 93.581304 | 97.370291 |
| v2.0 | 0.222078 | 19.889434 | 70.246961 | 25.896453 | **5.959035** | 14.348875 | 93.486181 | 97.157072 |
| v2.1 | 0.222078 | 6.571753 | 55.346922 | 12.547735 | 3.763442 | 12.261365 | **96.869010** | **98.223170** |

### Protocol C — fit long side to 1024

| Model | Valid content | P-AP | P-AUROC | P-F1max | Boundary-F1 | clDice | Skeleton recall | Component recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Row 0 | 0.567969 | **31.753532** | **80.975278** | **37.058957** | 13.315753 | **36.332410** | 56.849367 | 75.195451 |
| Fine-v1.3 | 0.567969 | 17.598342 | 72.655484 | 21.078301 | 13.406016 | 24.224704 | 68.773335 | **92.466240** |
| v2.0 | 0.567969 | 17.683554 | 72.650102 | 21.179004 | **13.821354** | 24.393995 | 67.752422 | 90.120824 |
| v2.1 | 0.567969 | 10.291516 | 68.988037 | 18.690329 | 9.372314 | 19.764862 | **74.161843** | 90.049751 |

### Crack500 interpretation

- Scaling again helps Row 0 substantially: P-AP rises from 23.46 to 31.75 (`+8.29`).
- It does **not** recover the structural models. v2.0 falls from 19.60 to 17.68 (`-1.92`), Fine-v1.3 falls by `-2.01`, and v2.1 only rises from 9.81 to 10.29 while remaining far below Row 0.
- Protocol C improves Boundary-F1 but not structural-model AP. The structural branch can detect more crack-like geometry while its score ordering/calibration remains poor.
- Therefore the Crack500 failure cannot be explained mainly by small content occupying the 1024 canvas. Domain texture, annotation style, and structural-score calibration remain the leading explanations.
- Symmetric padding raises skeleton/component recall but generally degrades AP and morphology precision, again showing strong placement/padding sensitivity.

## Correction-path sensitivity

Mean absolute correction values expose an important mechanism behind the padding sensitivity:

| Dataset | Model/branch | A | B | C |
|---|---|---:|---:|---:|
| CamCrack789 | v2.0 fine | 1.684078 | 1.639637 | 1.829198 |
| CamCrack789 | v2.0 broad | 2.575116 | 1.005446 | 2.772975 |
| CamCrack789 | v2.1 fine | 2.621110 | 2.305498 | 3.173879 |
| CamCrack789 | v2.1 broad | 2.012122 | 0.469676 | 2.227248 |
| Crack500 | v2.0 fine | 1.532963 | 2.125287 | 1.483640 |
| Crack500 | v2.0 broad | 0.690569 | 0.087649 | 0.934859 |
| Crack500 | v2.1 fine | 2.240572 | 3.290479 | 2.032716 |
| Crack500 | v2.1 broad | 0.583284 | 0.055323 | 1.053406 |

The broad correction nearly switches off under centered symmetric padding, while the fine correction remains active or grows. This is most severe for v2.1 and explains much of its collapse. The broad calibration path has learned a strong dependency on the spatial/padding layout and is not translation-neutral.

## DACL10K valid-core halo audit

The valid-core path uses 1024 input tiles with a 128-pixel symmetric replicate halo, retains a 768-pixel valid core, advances by stride 768, and Hann-blends valid-core overlaps back into the original image geometry. No GT is resized.

| Model | Subset | Current Hann P-AP | Valid-core P-AP | Delta |
|---|---|---:|---:|---:|
| Row 0 | Bridge4 | 47.988205 | 43.408021 | -4.580184 |
| Row 0 | Crack | 5.527643 | 4.825730 | -0.701913 |
| Row 0 | Spalling | 45.737490 | 40.238684 | -5.498806 |
| Row 0 | Corrosion | 33.105355 | 28.684405 | -4.420950 |
| Row 0 | Efflorescence | 16.727113 | 13.289083 | -3.438030 |
| Row 0 | Macro Bridge4 | 25.274400 | 21.759476 | -3.514924 |
| Row 0 | All mapped | 43.281257 | 41.100152 | -2.181105 |
| Row 0 | Unseen | 27.152585 | 26.118102 | -1.034483 |
| v2.0 | Bridge4 | 40.877512 | 35.273588 | -5.603924 |
| v2.0 | Crack | 3.302604 | 2.989482 | -0.313122 |
| v2.0 | Spalling | 39.277940 | 30.697552 | -8.580389 |
| v2.0 | Corrosion | 29.693848 | 23.660209 | -6.033640 |
| v2.0 | Efflorescence | 13.201312 | 9.275637 | -3.925676 |
| v2.0 | Macro Bridge4 | 21.368926 | 16.655720 | -4.713206 |
| v2.0 | All mapped | 36.357685 | 35.233163 | -1.124522 |
| v2.0 | Unseen | 21.959123 | 22.172319 | +0.213196 |
| v2.1 | Bridge4 | 42.632422 | 35.203481 | -7.428941 |
| v2.1 | Crack | 3.838138 | 3.248170 | -0.589969 |
| v2.1 | Spalling | 40.776064 | 30.343198 | -10.432866 |
| v2.1 | Corrosion | 29.283817 | 21.157005 | -8.126813 |
| v2.1 | Efflorescence | 13.228054 | 8.474691 | -4.753363 |
| v2.1 | Macro Bridge4 | 21.781518 | 15.805766 | -5.975753 |
| v2.1 | All mapped | 37.509051 | 35.412537 | -2.096513 |
| v2.1 | Unseen | 22.500454 | 22.173458 | -0.326996 |

Valid-core tiling used 8,405 tiles versus 6,396 in the current path (`+31.4%`); per-image tile counts were min/mean/max `1 / 8.6205 / 48`.

### Existing Hann edge/center diagnostic

| Model | Edge P-AP | Center P-AP | Overlap P-AP | Non-overlap P-AP |
|---|---:|---:|---:|---:|
| Row 0 | 24.246584 | 51.114887 | 50.365969 | 44.826335 |
| v2.0 | 16.309007 | 44.633822 | 43.706006 | 37.142937 |
| v2.1 | 16.727749 | 46.937158 | 45.719641 | 38.557462 |

Tile edges are harder than centers, but structural-model degradation is present in both. Relative to Row 0, v2.0 loses `7.94` P-AP at edges and `6.48` at centers; v2.1 loses `7.52` at edges and `4.18` at centers. The valid-core halo path makes every Bridge4 result worse, especially v2.1, so missing tile-edge context is not the primary explanation for DACL10K failure. The changed tile alignment, greater replicated-boundary exposure, and loss of the current full-tile averaging behavior are all more plausible contributors.

## Frozen conclusions

1. **The current results are real and reproducible.** Protocol A exactly reproduces the prior formal results.
2. **Canvas under-fill materially hurts Row 0.** Scaling improves Row 0 by `+10.89` P-AP on CamCrack789 and `+8.29` on Crack500.
3. **Under-fill does not explain the structural-model behavior.** Scaling shrinks the CamCrack789 advantage but preserves a substantial morphology benefit; it does not repair the Crack500 structural-model deficit.
4. **Padding placement is not neutral.** Centered symmetric padding often increases raw skeleton/component recall while degrading AP and calibrated morphology metrics. The v2.1 broad path is particularly position-sensitive.
5. **DACL10K tile boundaries are only a secondary factor.** Edge performance is weaker, but center degradation remains and valid-core halo inference is worse rather than better.
6. **The most defensible external-generalization diagnosis remains domain/calibration sensitivity.** Effective scale matters for Row 0, while texture, annotation style, padding layout, and learned broad-correction calibration dominate the structural models' failures.

No checkpoint, architecture, or external-test-tuned hyperparameter is changed by this audit. Any subsequent scale augmentation, translation/padding augmentation, or calibration experiment must be registered as a new training experiment and evaluated without tuning on these same external test labels.
