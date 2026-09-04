# BridgeAdaptCLIP-v2.1 Results

## Protocol

- Fine expert: validation-selected v2.1-Fine Epoch 9
- Broad head: validation-selected v2.1 Epoch 11
- CLIP input: 518x518
- Structural input and metric resolution: native 1024x1024
- Reference mode: zero-reference
- Selection: overall Validation P-AP
- Test: one run after checkpoint selection

## Validation selection

The v2.1-Fine Phase-1 checkpoint was Epoch 9 with P-AP 76.900672. After
retraining the unchanged v2.0 Broad Head, Epoch 11 was selected with P-AP
79.832707, P-AUROC 96.344947, and P-F1max 79.324266.

## Test comparison

| Metric | v2.0 | v2.1 | Delta |
|---|---:|---:|---:|
| I-AUROC | 93.523628 | 93.523628 | 0.000000 |
| I-AP | 94.509691 | 94.509691 | 0.000000 |
| I-F1max | 84.974092 | 84.974092 | 0.000000 |
| P-AUROC | 95.994931 | 95.304825 | -0.690106 |
| P-AP | 74.811347 | 75.478776 | +0.667429 |
| P-F1max | 74.070874 | 74.863115 | +0.792241 |
| Macro diagnostic P-AP | 57.503248 | 57.420477 | -0.082770 |
| Crack P-AP | 40.062343 | 39.834205 | -0.228138 |
| Spalling P-AP | 77.536687 | 79.087761 | +1.551073 |
| Corrosion P-AP | 62.283338 | 61.841000 | -0.442338 |
| Efflorescence P-AP | 50.130623 | 48.918944 | -1.211679 |

## Interpretation

Multi-level frozen CLIP guidance improves the primary overall P-AP and
P-F1max, with the largest class gain on Spalling. It does not dominate v2.0:
pixel AUROC, macro P-AP, Crack, Corrosion, and Efflorescence are lower. Image
metrics remain exactly unchanged as required. Therefore v2.0 remains the more
balanced stable model, while v2.1 is the overall-P-AP/P-F1max alternative.

The v2.1 Broad gate remains FP-sensitive (FP-like/correct-background gate ratio
1.730709; gate/FP-target correlation 0.214428), but applies stronger global
background suppression than v2.0. This helps global AP and Spalling while
partially eroding the Fine expert's gains on smaller or texture-like classes.
