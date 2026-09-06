# DeRCLIP paper naming and comparison protocol

The default paper method is **DeRCLIP — Decoupled Residual Adaptation of CLIP**.
Engineering names and checkpoint keys remain unchanged for reproducibility.

| Engineering result | Paper name | Role |
|---|---|---|
| Original AdaptCLIP Row 0 | AdaptCLIP-BD | Bridge-domain trained inherited baseline |
| BridgeAdaptCLIP-v1.3 | DeRCLIP-F | Fine structural correction only |
| BridgeAdaptCLIP-v2.0 | DeRCLIP | Default complete method |
| BridgeAdaptCLIP-v2.1 | DeRCLIP-ML | Multi-level-guidance variant |

The three first-level components are FSA (Frozen Semantic Anchor), ESC
(Error-aware Structural Correction), and BSC (Background Score Calibration).
The former DEGConv-lite block is described as SGE (Strip-Gated Enhancement).
SGE is inspired by MixerCSeg DEGConv; renaming does not remove the citation or
the obligation to state the architectural differences.

The primary task is supervised binary localization of the union of Crack,
Spalling, Corrosion, and Efflorescence. “0-reference” describes inference and
does not imply zero-shot use of Bridge2893 training images or masks.

All comparisons use the parent-isolated seed-42 split, continuous prediction
scores, frozen original 1024 raster GT, validation overall P-AP checkpoint
selection, and one test execution after selection. No external data are added.
The seed-42 test set has participated in method development and must not be
described as an untouched blind test.

Core controlled ablations:

| ID | Control |
|---|---|
| A1 | Semantic-only residual refiner; structural tensor is exactly zero |
| A2 | Two independent 3x3 depthwise paths replace the 1x5/5x1 SGE paths |
| A3 | Gate supervision weight is zero |
| A4 | Fine preservation weight is zero |
| B1 | BSC positive preservation weight is zero |
| B2 | BSC negative-only ranking weight is zero |

A1/A2 start from a shared compatible seed-10 initialization. A3/A4 retain the
same architecture and only change the named loss weight. B1/B2 load the exact
same frozen DeRCLIP-F checkpoint and freshly initialize BSC.
