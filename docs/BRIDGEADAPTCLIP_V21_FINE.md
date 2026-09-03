# BridgeAdaptCLIP-v2.1-Fine

Phase 1 tests one controlled change to the v1.3 fine-localization model: frozen
CLIP patch tokens from layers 6, 12, and 18 are projected into zero-initialized
semantic residuals and added to the existing deep layer-24 semantic feature.

```text
F_ML = F_base + delta_6 + delta_12 + delta_18
```

Each shallow branch applies channel normalization, `1x1 768->128`, GroupNorm,
GELU, and a zero-initialized `1x1 128->128` projection. Consequently the initial
prediction is exactly the standard Row-0-initialized v1.3 fine architecture.
The CLIP backbone, Row 0 Visual/Textual adapters, prompt ensemble, image score,
1024 structural branch, DEGConv-lite, gate, residual heads, and v1.3 losses are
otherwise unchanged.

Phase 1 trains 15 epochs and selects one checkpoint using overall Validation
native-1024 P-AP. It intentionally does not evaluate Test. BridgeAdaptCLIP-v2.0
remains the frozen stable model until the validation gate authorizes Phase 2.

The validation-selected checkpoint is Epoch 9 (P-AP 76.900672). Relative to
the v1.3 Fine validation reference, it improves overall P-AP by 0.079339,
macro diagnostic P-AP by 3.984756, Crack by 9.019956, Spalling by 3.597527,
and Corrosion by 4.493920, while Efflorescence decreases by 1.172379. This
satisfies the predeclared Phase-2 admission rule. Phase 2 freezes this Fine
checkpoint and retrains the unchanged v2.0 Broad Head from initialization.
