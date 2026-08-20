# BridgeAdaptCLIP-v1.1 frozen-semantic gated residual

## Frozen research question

Can native-resolution structural evidence improve thin-crack localization while
preserving the formal Protocol-v2 Row-0 semantic prediction?

## Controlled change

The formal Row-0 Epoch 14 CLIP, VisualAdapter, TextualAdapter, original prompt
ensemble, pixel prediction, and image score are loaded and frozen. The retained
1024 Conv Stem and DEGConv-lite produce structural features. A spatial gate and
bidirectional residual head observe `F_high`, frozen `F_sem`, and `P_row0`.

The only pixel-output change is:

```text
Z_final = logit(P_row0) + sigmoid(G) * R
```

The residual head is zero initialized. The gate head has zero weights and bias
`-4`. Therefore the untrained v1.1 output is exactly Row 0. `R` has no sigmoid
and may suppress false-positive texture as well as enhance missed cracks.

## Fixed training and evaluation protocol

- Train/validation/test: frozen Bridge2893 parent-disjoint seed-42 split.
- CLIP input: 518; structural input and native GT: 1024.
- Trainable: structural stem, DEGConv-lite, semantic projection, joint
  projection, residual decoder/head, and gate head only.
- Loss: FP32 Focal + Dice. The first controlled run fixes gated-residual L1 to
  zero.
- Adam `(0.5, 0.999)`, LR `3e-4`, no weight decay, effective batch 8, seed 10.
- Select one checkpoint by validation native-1024 P-AP; run test once.
- Report overall image/pixel metrics, four defect diagnostics, macro diagnostic
  P-AP, mean gate, and mean absolute gated residual.

The validation-selected fixed 0.5 Row0/Full-v1 fusion (test P-AP 72.70) is a
diagnostic complementarity control, not a training target and not used to tune
v1.1 on test.
