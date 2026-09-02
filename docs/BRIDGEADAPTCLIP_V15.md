# BridgeAdaptCLIP-v1.5

BridgeAdaptCLIP-v1.5 is a single-variable experiment derived from v1.3. The
architecture, frozen Row-0 semantic base, optimizer, learning rate, batch,
initialization, and selection protocol are unchanged. The only addition is an
AP-aligned hard-pixel ranking loss with weight `0.01`.

For each defect image, the loss selects the 256 lowest final logits inside the
GT defect mask and the 256 highest final logits in GT background. If fewer than
256 positive pixels exist, all positive pixels are used. Normal images do not
participate in pairwise ranking.

The ranking objective is:

```text
mean softplus(Z_hard_negative - Z_hard_positive)
```

It is applied to final `output["mask_logits"]`. v1.3 signed correction remains
at `0.05`; the v1.4 final-logit margin is disabled.
