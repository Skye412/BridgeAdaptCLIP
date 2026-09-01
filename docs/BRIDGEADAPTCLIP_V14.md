# BridgeAdaptCLIP-v1.4

BridgeAdaptCLIP-v1.4 is a single-variable experiment derived from v1.3. The
architecture, frozen Row-0 base, initialization, optimizer, learning rate,
batching, gate loss, preservation loss, data, seed, and evaluation protocol are
unchanged. The structural branch is freshly initialized; v1.3 is not loaded.

The only controlled change is:

`0.05 * signed_correction_loss -> 0.05 * final_logit_margin_loss`

For `E = stopgrad(abs(Y - P0))`, margin `m = 1`, and final model output
`Z_final = output["mask_logits"]`:

- FN term: `sum(Y*E*softplus(m-Z_final)) / sum(Y*E)`
- FP term: `sum((1-Y)*E*softplus(m+Z_final)) / sum((1-Y)*E)`
- margin loss: `0.5 * FN + 0.5 * FP`

The margin loss never supervises the raw residual or gated residual in place of
the final logits. Checkpoint selection remains overall Validation native-1024
P-AP, followed by exactly one formal Test evaluation.
