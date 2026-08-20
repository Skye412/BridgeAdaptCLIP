# BridgeAdaptCLIP-v1.3

BridgeAdaptCLIP-v1.3 is a controlled single-variable extension of v1.2. The
network, frozen Row-0 semantic base, initialization, optimizer, learning rate,
gate loss, preservation loss, data, seed, and checkpoint protocol are unchanged.

The only addition is a signed, Row-0-error-weighted correction loss. For
`C = G * R` and `E = stopgrad(abs(Y - P0))`, it balances:

- defect correction: `Y * E * softplus(-C)`, encouraging `C > 0`;
- background correction: `(1-Y) * E * softplus(C)`, encouraging `C < 0`.

Each direction is normalized per image and averaged only over images with valid
support. The fixed loss weight is `0.05`.

The formal checkpoint is selected only by overall Validation native-1024 P-AP.
Test is evaluated once after selection. Gate/correction diagnostics never select
the checkpoint. The primary mechanism check is whether FN-like mean signed
correction changes from negative to positive without sacrificing v1.2's gate
selectivity.
