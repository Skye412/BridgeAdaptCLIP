# BridgeAdaptCLIP-v2.0

BridgeAdaptCLIP-v2.0 freezes the validation-selected v1.3 Epoch 3 model as a
fine-localization expert and trains only a low-resolution broad calibration
head. The head consumes the frozen 256x256 joint feature after average pooling
to 128x128, plus downsampled v1.3 and Row-0 probabilities.

Its correction is constrained to be non-positive:

```text
C_b = -sigmoid(A_b) * softplus(R_b)
Z_final = Z_fine + C_b
```

The broad gate target is the frozen v1.3 background score `(1-Y)*P_fine`.
Training uses final Focal and Dice, `0.1` broad gate BCE, `0.05` positive
preservation, and `0.01` negative-only ranking against detached v1.3 positive
references. All semantic and v1.3 fine components remain frozen. Checkpoint
selection uses overall validation native-1024 P-AP only.
