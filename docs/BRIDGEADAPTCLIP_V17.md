# BridgeAdaptCLIP-v1.7

BridgeAdaptCLIP-v1.7 is a single-variable gradient-routing experiment derived
from v1.5. It keeps v1.5 uniform hard-pixel sampling (256 lowest GT-positive
and 256 highest GT-negative final logits per defect image) and changes only
which selected logits receive ranking gradients.

The positive-raising and negative-suppressing terms are:

```text
L_raise    = softplus(stopgrad(Z_hard_negative) - Z_hard_positive)
L_suppress = softplus(Z_hard_negative - stopgrad(Z_hard_positive))
L_rank     = 0.8 * L_raise + 0.2 * L_suppress
```

The forward scalar equals the original v1.5 pairwise ranking scalar because
both terms have the same numerical value and their weights sum to one. The
controlled change is therefore gradient allocation: 80% raises missed defect
pixels and 20% suppresses hard background pixels.

All architecture, frozen Row-0 base, initialization, optimizer, learning rate,
batching, 15-epoch budget, seed, losses, and native-1024 Protocol-v2 evaluation
remain unchanged from v1.5. Skeleton-balanced sampling and margin loss are
disabled. Validation records both overall and per-class metrics, while checkpoint
selection uses overall validation P-AP only.
