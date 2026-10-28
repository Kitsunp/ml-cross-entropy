# Finalizing accumulated language-model metrics

`finalize_lm_metrics` is an explicit reporting-time API. It finalizes detached
CCE/MiLe and Top-1/3/5 sufficient statistics without another vocabulary
projection or dense logits. CUDA uses one Triton kernel with one warp. Other
devices have a Torch reference; ordinary CCE forwards do not invoke it.

For valid target count `N`, accumulate `N * ntp_ce_unweighted` and
`N * (ntp_ce_unweighted + mile_reweighting_delta)` over microbatches. Exclude
mu-loss and other auxiliary losses. Accumulate the existing `topk_counts`
returned by `return_topk=True`. Keep ordinary-token and patch K>1 totals
separate; do not collect ordinary-token Top-k for K>1 patches.

`return_topk=3` requests only Top-1/3 and reduces candidate workspace from
five keys to three. Its compatible `topk_counts` vector is `(hits1,hits3,0,N,bad)`;
the hits5 slot and final accuracy5 slot are inactive placeholders. Do not
publish them as a measured zero accuracy. In this mode validate only
`0 <= hits1 <= hits3 <= N`, plus a zero placeholder and zero nonfinite flag.
The PPL finalizer and its wire format need no change.

Use FP64 scalar accumulators from the start of the interval to preserve large
counts. Sum them across workers before finalization. The helper neither runs
a collective nor changes the model's arithmetic/gradients.

```python
import torch
from cut_cross_entropy import finalize_lm_metrics

# All inputs below are detached scalar tensors already accumulated on device.
# token_sums/patch_sums: (sum_CE, sum_MiLe_objective, valid_targets).
# topk_sums: (hits1, hits3, hits5, valid_targets, nonfinite_targets).
# Use device zeros for inactive phases and disabled Top-k.
totals = torch.cat((token_sums, patch_sums, topk_sums)).to(torch.float64)
# If distributed: perform SUM on totals here, before normalizing.
packed = finalize_lm_metrics(totals)
# One optional host copy at the reporting boundary, not inside the helper.
values = packed.cpu().tolist()
```

Input is a contiguous, detached FP64 vector of length 11; output is a new FP64
vector of length 22 on the same device:

| Slice | Contents |
|---|---|
| `0:11` | Original totals, unchanged, for validation and reporting counts |
| `11:15` | Token CE, PPL, MiLe objective, exp-MiLe |
| `15:19` | Patch CE, exp-CE, MiLe objective, exp-MiLe |
| `19:22` | Ordinary-token accuracy at 1, 3, 5 |

PPL is `exp(sum_CE / sum_N)`. Averaging per-microbatch PPLs is incorrect.
Exp-MiLe describes microbatch-normalized weighted objectives and is **not**
conventional autoregressive perplexity. Patch exp-CE is also reported
separately. MiLe weights do not change the logits/rankings: Top-k is shared,
not recalculated as a post-MiLe variant.

Empty phases produce zero placeholders; omit their fields when publishing.
Validate raw counts as finite, nonnegative integers, ensure
`hits1 <= hits3 <= hits5 <= topk_N`, and reject nonzero nonfinite-target flags.
When both ordinary-token CE and Top-k are collected, their target counts must
match. The kernel preserves raw values/health flags rather than sanitizing
them. Value checks belong at the reporting boundary to avoid a device-to-host
synchronization in the helper. Exponentials can overflow to infinity, as with
Python-float `exp`; scalar FP64 preserves its range, including CE=100.

The helper allocates 176 bytes of output payload; it does not allocate
per-token/per-vocabulary storage. This is not a whole-training VRAM or speed
claim: allocator pools, compiled graphs and pre-existing Top-k workspace have
their own costs. Compilation/capture setup is separate from steady execution.

Tests cover CPU/CUDA reference parity, large counts, empty phases, health
flags, exponential range, invalid inputs, fullgraph `max-autotune`, changed
data replay and a warm profiler check of three calls producing three kernels,
without `item`, host scalar reads or device-to-host copies in finalization.
