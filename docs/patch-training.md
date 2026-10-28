# Graph-stable patch-level training

Patch-level training is an opt-in CCE path for predicting a fixed group of
future tokens from one hidden-state row.  It follows the patch objective from
[Beyond Next Token Prediction: Patch-Level Training for
LLMs](https://arxiv.org/abs/2407.12665) without materializing a
`rows × patch_size × vocabulary` logit tensor.

## Public contract

### Original-sequence curriculum comparison

The existing real-data runner also supports a fixed **raw** length of 512:
`--patch-size 4 --patch-steps 67` keeps every original example intact, uses
128 core positions for updates 1–67 and 512 positions for updates 68–100.
It does not concatenate examples. Padding-aware means count only valid slots;
empty patches are masked. Validation remains 100 ordinary token-level batches.

`--patch-size 4 --patch-steps 0` is a matched token-only control using the same
row permutation and fixed target-slot contract. It is not an untouched legacy
CCE benchmark. Compare each architecture with its own matched control, rather
than attributing architectural differences to the curriculum.

This fixed-raw policy differs from the paper's fixed-core `K*T -> T` batching
below: raw reserved slots per update are equal in both phases, so a 67/33 step
split has no extra factor of K in its reserved-slot data accounting. Valid raw
token counts can still differ with padding. The boundary is completed optimizer
updates, not an adaptive token counter; continuous optimizer/scheduler history
is a deliberate policy rather than the paper's reset.

The opt-in runner requires a compatible Trainer prewarm API. It warms both
training signatures and ordinary evaluation, rolls model/master/moments,
stabilizer, scheduler and RNG back in place, and checks compilation/capture/JIT
counts before and after transition, training and validation. Warmup, initial
training and transition traces are separate private artifacts. Cold and profiled
updates are excluded from stable timings, which are also reported by curriculum
phase. Neither these controls nor the presence of a profile proves a speed win;
the full real-data comparison must pass first. Ordinary validation has its own
private execution trace and batch latencies. Report the 100-update/100-batch
wall totals separately from compute sums and stable medians: trace export,
logging, dataset iteration and host instrumentation affect wall time.

`--control-results <private-control.json>` reuses a completed matched K1
control for memory and total-time deltas. Token recovery compares common
unprofiled optimizer-step indices after transition. Source hashes, CCE tree,
runtime, data and metric policies must match. A single sequential comparison
does not establish statistical equivalence, and a matched K1 control does not
replace the separate ordinary-training baseline. `--no-perplexity-metrics`
is an explicit metrics-overhead control, not an optimization to use silently.
The runner rejects impossible phase-specific target counts and nonfinite
CE/MiLe telemetry before accepting a run or reusing a control. A historical
`status=ok` without these semantic checks is insufficient evidence by itself.

A sequential real-data K1-versus-67/33 comparison on the runtime described
below measured 28.103 versus 20.223 seconds for the 100-update training
window (-28.04%). Stable K4 compute was 81.138 ms/update and recovered K1
compute 238.986 ms/update. Recovery uses the same stable update indices in
the control, not its earlier median. The 100-batch ordinary validation
windows were 12.878 versus 12.988 seconds. Both raw shapes were prewarmed
and no post-warmup compilation growth occurred. This is a speed result,
not equal-quality evidence; short token adaptation did not erase the
validation-loss gap. Other architectures still need mode-matched pairs.

Memory must distinguish allocated tensors from allocator reservations. In
that comparison, the training allocated peak was 3.401 GB in the K1-only
control, 2.400 GB during K4 (-29.43%) and 3.405 GB after returning to K1
(+0.10% versus control). Reserved memory was 23.763 GB in the control and
25.503 GB in both curriculum phases (+7.33%). These are decimal GB. The
higher reservation already existed after prewarming both execution shapes;
it was not a new 7.33% tensor allocation at transition. Resetting peak
statistics does not release cached or captured pools. The aggregate measured
reservation does not identify how many bytes belong to each retained pool.

A separate JToK matched pair also completed 100 updates and 100 token-level
validation batches. K1 versus 67/33 training wall time was 32.962 versus
23.433 seconds (-28.91%); validation wall time was 14.485 versus 14.561
seconds. Stable patch compute was 95.880 ms/update; recovered token compute
was 284.371 ms/update, versus 290.039 ms on common control update indices.
There were no post-prewarm graph breaks, recompiles or compilation/capture
growth. Training allocated peaks were 3.410 GB for the control, 2.417 GB
during patches and 3.413 GB after transition; reservation increased by 5.77%.
Ordinary validation CE was higher after the short curriculum, so this is
execution/performance evidence, not equal-quality evidence. The corresponding
full JToK-M matched pair is still pending.

### Independent monitoring switches

The public CCE API keeps telemetry opt-in: `return_loss_metrics=True`
exposes detached CE/MiLe components, and `return_topk=True` additionally
requests full-token Top-1/3/5 counts. Neither switch changes the loss or its
gradient. Applications can independently disable PPL aggregation and Top-k
collection before compilation; do not change these switches during replay.
The upstream integration uses `USE_PERPLEXITY_METRICS` and `USE_TOPK_METRICS`
and enables both for the requested monitoring setup. Those private source
files are not distributed here.

Aggregate CE as `sum(N_batch * CE_batch) / sum(N_batch)` and exponentiate
once for ordinary-token PPL. Report `exp(MiLe objective)` separately as
`mile_ppl`; it is not conventional unweighted perplexity. Exclude auxiliary
losses and mu regularization. During K>1, label these values `patch_exp_ce`
and `patch_mile_exp_loss`, not autoregressive PPL, and do not collect token
Top-k. Ordinary training and validation remain monitorable with patching off.
The measured Top-k overhead below must not be described as negligible.

[`finalize_lm_metrics`](metrics.md) performs the final divisions/exponentials
on device after interval accumulation and the distributed sum. It reuses these
existing components/counts; it does not calculate batch PPL on every forward
or average batch PPLs.

### Ordinary-token Top-1/3/5 diagnostics

To collect only Top-1/3, use `return_topk=3`. Both lock and split kernels
specialize selection, merge networks and candidate allocations to three keys;
ranks 4/5 are not selected or counted. `True` and `5` preserve the legacy
Top-1/3/5 behavior. The five-scalar count layout is unchanged: its hits@5 slot
is zero when inactive, and must not be reported or used for ordering checks.
Validate `0 <= hits1 <= hits3 <= N` in this mode. `SUPPORTED_TOPK` exposes
the active ranks supported by the installed package.

`return_topk=True` with `return_loss_metrics=True` adds detached counts
`(hits@1, hits@3, hits@5, N, nonfinite_targets)` to CCE's metrics dictionary.
For k in {1,3,5}, accuracy is `sum_i 1[y_i in Top_k(z_i)] / N`, over **all**
valid supervised token targets, not a token sample. Shifted-out and ignored
targets are excluded. Equal rounded logits prefer the lower vocabulary ID.
CE and MiLe share these unweighted accuracies; MiLe does not reweight the
counts. Input corruption changes the conditioning context and must be
disclosed when comparing training and clean validation.

The native forward epilogue selects candidates from CCE's existing rounded
logits; it adds neither a second vocabulary projection nor dense logits.
It still costs reductions, registers, workspace and time. The lock path
stores five winners per CCE row (after compaction, when used), plus small
count partials. When logits
are FP16/BF16, the vocabulary has at most 65,536 entries, bias is absent or
has the same dtype, and softcap is absent, global score/ID keys also fit
32 bits (20 bytes/row). Other cases retain 64-bit keys (40 bytes/row).
The split path additionally stores winners per row per split.
Row count need not equal the number of supervised targets for fixed-slot
inputs. These workspace sizes do not predict allocator or capture-pool peaks.
Consumers must reject nonzero `nonfinite_targets` rather than interpreting
invalid rankings as an accuracy measurement.

For actual FP16/BF16 logits, local selection losslessly packs the ordered
16-bit score and a 16-bit tile-local tie-break ID into a single 32-bit key.
For eligible vocabularies, global keys use the same exact packed score and
global ID representation; otherwise only the five winners widen to 64 bits.
This avoids paired score/index reductions without introducing quantization.
The decision uses the logit dtype **after** bias and softcap: an FP32 bias
or FP32 softcap result retains exact FP32 selection instead of assuming the
embedding dtype. Signed zeros tie; padded columns never win.

Merging two descending candidate lists uses the exact zero-based identity
`C[r] = max(A[r], B[r], max_{i=0..r-1} min(A[i], B[r-1-i]))`.
For five outputs, this requires 25 min/max operations instead of 45 serial
insertion comparisons, shortening the lock's dependency chain. This is an
algorithmic work reduction, not by itself evidence of faster real training.
Resource and latency checks must cover the selected lock/split backend;
improvements to one launch layout do not establish a win for another.

The curriculum integration collects these metrics only in ordinary token
training and validation, including the raw-K1 phase, **not K>1 patches**.
`--no-topk-metrics` is the matched real-data overhead control. Compare both
100-update and 100-validation-batch timing/memory totals and stable latencies
before calling the diagnostics cheap. Do not change the flag mid-run to
conceal its cost or create an unwarmed compilation signature.

One sequential real-data comparison on RTX 5090 with Torch 2.14.1/CUDA 13.2
and Triton 3.8 completed 100 optimizer updates and 100 token validation
batches per case. PPL stayed enabled; the same private model/training inputs
were used. The compact-key candidate changed only the investigated CCE
implementation, not the model sources. These are full-flow timings, not
public model traces or an equal-quality/scaling claim:

| Ordinary-token case | Stable training ms/update | Stable updates/s | Validation median ms/batch | Train wall s | Validation wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-k off | 219.443 | 4.557 | 111.432 | 25.945 | 12.287 |
| Top-k on, 64-bit global keys | 241.310 | 4.144 | 125.133 | 28.165 | 13.615 |
| Top-k on, eligible 32-bit global keys | 240.617 | 4.156 | 117.513 | 28.103 | 12.878 |

The compact-key candidate still adds 9.65% training and 5.46% validation
median latency versus metrics-off: the diagnostics are **not free**. Its
0.29% training difference versus the preceding candidate is not a robust
speedup claim. Measured allocated peaks increased by 512 bytes and reserved
peaks were unchanged; that does not negate transient workspace costs.
All valid validation targets were counted, with no post-prewarm graph
breaks or recompiles. Validation used disjoint training-source rows, not
the original held-out split. Kernel-only profiling attributes the remaining
cost to fused selection/merge rather than the final counts reduction.

Moving candidate selection before FP32 loss-statistic widening was tested
and rejected: the same real 100+100 contract regressed to 297.962 ms/update
(3.356 updates/s), versus 240.617 ms with later selection. Validation stayed
effectively unchanged. The representative lock specialization reported fewer
registers but more compile-time spills; the profiled CCE forward also became
slower. This is not a measurement of dynamic spill traffic or an isolated
causality proof. The implementation retains the earlier validated schedule;
native/widened and forced-subnormal correctness regressions remain in tests.

### Why patch CE can be higher without implying worse token modeling

For one patch with n valid targets, all targets share one predicted distribution
q. Let r(v) be the fraction of these targets equal to vocabulary item v. Then

```text
L_patch = -sum_v r(v)*log(q(v)) = H(r) + KL(r || q) >= H(r)
dL_patch/dlogits(v) = q(v) - r(v)
```

This is the ordinary unweighted CE, not an additional auxiliary loss. If all
n targets are distinct, H(r)=log(n); if all are identical, H(r)=0. Padding and
ignored labels do not enter n or r. Thus log(K) is an illustrative floor for
a fully valid, distinct-target patch, not a universal additive penalty versus
token CE. Token prediction uses separate distributions and preceding tokens
within the group. Sharing predictions and omitting that within-patch context
can make the patch task harder. It does not guarantee a measured ordering of
losses from different, nonoptimal models or training steps.

Measure this effect with CE per valid supervised target, excluding MiLe,
mu and other auxiliary losses. exp(patch CE) is not conventional autoregressive
perplexity. MiLe is a separately reported weighted objective; its gradient
and lower bound cannot be inferred from the unweighted formula above. Compare
ordinary token-level validation after transition to assess token modeling,
and downstream tasks separately to assess transfer. Neither a higher patch
training loss nor a lower token training loss alone establishes transfer quality.

### Paper-aligned practical size and duration

The archived [paper, v2](https://arxiv.org/pdf/2407.12665v2), sections 3.1,
3.5 and 3.6, uses K=4 as its main efficiency/quality trade-off. K=2 and K=4
had nearly matching adaptation curves in the size ablation; K=8/16 degraded
quality under that comparison. Its main compute-limited setting uses a
patch-data fraction lambda=2/3, not a universal optimum. The implementation
target is therefore K=4 followed by K=1, without learned compression maps.

The fraction is measured in raw training data. Do not directly translate it
into a fraction of optimizer steps when raw tokens per update differ. For
fixed core batch B and length T, one patch microbatch consumes K*B*T raw slots,
while one token microbatch consumes B*T. With equal accumulation in both
phases, the patch step fraction q must satisfy

```text
lambda = K*q / (K*q + 1-q)
q = lambda / (K*(1-lambda) + lambda)
```

Thus K=4 and lambda=2/3 correspond to q=1/3 under that particular batching
policy, not q=2/3. Count actual valid raw tokens when padding varies. Matching
raw tokens per optimizer update instead requires explicit batch/accumulation
accounting and warmup of its complete execution paths; changing accumulation
also changes update dynamics unless normalized consistently.

The paper resets optimizer and scheduler at transition. Keeping continuous
AdEMAMix state is a deliberate different policy, not a faithful reproduction
of that reset. Neither K=4 nor its reported compute proxy guarantees the same
wall-time savings or quality on another architecture or token budget.

Enable the path with `patch_training_enabled=True`.  The embedding has shape
`(..., D)` and targets have shape `(..., Kmax)`.  The leading dimensions must
match and `Kmax` must remain constant for the lifetime of the compiled graph.

```python
loss = linear_cross_entropy(
    patch_hidden_states,          # [batch, patch_rows, hidden_dim]
    classifier,                  # [vocabulary, hidden_dim]
    patch_targets,               # [batch, patch_rows, Kmax]
    impl="cce_kahan_full_c",
    patch_training_enabled=True,
    mile_enabled=True,
    mile_gamma=1.0,
    mu_loss_enabled=True,
    mu_loss_lambda=1e-4,
)
```

The flag is deliberately static.  Do not turn it off at the patch-to-token
transition. Keep the same embedding/target leading dimensions and the same
`[..., Kmax]` target tensor, then represent inactive slots with `ignore_index`:

```python
# Patch phase: Kmax valid targets per row.
patch_targets = next_patch_ids

# Token phase: one valid target, Kmax - 1 inactive slots. Shape is unchanged.
token_phase_targets = torch.full_like(patch_targets, -100)
token_phase_targets[..., 0] = next_token_ids
```

Only tensor values change between phases.  The public signature, rank, shape,
K specialization and custom-op boundary remain unchanged, so the phase change
does not itself require Dynamo or Inductor to build another graph.
Changing the number of hidden rows or any other tensor shape can still trigger
ordinary shape specialization; the flag cannot make two different model input
signatures share one compiled graph.

## Controlling the patch duration

`PatchTrainingSchedule` exposes the phase boundary to the trainer while keeping
the optimizer step out of the kernel and compiled model:

The selected control policy is **optimizer updates**, not a consumed-data
threshold. `global_step` counts completed optimizer updates before the next
update. Keep it unchanged across that update's gradient-accumulation
microbatches and restore it from the checkpoint on resume. Track raw valid
tokens separately for accounting; neither variable batch sizes nor padding
change the configured boundary. This stateless helper does not itself observe
whether an optimizer update ran: the trainer must supply the correct count.

The cutoff below is an illustrative explicit step budget, not a default and
not an automatic conversion of the paper's data fraction. For a 100-update
transition check, choose and record a cutoff strictly between 0 and 100 so
both phases execute; validation batches do not advance this counter.

```python
from cut_cross_entropy import PatchTrainingSchedule, linear_cross_entropy

schedule = PatchTrainingSchedule(
    patch_training_steps=120_000,
    patch_size=4,
    ignore_index=-100,
)

# global_step = completed optimizer updates (unchanged during accumulation).
# Run this branch in the trainer/data pipeline, before the compiled core.
phase = schedule.phase(global_step)
if phase == "patch":
    # raw_input_ids/labels contain K * T tokens.
    T = raw_input_ids.size(-1) // schedule.patch_size
    token_embeddings = token_embedding(raw_input_ids)
    core_inputs = token_embeddings.unflatten(-2, (T, schedule.patch_size)).mean(-2)
    next_patch_ids = labels[..., schedule.patch_size :].unflatten(
        -1, (T - 1, schedule.patch_size)
    )
    targets = schedule.targets_for_step(
        global_step,
        patch_targets=next_patch_ids,  # [..., T - 1, K]
    )
else:
    # Token input/labels contain T tokens.
    core_inputs = token_embedding(raw_input_ids)
    targets = schedule.targets_for_step(
        global_step,
        token_targets=labels[..., 1:],  # [..., T - 1]
    )

# The compiled core receives the same [..., T, D] input shape in both phases.
hidden_states = compiled_transformer(core_inputs)[..., :-1, :]
loss = linear_cross_entropy(
    hidden_states,
    classifier,
    targets,
    patch_training_enabled=True,  # deliberately unchanged across phases
)
```

Steps are zero-based. With `patch_training_steps=120_000`, steps 0 through
119,999 are patch-level and step 120,000 is the first token-level step.
`is_transition_step(global_step)` identifies that boundary so the trainer can
save model parameters and recreate optimizer/scheduler state when exact paper
reproduction is desired. The schedule does not reset training state silently.
It is not a one-shot event: it remains true for every microbatch sharing the
boundary count. Any boundary action must run once at the optimizer-update
boundary, not once per microbatch. Continuous training keeps optimizer and
scheduler state; no reset is implied by switching K.

The phase branch must remain outside the compiled core. Passing `global_step`
through `linear_cross_entropy` or switching `patch_training_enabled` at the
boundary would introduce value guards or another graph. Both phases feed the
Transformer the same T embedding rows and CCE the same T - 1 aligned hidden
rows: the paper uses `K * 2048` raw tokens followed by embedding averaging in
the patch phase and 2048 raw token rows in the token phase. Position IDs and
attention masks must likewise be rebuilt for T rows before the compiled core.

`labels[..., K:].unflatten(...)` is a non-contiguous view whose leading stride
still includes the skipped patch. `prepare_patch_targets` normalizes that small
target tensor to the same contiguous layout produced by
`prepare_token_targets`. Without this normalization, shapes and dtypes match
but Dynamo still specializes a second graph on stride/storage offset. Already
contiguous patch targets are returned without a copy.

## Relationship to the paper

The paper's complete method contains model/data responsibilities in addition to
the loss implemented here:

1. Every consecutive group of K token embeddings is averaged, without adding
   projection parameters. A raw context of `K * T` tokens therefore becomes T
   patch rows before entering the otherwise unchanged sequence model.
2. A single ordinary output head predicts all K tokens of the next patch. The
   authors explicitly avoid K separate heads so the patch model remains aligned
   with the later token model.
3. The paper keeps the global raw-data batch per update fixed by adjusting
   gradient accumulation. Under that policy, patch training runs for
   `N * lambda` updates and token training for `N * (1 - lambda)` updates.
   Only model parameters initialize the second stage; the paper resets
   optimizer and learning-rate-scheduler state at the boundary. These step
   proportions do not transfer unchanged to equal-accumulation, fixed-core
   batching described above.

CCE implements item 2's multi-target loss and its memory-efficient backward. It
does not average token embeddings, construct patch-aligned labels, choose the
stage boundary, or reset training state. Those operations remain explicit in
the model/data/trainer pipeline, preventing a loss flag from silently changing
the architecture or optimizer.

The main experiments used `K=4` and `lambda=2/3`, giving the theoretical cost

```text
lambda / K + 1 - lambda = (2/3) / 4 + 1/3 = 0.5
```

for 180,000 total steps. This corresponds to 120,000 patch steps followed by
60,000 token steps. Their measured eight-A100 patch stage was 3.50x faster
after reducing gradient accumulation to preserve global batch size, and the
reported total runtime was approximately 0.523x rather than the theoretical
0.5x because of data-loading and gradient-synchronization overhead.

The supplied official `PatchTrain-main` example uses 90,000 patch steps and
45,000 token steps instead. That example trains on `pile-uncopyrighted`, which
the authors describe as containing roughly 25% fewer tokens than the original
Pile setup. It launches two separate training processes (`patch_size=4`, then
`patch_size=1` initialized from the first stage), which naturally resets the
optimizer and scheduler. The graph-stable CCE integration keeps K fixed inside
the loss instead, but the trainer must still change embedding aggregation and
training state at the same semantic boundary.

The paper's 370M-model ablation found K=2 and K=4 loss curves nearly identical,
while K=8 and K=16 degraded performance; K=4 was selected as the efficiency/data
trade-off. Under its fixed-compute lambda study, performance peaked around
`lambda=2/3`, whereas values above 3/4 left too little token-level adaptation.
These are experiment-specific observations, not a scaling law. The article
evaluated 370M to 2.7B parameters, so applying them to a roughly 130M model is an
engineering starting point that must be validated, not a paper-backed optimum.

## Mathematics and kernel decomposition

For a hidden row `e_i`, classifier rows `c_v`, and valid patch targets
`T_i`, CCE evaluates one vocabulary reduction

```text
LSE_i = log sum_v exp(e_i · c_v)
```

and only `|T_i|` indexed target logits.  It does not repeat `e_i` K times:

```text
loss_i = sum_{t in T_i} w_i (LSE_i - e_i · c_t)
loss   = sum_i loss_i / sum_i |T_i| w_i
```

The backward is separated into two terms:

```text
dlogits_dense(i, v) = |T_i| w_i softmax(i, v) / denominator
dlogits_target(i, t) = -w_i / denominator,  t in T_i
```

The CCE vocabulary kernel computes the probability term once per row. For
`Kmax <= 4`, it also subtracts the weighted target multiplicities inside that
tile, before casting the complete logit cotangent for its gradient matrix
products. This avoids cancellation error from rounding the two terms
separately and removes the extra sparse correction launch. For larger public
patch widths, the existing sparse Triton correction remains, bounding the
unrolled comparison budget. Both paths avoid a K-fold vocabulary reduction,
repeated embeddings and a `[rows,K,vocabulary]` tensor. Duplicate target IDs
contribute once per occurrence.

Indexed classifier loads require `0 <= target < vocabulary`.  `ignore_index`
is normalized to an invalid negative value before launch, and every target
kernel guards both lower and upper bounds, preventing negative pointer offsets
or out-of-range classifier reads.

## MiLe, μ-loss, and MEAP

| Extension | Patch behavior |
|---|---|
| MiLe | Entropy is computed once per hidden row. Its detached weight is normalized over valid target slots, so rows with fewer active slots do not receive an accidental K-fold weight. |
| μ-loss | The classifier-mean penalty and its direct `dC` update are evaluated once per CCE call, never once per patch target. |
| MEAP | Remains an input transformation before the model forward. Apply MEAP to clean token IDs before patch embedding aggregation; keep labels clean and pass the resulting hidden rows and fixed-shape targets to CCE. |

MEAP does not run inside the loss head, so its latency and allocations are not
included in the CCE measurements below.  TorchAO FP8 transformation likewise
remains upstream: FP8 GEMMs may emit BF16/FP16 hidden states, which enter the
ordinary CCE dtype path.  This change does not implement a native FP8 CCE
accumulator or alter TorchAO layer selection.

### Combined-model integration and scope

The measured CCE and small-Transformer examples below do **not** establish that
a full model with MEAP population-normalized MiLe, NITP, TWEO and conditional
token geometry supports a continuous transition. The separate real-data
comparison reported above validates one complete curriculum execution;
other architectures and configurations still require their own gate.
Public loss/pooling primitives are not a replacement for upstream model
and Trainer integration. In particular,
`mile_group_mask` remains rejected in patch mode; mixed input patches use the
explicit `mile_group_fraction` contract below. Never discard grouping or disable
an auxiliary objective to make a combined-model experiment pass.

For fixed backbone length T and patch size K, let `u[b,j,k]` be clean IDs,
`m[b,j,k]` valid input slots and `alpha=m/max(sum_k m,1)`. Apply input corruption
first and pool the **effective embedding outputs**, including any independent
mask-vector override or other enabled embedding composition:

```text
e[b,j] = sum_k alpha[b,j,k] * E(corrupted_u[b,j,k])
```

The hidden row j predicts clean targets from patch j+1, not its own patch.
Use fixed targets `[B,T-1,Kmax]`, mark inactive/invalid slots with ignore_index,
and pass `shift=0`. The backbone attention mask and position IDs have T rows.
Exclude targets without a valid causal predictor. At K=1, embedding values,
next-token alignment and all auxiliary masks must recover the token path.

For a nonlinear generator or coordinate-conditioned expert, pooling inputs to
that function is generally not equivalent to pooling its outputs:
`F(mean(z)) != mean(F(z))`. Geometry additionally needs its existing MEAP
exclusion: average coordinates over valid **uncorrupted** slots, not every
slot contributing to the effective embedding. Otherwise a mixed patch could
reintroduce a coordinate/parameter gradient at a masked token. A nonempty
convex mean preserves the coordinate bound; its local Jacobian is
`I / number_of_clean_slots` at a contributing slot and zero elsewhere.
Sharing this pooled geometry across JToK and JToK-M is a proposed curriculum
extension, not a result proved by the [JToK paper](https://arxiv.org/abs/2602.00800).
It must remain differentiable and use only the current input patch, never
future targets. Fully corrupted patches retain the trainable MEAP embedding
but disable JToK modulation and router auxiliary membership.

Pooling changes representation statistics and removes within-patch order.
For equal-sized patches, `Cov(e_mean)=sum_kl Cov(e_k,e_l)/K^2`; it only reduces
variance by exactly K under extra independence assumptions. RMS normalization
does not recover the discarded token order or make the objectives identical.

#### Implemented preparation primitive (not full model integration)

`pool_patch_inputs` implements that selection and pooling contract. It accepts
effective embedding outputs and optional coordinate outputs, not token IDs or
generator seeds. Use it **after** all active embedding compositions and the
independent MEAP override:

```python
from cut_cross_entropy import pool_patch_inputs

embedding, geometry, core_valid, geometry_valid, corrupted_fraction = pool_patch_inputs(
    effective_embeddings,   # [..., K*T, D]
    raw_valid_mask,          # [..., K*T], boolean
    patch_size=K,
    coordinates=token_geometry,       # [..., K*T, d_seed], optional
    corruption_mask=meap_positions,  # [..., K*T], boolean, optional
)
```

Outputs have T contiguous core rows. Embeddings average all valid input
slots; geometry averages only valid uncorrupted slots. Empty domains return
zero placeholders and false validity. Do not route an invalid placeholder.
The corrupted fraction is measured on **current input slots** and ignores
padding; it is not the fraction of corrupted future labels. It provides data
for the population normalization below, accepted as `mile_group_fraction`.

FP16/BF16 sums accumulate in FP32 and cast back to the respective input dtype;
FP64 reference inputs retain FP64. No producer gradient is detached. K=1
recovers each valid effective embedding and each valid uncorrupted coordinate.
Corrupted coordinates may use a different placeholder because their modulation
and auxiliary membership must be disabled in either phase.

Tests cover mixed/fully corrupted/empty patches, excluded NaNs, the exact
FP64 adjoint including the independent MEAP-vector gradient, K=1 identity,
layout consistency and locality to the current patch. Two separately prewarmed
K=1/K=4 preparation signatures kept Dynamo graph/break counts unchanged as
mask values changed, with `max-autotune` and disabled caches. This is synthetic
preparation/VJP coverage: it does **not** prove a full real-training transition,
stable CUDA Graph recordings or combined-model auxiliary integration. The
combined-model gate remains required.

#### Preserving MEAP population mass

The current causal CCE grouping uses the **predictor's** input-mask position:
compacted shift=1 indices identify hidden rows, not shifted target positions.
Any patch extension must preserve this alignment at K=1.

The opt-in extension assigns corrupted/clean fractions `r_i,g` to
input row i, with `sum_g r_i,g=1`. Let `n_i` be the number of valid clean target
slots, `a_i=(1+H_i)^gamma`, and define

```text
N_g = sum_i n_i * r_i,g
A_g = sum_i n_i * r_i,g * a_i
w_i = stopgrad(sum_g r_i,g * a_i * N_g/A_g)
L   = sum_i,k w_i * CE_i,k / sum_i n_i
```

Empty groups contribute zero. The attributed group coefficient satisfies
`sum_i n_i*r_i,g*a_i*N_g/A_g = N_g`. With binary membership and K=1, this
reduces to the existing independent population normalization; with fractional
membership it is a new attribution policy, not a causal decomposition of
corruption effects. The separate API does not reinterpret the boolean mask:

Summing that identity over groups gives `sum_i n_i*w_i = N`, where
`N=sum_i n_i`. For each row's empirical target distribution r_i, this yields
the weighted decomposition

```text
L_MiLe = sum_i n_i*w_i*(H(r_i) + KL(r_i || q_i)) / N
       >= sum_i n_i*w_i*H(r_i) / N
dL_MiLe/dlogits_i(v) = n_i*w_i*(q_i(v)-r_i(v)) / N
```

The derivative treats the entropy-derived weights as detached, exactly as
the implementation does; differentiating through their dependence on q
would define a different update. The bound describes the current weighted
task loss, not the total loss including auxiliary terms. If every supervised
row has K distinct targets, its entropy is log(K) and normalization preserves
that particular floor even with MiLe. Mixed duplicates, padding and K1 slots
give a different weighted entropy floor. This does not impose an ordering
between measured patch and token losses from different training trajectories.

Use fractional populations through the explicit API:

```python
loss = linear_cross_entropy(
    hidden[..., :-1, :], classifier, clean_next_patch_targets,
    patch_training_enabled=True, mile_enabled=True,
    mile_group_fraction=corrupted_fraction[..., :-1],
)
```

Fractions have the hidden row's leading shape, not its K target slots. They
describe current input corruption, must be detached floating metadata in
[0,1], and must remain tensors in both phases. K=1 uses binary fractions while
keeping the same Kmax target layout. The existing boolean token API is unchanged.
Do not differentiate through the entropy weights or population fractions.

For extreme finite gamma, evaluate each group's normalized weights using its
own supported-row maximum of `log(1+H_i)` before exponentiation. A single
global maximum can underflow an entire lower-entropy group before normalization.
Mask unsupported rows before exponentiation to avoid `0*inf`; empty/all-ignored
populations must produce finite zero task loss and gradients. Clamp exact
entropy to `[0,log(V)]`. The fractional reducer implements these rules; the
legacy no-fraction reducer retains its existing behavior. Gamma must be finite,
nonnegative and representable in FP32. Invalid active fractions produce NaN
rather than being silently clamped. Excluded rows are neutral even when their
unused statistics contain NaN. No device value is inspected on the host.

Small populations use one Triton kernel. Larger populations use four launches:
prepare row statistics/group maxima, accumulate rescaled group partials,
finalize scalar loss/scales, then save dense/sparse backward coefficients.
For R rows and P=ceil(R/128) tiles, temporary payload is `16*R + 40*P + 8`
bytes (excluding outputs and allocator rounding), independent of vocabulary.
The partial stage reloads the P maxima per tile; this trades a fifth launch
for O(P^2) small-array reads. It is not an unbounded linear-scaling claim.
CCE backward keeps its static 1/R factor by saving `R*w_i/sum(n_i)` per target
and `n_i` times that coefficient for the dense softmax term. No repeated hidden
rows or target-by-vocabulary tensor is introduced.

Focused CUDA checks cover FP64 reduction agreement, high gamma with separated
entropy populations, empty/tiny populations, invalid metadata, dense F+B with
bias and mu loss, and K=1 agreement with the existing binary token grouping.
Inductor F+B checks use fullgraph/max-autotune/disabled caches for BF16 storage
and FP32 storage under BF16 autocast, changing target/fraction values after
warmup. These operator checks are not the real 100+100 combined-model gate;
CCE's opaque operations remain CUDAGraph-unsafe by their existing contract.

#### Auxiliary objectives and state

Keep the configured coefficients and the existing activation taps:

```text
L_mu   = lambda_mu * ||mean_v C_v||^2
L_nitp = mean_valid(1-cos(P(h_final_j), stopgrad(h_shallow_(j+1))))
L_tweo = mean_layers mean_valid_rows mean_features (abs(A)/(tau+eps))^p
L_all  = L_task + L_mu + lambda_nitp*L_nitp
         + lambda_tweo*L_tweo + L_router_aux
```

Evaluate mu loss once per CCE call, not K times; its direct classifier gradient
is `2*lambda_mu*mean(C)/V`. NITP needs a separate 2D core validity mask, not the
3D patch target tensor. Predicting the next *patch* shallow state is a deliberate
extension of [NITP](https://arxiv.org/abs/2605.24956), not token-equivalent for
K>1. TWEO keeps the current per-layer tap and averaging units: do not multiply
it by K or change streams to make its scalar value look comparable. Monitor
activation distributions and gradient scales as well as the auxiliary losses.
JToK-M routing and its auxiliary loss remain outside the normalization kernel.

#### Compilation and verification boundaries

Distinguish the two batching policies. With fixed **core** length T, one
backbone/loss shape can serve both phases while preparation consumes K*T or
T raw slots. With fixed **raw** length L, as in the original-sequence runner,
the core lengths are L/K and L: two intentional training shapes must be
compiled and fully exercised before counted updates. Kmax target slots stay
fixed in either policy. Do not create Kmax dummy embeddings every token
step merely to force one monolithic shape.

Preparation may be part of a compiled raw-forward composition or a separate
prewarmed producer, but gradients must reach the same embedding parameters.
Select the phase from completed optimizer updates outside the compiled
tensor computation; reuse the already-warmed signature. Prewarm forward,
backward and optimizer specializations, then restore model/master weights,
optimizer/stabilizer, scheduler and RNG in place before real training.
Zero post-prewarm compilation growth, not one universal graph, is the gate.

Count Dynamo graphs/breaks, Inductor compilation events, external JIT entries
and CUDA Graph recordings separately. A graph capture is not an Inductor
recompile. Verify unchanged counts after prewarm, particularly across the
phase transition, with execution profiles and real-data optimizer steps.

Prewarm **compositions**, not just functions in isolation. [CUDA Graph
Trees](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
can record another invocation path when a shared downstream function follows
a different producer or tensor-lifetime pattern. A common core signature
therefore does not guarantee a single capture. Exercise preparation, core,
loss, backward and optimizer on both phase paths, then revisit both paths
before taking the warm snapshot. If validation uses compiled execution, warm
its distinct grad/training-mode path too. Do not disable graph replay to make
the test pass. Retain only intentional cross-step tensors; changing logging
lifetimes can change replay eligibility. Restore weights, optimizer/master
state, scheduler and RNG after this warmup, without replacing graph-owned
storages. These are requirements for the combined-model transition, not a
claim that the existing small transition benchmark already covers them.

Report raw input tokens, valid targets and backbone rows separately. At fixed
core T, a patch step consumes K times more reserved raw slots than a token
step. At fixed raw L, both phases reserve L slots per sequence, but padding
and next-patch alignment can change valid supervised counts. A step-speed
comparison alone is not an equal-training-budget quality comparison.

The original [patch curriculum](https://arxiv.org/abs/2407.12665) resets
optimizer/scheduler state between stages. Keeping continuous optimizer state
is a different policy and must be recorded explicitly. Preserving graph shape
does not by itself choose that policy or prove downstream quality.

## Precision policy

Each correct-class logit is extracted from the exact vocabulary tile that also
feeds the LSE reduction. This preserves the accumulation and rounding identity
between both values; a separate scalar or K-target dot reduction could otherwise
produce a negative/nonzero cross-entropy for a one-class or highly concentrated
distribution. Algebraic independence of the backward terms does not imply
finite-precision equivalence: `Q(a-b) != Q(a)-b`. The small-width fused backward
forms `dense_weight*p - target_weight*multiplicity` before its dot-input cast.
Multiplicity is computed on row vectors: for each target slot, count equal
target IDs and apply the correction only at that ID's first occurrence.
For valid target slot k, define `c_ik = sum_j 1[y_ij=y_ik]` and let `f_ik`
indicate that no earlier slot has the same ID. Then the target correction is
`target_weight_i * sum_k f_ik*c_ik*1[v=y_ik]`. This needs bounded O(B*K^2)
row comparisons plus O(B*V*K) vocabulary comparisons, not another FP32 B*V
multiplicity tile. Each vocabulary element receives at most one nonzero
subtraction. A simpler per-slot subtraction is deliberately avoided:
`1-FP32(1/3)-FP32(1/3)-FP32(1/3)` leaves a residue whereas
`1-FP32(3*FP32(1/3))` is zero. The three-repeated-target regression covers
this finite-precision distinction; it does not claim an exact softmax at
large logits, where even ordinary CCE can have LSE rounding error.
In the BF16 regression with logits `[6,0]` and one active target, the earlier
sparse route differs by about 58% from the ordinary token cotangent at a
near-zero gradient; the fused Kmax=1/4 routes match the ordinary gradient
exactly in that test. This is not a bitwise-equivalence guarantee for every
input, width, accumulation order or filtering case. Larger widths retain
the earlier separate-term rounding behavior.

The configured filter thresholds and accumulation/dtype policy are not
changed. Small-width filtering now observes the complete target-inclusive
cotangent instead of only its probability component. The extra K comparisons
can increase register/compute cost; one fewer launch alone is not proof of a
whole-training speedup.

Patch row weights are saved in FP32.  Because their bound depends on the number
of valid slots, patch mode keeps automatic mixed-gradient accumulation on the
conservative FP32 path.  Explicit environment overrides retain their existing
meaning, but are not enabled or changed by this feature.

## Supported subset

The initial production path intentionally targets the pretraining contract:

- CUDA CCE implementations (`impl="cce"`, `cce_exact`, or the existing CCE
  presets), including use from a surrounding `torch.compile` graph;
- `reduction="mean"`;
- `shift=0`; patch alignment is performed by the data/model pipeline;
- no logit softcap, `return_lse`, or vocabulary parallelism yet;
- fixed `Kmax >= 1` for both patch and token phases.

`impl="torch_compile"` names the separate dense reference implementation and
does not support this flag.  This is distinct from compiling a model that calls
a CCE implementation: the latter uses CCE's opaque compiler boundary and is a
supported path for BF16/FP16 compute. FP32 remains available in eager CCE; the
opaque compiled boundary currently follows its pre-existing BF16/FP16 contract.

## Measured operator feasibility

An earlier sparse-backward checkout was measured on an RTX 5070 Ti with PyTorch
`2.12.1+cu130`, CUDA 13.0, Triton `3.7.1`, BF16, 512 rows, `D=512`,
`V=65,536`, `K=4`, bias, `cce_exact`, MiLe `gamma=1`, and μ-loss
`lambda=1e-4`.  Float32 matmul precision was the training policy `high`; the
library does not override it.  Each value below is the mean of the medians from
two fresh, interleaved processes with 30 warmups and 200 repetitions.  The test
process—not the library—was capped at 10 GiB.

| Route | Forward | Forward + backward | Incremental peak (forward / training) |
|---|---:|---:|---:|
| Existing one-target CCE | 1.0092 ms | 4.0576 ms | 41,984 / 202,518,016 bytes |
| Patch phase, four valid targets | 0.8784 ms | 3.7261 ms | 71,168 / 202,532,352 bytes |
| Token phase, one valid plus three ignored | 0.8814 ms | 3.6703 ms | 71,168 / 202,532,352 bytes |
| Repeat the embedding through ordinary CCE four times | 2.1554 ms | 11.1848 ms | 2,142,208 / 207,769,088 bytes |

The patch route is 59.2% faster in forward and 66.7% faster in the measured
forward+backward operator than repeating ordinary CCE four times.  Its small
fixed target/loss workspace adds 29,184 forward bytes or 14,336 training bytes
over one-target CCE; it avoids the much larger repeated-embedding reference.

The out-of-range target fix was also checked directly against its immediate
parent commit in the same alternating protocol.  Patch forward was 0.8784 ms
after the guard versus 0.8787 ms before it (−0.03%), and forward+backward was
3.7261 ms versus 3.7737 ms (−1.26%); incremental peaks were byte-identical.
The hot lock kernel only adds the vocabulary predicate on its already-masked
final tile, rather than adding bilateral range checks to every tile.

These historical isolated CCE results predate the small-width fused backward;
they are neither measurements of that new path nor end-to-end model-step claims.
Rerun the benchmark below on the final target GPU before quoting whole-training
throughput or memory.

## End-to-end `max-autotune` validation

`benchmark.patch_training_e2e` exercises a tied-embedding causal Transformer,
fused CCE, MiLe, μ-loss, backward, and AdamW. Embedding aggregation and phase
selection stay outside the compiled function; the Transformer plus loss are
compiled with `fullgraph=True`. The benchmark uses the training precision
policy `high` and limits only its own process to 10 GiB. This compiled benchmark
accepts BF16 or FP16; use the eager operator benchmark for FP32. Compilation
warmup also allocates AdamW state, after which the benchmark restores the
initial model and zeros the preallocated optimizer state before measurement.

A 99,517-parameter model completed 4,000 measured BF16 steps (2,000 patch then
2,000 token) after warmup. AdamW was recreated at the transition. The run
produced one Dynamo graph and zero graph breaks; loss remained finite and moved
from 6.9351 to 5.5466. Peak allocated memory was 1,037,312 bytes, 418,816 bytes
above the post-warmup baseline. The patch and token phases measured 1.8408 and
1.8385 ms/step respectively. Their raw-token throughputs were 69,535 and 17,406
tokens/s because each patch step consumes four times as many raw tokens. The
five boundary steps 1998--2002 measured 1.656, 1.474, 1.760, 2.364, and 2.167
ms: there was no compile-sized transition spike. Recreating AdamW itself took
0.091 ms of host time. The timer retains the loss on device and converts only
the final value after synchronization, avoiding a CPU/GPU synchronization in
every measured step.

The following comparisons process the same number of raw tokens per step.
"Token base" sends all `K*T` rows through the Transformer and uses ordinary
causal CCE with `shift=1`; "patch" averages each K-token group, sends T rows,
and predicts K targets per row. Every row below used BF16, MiLe, μ-loss,
`high`, `max-autotune`, one graph, and zero graph breaks on the RTX 5070 Ti.

| Geometry (`B,T,D,L,V,K`) | Parameters | Patch / token-base ms | Patch latency delta | Raw-token throughput delta | Forward FLOP proxy saved | Incremental peak patch / base |
|---|---:|---:|---:|---:|---:|---:|
| `2,16,64,1,1021,4` | 99,517 | 1.8403 / 2.3479 | -21.62% | +27.58% | 77.19% | 417,792 / 450,560 B |
| `3,17,96,2,4093,4` (odd) | 545,437 | 2.0621 / 2.2794 | -9.53% | +10.54% | 76.65% | 2,412,032 / 2,493,440 B |
| `4,64,256,4,8191,4` | 4,206,847 | 2.4001 / 2.8774 | -16.59% | +19.88% | 77.22% | 13,139,456 / 14,201,856 B |
| `2,32,128,2,65537,4` (wide vocabulary) | 8,717,697 | 2.1813 / 2.9531 | -26.13% | +35.38% | 75.71% | 50,537,472 / 50,670,592 B |

Across these shapes, raw-token throughput improves by 10.54--35.38% (23.35%
arithmetic mean), step latency falls by 9.53--26.13% (18.47% mean), and
incremental peak allocation falls by 0.26--7.48% (4.57% mean). The smallest
case no longer loses to fixed overhead: the small-patch path fuses its scalar
reductions and the classifier autotuner selects a tile proportional to the
actual row count instead of padding 30 rows to 128. The wide-vocabulary case
saves little peak allocation because the shared dense classifier gradient and
AdamW state dominate both runs, even though step latency improves.

The model FLOPs are counted with PyTorch's `FlopCounterMode`. The CCE column
adds the dense-dot work implied by the fused vocabulary LSE,
`2 * rows * V * D`; it is a transparent forward-work proxy, not a claim about
hardware instruction counts. Forward and backward speed and VRAM are measured,
not inferred. Because patch targets omit one cross-patch boundary, predicted
target counts differ slightly at these short sequence lengths; raw-token
throughput is used for the fair speed comparison.

Nsight Compute 2025.3.1 was also used to measure physical forward work for the
smallest geometry in the table. The two reports used the same compiled model,
precision, extensions, and inputs as the latency comparison. Unlike the
algorithmic proxy, these counters include tile padding and auxiliary kernels:

| Physical NCU counter | Optimized patch | Token base | Patch delta |
|---|---:|---:|---:|
| Forward kernels | 23 | 31 | -25.81% |
| Tensor BF16-to-FP32 FLOPs | 23,068,672 | 42,467,328 | -45.68% |
| All measured floating-point operations | 24,662,785 | 44,943,407 | -45.12% |
| DRAM bytes | 968,704 | 1,561,088 | -37.95% |
| Summed kernel duration | 69,440 ns | 99,392 ns | -30.14% |

The previous implementation required 47 forward kernels, 37,943,675 measured
floating-point operations, 1,804,032 DRAM bytes, and 134,432 ns of summed kernel
duration for the same patch input. Fusing the small patch reduction and adapting
the classifier tile therefore remove 51.06% of its launches, 35.00% of its
executed floating-point work, 46.30% of its DRAM traffic, and 48.35% of its
summed kernel duration. The 77.19% proxy still describes mathematical work
rather than hardware instructions, but the optimized path now converts a
material part of that saving into a measured 45.12% physical FLOP reduction
against token base.

The matching backward profile measured 12,854,272 floating-point operations,
635,034 DRAM bytes, and 14,284 ns for patch, versus 51,403,776 operations,
584,090 bytes, and 17,011 ns for token base. That is 74.99% fewer operations and
16.03% less summed kernel duration; DRAM traffic is 8.72% higher because the
patch route updates several correct-class targets per retained row. NCU replays
kernels to collect counters, so the end-to-end wall time printed by a profiled
run is intentionally not used as a latency result.

Rare compiled cases also passed: `K=8,T=2,D=33` and the degenerate
`K=1,T=3,D=33`, including MiLe, μ-loss, backward, AdamW reset, one graph, and
zero breaks. During `max-autotune`, candidates that exceeded the GPU's per-block
resource limit were rejected and valid candidates were selected; those
candidate rejections are neither VRAM OOMs nor runtime training failures.

## Reproduction

Forward, with the same extensions and a 10 GiB test-process ceiling:

```bash
python -m benchmark.cce_patch \
  --rows 512 --dim 512 --vocab 65536 --patch-size 4 \
  --dtype bf16 --mile --mu-loss \
  --warmup 30 --repetitions 200 --max-test-vram-gib 10
```

Forward plus backward:

```bash
python -m benchmark.cce_patch \
  --rows 512 --dim 512 --vocab 65536 --patch-size 4 \
  --dtype bf16 --mile --mu-loss --training \
  --warmup 30 --repetitions 200 --max-test-vram-gib 10
```

The benchmark clears warmup gradients and synchronizes before sampling the
memory baseline.  Its JSON records GPU, PyTorch/Triton versions, dtype, shape,
warmup/repetition settings, absolute latency, incremental peak bytes, and
deltas versus one-target CCE.

End-to-end 4,000-step transition:

```bash
python -m benchmark.patch_training_e2e \
  --case transition --steps 4000 --patch-steps 2000 \
  --batch 2 --sequence 16 --dim 64 --layers 1 --heads 4 \
  --vocab 1021 --patch-size 4 --dtype bf16 \
  --compile-mode max-autotune --mile --mu-loss \
  --reset-optimizer-at-transition --max-test-vram-gib 10
```

For an equal-raw-token performance pair, run the same command twice with
`--case patch` and `--case token_baseline`. `--sequence` is T; the token base
automatically uses `patch_size * T` Transformer rows.

To reproduce the physical forward counters, first run each case normally once
so `max-autotune` can populate its cache. Then profile the measured forward NVTX
range. This example records the patch case; replace both `patch` occurrences
with `token_baseline` for the matching baseline:

```bash
ncu --export patch-forward --target-processes application-only \
  --nvtx --nvtx-include 'patch_training_e2e_patch_forward/' \
  --metrics \
sm__ops_path_tensor_src_bf16_dst_fp32.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,dram__bytes.sum,gpu__time_duration.sum \
  python -m benchmark.patch_training_e2e \
  --case patch --steps 1 --warmup 4 --batch 2 --sequence 16 --dim 64 \
  --layers 1 --heads 4 --vocab 1021 --patch-size 4 --dtype bf16 \
  --compile-mode max-autotune --mile --mu-loss --max-test-vram-gib 10

python -m benchmark.patch_ncu_report patch-forward.ncu-rep
```

## Verification coverage

`tests/test_cce_patch.py` covers dense forward/backward agreement, bias,
duplicates, ignored/negative/out-of-range targets, MiLe, μ-loss, token-phase
equivalence, unsupported-option validation, and five compiled steps spanning
the phase transition while asserting one graph and zero graph breaks.
The confident-target BF16 cancellation regression additionally covers Kmax=1/4
with `cce_exact` and `cce_kahan_full_c`, MiLe and mu enabled, against the ordinary
token cotangent and an independent FP64 derivative.
Kmax=5 checks exercise the retained sparse path in BF16 and FP32, including
duplicate, ignored and out-of-range targets and fractional MiLe with mu loss.

The fused path is a precision correction, not an established training speedup. A
profiled synthetic CCE forward/backward pair on RTX 5090, with BF16,
R=2048, V=8192, D=128 and Kmax=4, observed median K4 latency of 0.473 ms
versus 0.472 ms for the earlier sparse path; K1 was 0.470 ms for both.
Both allocated 7,432,192 incremental peak bytes. Compiler metadata reported
255 registers and 148 spills for the initial fused backward, versus 200 registers and
zero spills for sparse backward. Those metadata counts are not measured
dynamic spill traffic. This single sequential pair had the profiler enabled;
it does not establish a statistically robust performance difference or
predict whole-model training throughput. Register pressure remains an
explicit engineering issue despite removing the separate target launch.

The later row-wise duplicate folding avoids that additional FP32 tile. On the
same synthetic shape, a four-warp diagnostic reported 255 registers/28 spills
and K4=0.546 ms, K1=0.465 ms; fewer spills alone did not ensure lower latency.
An eight-warp diagnostic reported 218 registers/zero spills and K4=0.452 ms,
K1=0.451 ms, with the same 57,344 shared-memory bytes and allocated peak.
The default heuristic now selects eight warps only for fused Kmax=1..4,
BF16, D>=128 and the 128x128x16/two-stage tile formerly assigned four warps.
Small tiles, FP32, FP16, ordinary token targets, larger sparse widths and
explicit autotuning keep their original selection. Tile dimensions, filtering
and accumulation precision do not change. This trades more threads against
per-thread register pressure, rather than moving a precision boundary.
These are single profiled diagnostics, not confidence intervals or proof of
a whole-training speedup; the complete real-training gate is still required.
