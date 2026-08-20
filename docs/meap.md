# MEAP with CCE, MiLe, and mu loss

This repository implements the pretraining form of
[Mask-Enhanced Autoregressive Prediction](https://arxiv.org/abs/2502.07490).
The implementation follows the separation used by the
[released MEAP training code](https://github.com/Lieisyourlie/MEAP): MEAP
corrupts autoregressive inputs, while CCE computes the objective from the
resulting hidden states and the original clean labels.

![MEAP kernel data flow](../assets/meap_kernel.svg)

## How to read the diagram

$T$ is the padded sequence length, $N$ is the number of eligible positions in
one sequence after exclusions, and $K$ is the exact number that MEAP replaces.
The Triton launch uses one program per batch row.

- **Panel (a)** loads the IDs and, when supplied, the existing padding or
  eligibility mask. A prefix sum simultaneously counts eligible positions and
  maps sparse positions onto the dense rank domain $[0,N)$. `exclude_last`
  acts before this count, so the protected final position is not part of $N$.
- **Panel (b)** permutes those dense ranks using two Philox-derived sequence
  keys and twelve fixed swap-or-not rounds. Every round pairs ranks and gives
  both members of a pair the same swap decision, so each round is an
  involution and their composition remains a bijection.
- **Panel (c)** compares each unique permuted rank with $K$. The predicate
  $\mathrm{permuted\_rank}\lt K$ therefore selects exactly $K$ distinct
  positions without
  scores, sorting, top-k, collision handling, or retry loops. `tl.where` then
  chooses between the original ID and the mask-token ID. The output IDs are the
  only required global write; the selected-position mask is optional.
- **Panel (d)** places the operation in the training pipeline. Corrupted IDs
  enter the causal Transformer, while clean labels and the original
  causal/padding masks remain unchanged. CCE, MiLe, and mu loss operate later
  on the resulting contextual hidden states.

MEAP has no differentiable backward kernel because it is an integer input
augmentation. Its training effect comes from changing the causal context from
which hidden states are computed, not from adding another term to the loss.

## Kernel architecture

The wrapper launches `_meap_mask_inputs_kernel` with grid `(batch_size,)`.
Consequently, one Triton program owns one complete sequence and can perform the
prefix sum, reductions, and permutation without communication between rows.

| Component | Kernel design | Consequence |
| --- | --- | --- |
| Program shape | `BLOCK_T = next_power_of_2(sequence_length)` | One static lane domain covers the complete row; lanes beyond $T$ are masked. |
| Launch | Four warps through `BLOCK_T=512`, eight above it, one stage | Fixed launch policy with no runtime autotuner lookup. |
| Compile-time flags | Existing-mask presence, padding-mask convention, `exclude_last`, `return_mask`, and `return_metrics` are `tl.constexpr` | Unused mask reads, branches, and diagnostic stores are removed from the compiled specialization. |
| Row state | Eligibility, prefix ranks, permutation state, and selection stay in registers | No global random-score, sorting workspace, top-k workspace, or atomic counter. |
| Required I/O | Read IDs and an optional existing mask; write masked IDs | Global memory traffic is linear in the input size. |
| Optional I/O | Write one boolean per position only for `return_mask=True`; atomically reduce two counters for `return_metrics=True` | Training can log exact eligible/masked counts without allocating a token mask. |
| Work | `O(B * T)` with twelve fixed permutation rounds | Execution does not grow as `T log T` as sorting would. |

The current register-resident design deliberately caps the Triton path at
sequence length 4096. Raising that limit is not a constant-only change: the
larger power-of-two block would increase register pressure and can reduce
occupancy, so it would require a separate multi-block design and benchmark.

## Data flow

1. Collate clean token IDs, clean shifted labels, and the padding mask.
2. Reuse the existing padding mask directly. Build an eligibility mask only
   when BOS, EOS, or other protected positions also need to be excluded.
3. Call `meap_mask_inputs` once, before the model forward.
4. Compose dense or Leviathan embeddings and optional Spelling Bee features.
5. Select the independent trainable MEAP vector in the epilogue that owns the
   final embedding write. This is Leviathan's output GEMM when Leviathan is the
   last producer, or the Spelling Bee epilogue when byte features follow it.
6. Run the ordinary causal model with the original causal and padding masks.
7. Pass its hidden states and the clean labels to `linear_cross_entropy`.

MEAP is not part of the CCE loss kernel. Calling it after the model or from
inside CCE would be too late to change the contextual hidden states.

## Dedicated embedding contract

The mask ID must be a reserved vocabulary ID that is absent from clean data and
different from PAD, BOS, and EOS. Reusing PAD is incorrect for three separate
reasons: a dense embedding commonly has `padding_idx=pad_token_id` and therefore
does not learn that row; Leviathan derives the ID through shared codebooks; and
Spelling Bee adds shared byte parameters. A final independent override avoids
all three couplings without changing clean tokens.

```python
from cut_cross_entropy import MEAPEmbeddingOverride

self.meap_embedding = MEAPEmbeddingOverride(hidden_size, mask_token_id)

# `composed` is already the result of either dense or Leviathan embedding and
# any optional Spelling Bee augmentation.
composed = self.meap_embedding(masked_input_ids, composed)
```

For a continuous migration from a legacy PAD-based checkpoint, compute the old
PAD representation after every active embedding augmentation and call
`meap_embedding.initialize_from(old_final_pad_embedding)` once. New training can
use the normal model initializer.

### Optional backend requirements

| Execution case | MEAP masking API | Triton/CUDA | Dedicated vector |
| --- | --- | --- | --- |
| Active MEAP training, `implementation="triton"` | Required | Required | Required |
| Active MEAP training, `implementation="torch"` | Required | Optional | Required |
| Clean training with MEAP disabled | Optional | Optional | Harmless/optional |
| Evaluation or inference | Not required by MEAP | Optional | Loaded from checkpoint when present |

The model must defer its backend availability check until active MEAP training.
Merely loading a checkpoint with `use_meap=True` must not make inference depend
on the corruption kernel. Other configured loss backends can retain their own
requirements. Labels and attention masks always remain clean.

The semantic override is deliberately owned by the last active producer, so the
same contract covers dense, dense+Spelling Bee, Leviathan, and
Leviathan+Spelling Bee without allowing an earlier producer to modify the mask
vector afterward.

### Leviathan epilogue fusion

`leviathan_embedding_compiler_safe` accepts the optional pair
`mask_embedding=` and `mask_token_id=`. Both must be supplied together. On the
CUDA path, Leviathan's final `modes @ W_out` Triton kernel loads the mask vector
and selects it in the same store that writes the BF16 embedding. There is no
second `[tokens, hidden]` read/modify/write pass.

The backward contract is equally important: rows whose ID equals
`mask_token_id` are zeroed before Leviathan's `dM`, `dW_out`, spline, projection,
and codebook gradients. Their original `grad_output` is reduced only into the
dedicated mask parameter. The normal `HAS_MEAP=False` Triton specialization
contains neither the ID load nor the mask-vector load.

| Embedding route | Owner of the final MEAP selection | Reason |
| --- | --- | --- |
| Dense only | model embedding epilogue | No custom Leviathan kernel is active. |
| Dense + Spelling Bee | Spelling Bee epilogue | Byte features are composed last. |
| Leviathan only | Leviathan output-GEMM store | This is the true Triton-fused route. |
| Leviathan + Spelling Bee | Spelling Bee epilogue | Selecting inside Leviathan alone would let Spelling Bee corrupt the dedicated vector afterward. |

The CPU/reference fallback remains a differentiable `torch.where` with the same
mathematics. This keeps checkpoints and inference usable when the optional
Triton package is absent; physical kernel fusion is a CUDA optimization, not a
different model definition.

`torch.no_grad()` always selects the checkpoint-free Leviathan inference op,
even when the mask vector is stored as a trainable `Parameter`. If every
Leviathan parameter is frozen and only the mask vector trains, the same
inference op supplies the forward values and a final differentiable selection
supplies only the mask-vector gradient. That mask-only route neither saves LEV
checkpoints nor launches the codebook/spline/projection backward.

## Kernel contract

```python
meap_mask_inputs(
    input_ids,
    mask_token_id,
    enabled=True,
    mask_ratio=0.15,
    eligible_mask=None,
    padding_mask=None,
    seed=0,
    exclude_last=True,
    return_mask=False,
    implementation="triton",
)
```

- `input_ids` is a contiguous or strided two-dimensional int32/int64 CUDA
  tensor for the Triton path.
- `padding_mask=True` means replacement is forbidden. Passing it directly
  avoids allocating and computing `~padding_mask` in a separate GPU operation.
- Alternatively, `eligible_mask=True` means replacement is allowed. Use this
  form when padding and protected-token exclusions are already combined. The
  two mask arguments are mutually exclusive.
- Each row receives $\max(1,\lfloor rN\rfloor)$ replacements, where $r$ is
  `mask_ratio`, when the ratio and eligible count are positive.
- Sampling is without replacement and independent between rows.
- `seed` accepts either the original Python integer or a scalar int32/int64
  tensor on the same CUDA device as `input_ids`. Prefer the device scalar for a
  seed that changes each training step: it prevents `torch.compile` from
  specializing the model graph on every Python integer value. The Triton path
  folds both halves of an int64 device seed into its 32-bit Philox seed; packed
  step, microstep, and rank fields therefore contribute even when they occupy
  bits above bit 31. Device seeds already in the uint32 range remain unchanged.
- `exclude_last=True` removes the last eligible input from sampling. This is
  appropriate when CCE uses `shift=1`, because that hidden state has no valid
  next-token target.
- The operation never edits the input or labels in place.
- `return_mask=False` allocates only the output ID tensor. Requesting the
  boolean mask adds one byte per input position.
- `enabled=False` returns the original input tensor without allocating an ID
  copy. When the diagnostic mask is requested, only an all-false boolean tensor
  is allocated.

The fixed-count Triton kernel supports sequence lengths through 4096. The
reference `implementation="torch"` path is also usable on CPU and exists for
validation and comparison, not as the recommended training path.

### Selection strategy

The kernel does not create a random score per token and does not execute sort,
argsort, or top-k. Its selection is constructed as follows:

1. A prefix sum maps every eligible position to a unique dense rank in
   $[0,N)$, even when eligibility contains holes.
2. Philox generates two keys per sequence, rather than one random value per
   token.
3. Each of twelve swap-or-not rounds draws a keyed pivot $p\in[0,N)$ and
   pairs rank $x$ with $(p-x)\bmod N$.
4. A keyed hash of the pair chooses whether both partners swap or both stay.
   Because both ranks use the same pair identifier and decision, every round
   is an involution and therefore a bijection.
5. The composition of the twelve rounds is a permutation over the exact
   $[0,N)$ domain. A token is selected exactly when its permuted rank is below
   $K$.

The permuted ranks are unique and exactly $K$ positions satisfy
$\mathrm{rank}\lt K$:
there are no collisions, rejection duplicates, data-dependent retry loops,
atomic updates, or approximate Bernoulli counts. Twelve fixed rounds make the
selection work $O(T)$ with the same execution structure for dense and padded
rows.

The implementation uses four warps through block length 512 and eight warps at
larger lengths. These are compile-time launch choices, so there is no runtime
autotuner lookup in the training loop.

All selection state lives in registers. The only global reads are the input IDs
and optional existing mask; the only required global write is the output IDs.
There are no `.cpu()`, `.item()`, host-generated random tensors, or intermediate
CPU-to-GPU transfers. Supplying `padding_mask` directly also removes the GPU
allocation and pass that `~padding_mask` would otherwise require.

The tests verify exact counts for ratios 0, 0.01, 0.15, 0.5, and 1.0 with
dense and sparse eligibility through length 4096. They also check marginal
position frequencies with a chi-square bound and adjacent-pair frequency over
4096 sequences for three independent seeds. These tests detect the positional
and local-correlation bias found in simpler affine/avalanche permutations that
were rejected during development. They are evidence appropriate for this
training augmentation, not a claim that twelve rounds provide a cryptographic
uniform permutation over every possible subset.

## Reproducibility and checkpointing

The Triton implementation uses a stateless seed. For distributed training,
derive it from at least the optimizer step, microstep, and rank. Different
ranks should not receive the same masks unintentionally. Generate masked IDs
outside an activation-checkpointed model function; otherwise a recomputed
forward could receive a different corruption.

The official pretraining script samples one shared set of columns for the
whole batch. This implementation instead samples per sequence because the
repository uses padded, non-packed examples of varying lengths. That preserves
the paper's fixed per-example ratio while increasing batch diversity.

## Combined objective

The initial supported experiment keeps the mechanisms independent and explicit:

- CCE implementation: `cce_kahan_full_c`
- MEAP ratio: `0.15`
- MiLe: enabled, `gamma=1.0`, detached and population-normalized weights
- mu loss: enabled, `mu_loss_lambda=1e-4`

MEAP changes the context before the Transformer. MiLe then reweights the clean
next-token losses using detached predictive entropy. Mu loss adds
$10^{-4}\lVert\mathrm{mean}(C,\mathrm{dim}=0)\rVert_2^2$. Neither MiLe
nor mu loss changes how
MEAP positions are sampled.

When MEAP and MiLe are active together, CCE receives the selected-input mask as
`mile_group_mask`. For `shift=1`, input position $t$ is aligned with the hidden
row at $t$ that predicts clean label $t+1$; the mask is flattened with exactly
the same valid-row mapping as CCE, including ignored labels. Let

$$w_i=(1+H_i)^\gamma,\qquad G=\{i:\text{input }i\text{ was replaced}\}.$$

The detached weights are normalized independently:

$$
\hat w_i =
\begin{cases}
|G|w_i / \sum_{j\in G}w_j, & i\in G,\\
|\bar G|w_i / \sum_{j\in \bar G}w_j, & i\notin G.
\end{cases}
$$

MiLe therefore still ranks examples by entropy *within* both populations, but
the configured MEAP ratio controls their aggregate objective mass. A single
global normalization can unintentionally defeat that ratio: in the 5,000-step
causal probe below, 15% corrupted positions reached mean weight `3.5933` while
clean positions reached `0.6158`, assigning approximately 50.7% of all NTP
gradient mass to the synthetic population. Population normalization held both
means at `1.0` and reduced peak mask-vector gradient RMS from `0.006185` to
`0.001633` (-73.6%). No NaN or Inf occurred in either 5,000-step run; the result
isolates and removes this amplification mechanism rather than claiming that a
short synthetic test proves indefinite training stability.

This conditional normalization is inactive when either MEAP or MiLe is off.
It does not change token selection, clean labels, entropy, CCE arithmetic,
mu loss, evaluation, or inference. The large-vector Triton path folds the
population sums and count into MiLe's existing reduction kernels instead of
launching a separate normalization pass.

Validation and inference should use clean inputs (`enabled=False`). A corrupted
validation pass may be reported separately as a robustness diagnostic, but it
is not the clean language-model NLL.

## Metrics

`masked_count / eligible_count` is the correct per-step **implementation
metric**. It verifies the requested ratio and catches padding/exclusion errors,
but it does not demonstrate that MEAP improves retrieval.

The paper's mechanistic metric is a paired evaluation on the same examples and
checkpoint: run clean inputs and a copy with fixed selected positions, retain
attention probabilities, then report (1) relative attention-score decay at the
selected keys and (2) relative attention-variance change at unselected keys.
`meap_attention_diagnostics` implements those paired calculations. Run them at
evaluation cadence rather than every training step because retaining attention
matrices changes the memory and latency profile. The helper applies causal
query-key visibility by default. Pass the original eligibility mask so padding
keys are excluded; for square self-attention it also excludes the corresponding
padded query rows. For cached, packed, sliding-window, or other nonstandard
attention, pass `query_mask=` and/or the explicit pairwise `visibility_mask=`.
Track the dedicated vector RMS as a cheap health metric, and use
`meap_mask_embedding_max_abs` beside it to distinguish broad drift from a
single-coordinate outlier. The Leviathan mask gradient is reduced in FP32 and
cast once to the parameter dtype; this matters at long token counts, where a
native BF16 reduction reached `0.992676` maximum absolute error in the 32,768-row
stress probe. Use Needle-in-a-Haystack or multi-document QA as the outcome metric; training loss
alone cannot measure MEAP's retrieval effect.

## Reproducing compile latency and memory

The isolated runner below uses `torch.compile(..., mode="max-autotune")` only
inside the benchmark, checks eager/compiled outputs and gradients, measures five
steps, clears warmup gradients before the memory baseline, and enforces a 10 GiB
total allocated-peak ceiling:

```bash
python -m benchmark.meap_embedding_profile \
  --batch 64 --sequence 512 --hidden 512 --dtype bfloat16 --steps 5
```

For the Leviathan-specific comparison, use:

```bash
python -m benchmark.leviathan_meap_profile --tokens 4096 --steps 100
```

To reproduce the MEAP-conditioned MiLe forward+backward comparison (BF16
activations and the vocabulary geometry used in development):

```bash
python -m benchmark.meap_mile_profile \
  --batch 8 --sequence 513 --hidden 512 --vocab 64402 \
  --warmup 10 --steps 100 --vram-limit-gib 10
```

On an RTX 5070 Ti (driver `610.47`) with Python `3.14.4`, PyTorch
`2.12.1+cu130`, CUDA build `13.0`, and Triton `3.7.1`, paired runs produced:

| Causal rows | Global MiLe | Population-normalized | Change | Allocated peak (both) |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2.7264 ms | 2.9951 ms | +9.9% | 316.49 MiB |
| 4,096 | 13.8361 ms | 14.2252 ms | +2.8% | 330.63 MiB |
| 8,192 | 24.4383 ms | 24.7604 ms | +1.3% | 346.79 MiB |

The 512- and 8,192-row entries use 50 paired measurements; 4,096 uses 100.
This is the full compiled CCE forward+backward, not just the small MiLe
reduction. Population statistics add a fixed reduction cost whose relative
impact falls as classifier work grows. Run-to-run variance remains
sub-millisecond. The production library does not set a compile mode or alter
Dynamo/Inductor flags.

MXFP8 weight training still presents BF16 (or the configured activation dtype)
at this embedding boundary, so benchmark that activation dtype rather than
passing an FP8 storage tensor directly to the override.

Reference measurements on an RTX 5070 Ti with PyTorch 2.12.1+cu130 are below.
They measure only the override plus a synthetic reduction/backward, not a full
training step. Every row uses five measured steps and passes eager/compiled
output and gradient checks.

| Shape `[B,S,D]` | Dtype | Eager mean | max-autotune mean | Compiled peak allocated |
| --- | --- | ---: | ---: | ---: |
| `[1,512,512]` | BF16 | 0.1869 ms | 0.1090 ms | 2.02 MiB |
| `[8,512,512]` | BF16 | 0.2343 ms | 0.1070 ms | 16.04 MiB |
| `[8,2048,512]` | BF16 | 0.7118 ms | 0.1640 ms | 64.14 MiB |
| `[64,512,512]` | BF16 | 1.6446 ms | 0.1801 ms | 128.26 MiB |
| `[64,512,512]` | FP16 | 1.5880 ms | 0.2300 ms | 128.26 MiB |
| `[64,512,512]` | FP32 | 1.6549 ms | 0.4306 ms | 256.26 MiB |

The FP16 compiled reduction differed from eager by at most `1.41e-4` in the
FP32 mask-vector gradient; BF16 passed its dtype tolerance, and FP32's maximum
absolute difference was `9.32e-10`. The compiled steady-state incremental peak
was zero because Inductor reused its live allocation pool; the table therefore
reports total peak allocated memory as the non-misleading bound.

The Leviathan epilogue comparison used its model geometry (`hidden=512`,
`d_seed=128`, eight heads, rank 64), BF16, and max-autotune on the same RTX 5070
Ti. The runner alternates both paths to reduce clock and thermal bias. Over 100
paired measurements at 4096 token rows, forward mean changed from `5.6025 ms`
with the separate override to `5.5224 ms` with the fused store (`1.45%` faster;
median `5.5844 -> 5.4217 ms`). A complete forward+backward synthetic step
changed from `26.6944 ms` to `26.4465 ms` (`0.94%` faster), while the medians
were effectively tied (`26.1767` and `26.1841 ms`). At 512 rows, forward means
were tied (`2.2613` and `2.2610 ms`) and the full step improved by `0.72%`.

Both 4096-row paths reported the same total allocated training peak (`101.58
MiB`) because Inductor reused the eliminated override buffer. The fusion
therefore removes a launch/activation traversal and guarantees the correct
producer boundary, but it does not claim a material end-to-end speedup or a
persistent-memory reduction. Backward still needs the mask-gradient reduction,
which bounds the full-step gain.

A route-selection regression audit at the same 4096-row geometry found that a
trainable mask parameter previously made `torch.no_grad()` call the checkpointed
forward op. Selecting the inference op reduced its incremental allocated peak
from `41.51 MiB` to `25.26 MiB` while leaving max-autotune inference latency
effectively unchanged. With every Leviathan parameter frozen and only the mask
vector trainable, the checkpoint-free route reduced the eager synthetic step
from `27.43 ms` to `5.55 ms` and its incremental peak from `100.02 MiB` to
`44.00 MiB`. Normal full-model training still uses the complete forward and
backward and retains its original gradients.
