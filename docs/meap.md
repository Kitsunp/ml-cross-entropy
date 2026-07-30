# MEAP with CCE, MiLe, and mu loss

This repository implements the pretraining form of
[Mask-Enhanced Autoregressive Prediction](https://arxiv.org/abs/2502.07490).
The implementation follows the separation used by the
[released MEAP training code](https://github.com/Lieisyourlie/MEAP): MEAP
corrupts autoregressive inputs, while CCE computes the objective from the
resulting hidden states and the original clean labels.

![MEAP kernel data flow](../assets/meap_kernel.svg)

## How to read the diagram

`T` is the padded sequence length, `N` is the number of eligible positions in
one sequence after exclusions, and `K` is the exact number that MEAP replaces.
The Triton launch uses one program per batch row.

- **Panel (a)** loads the IDs and, when supplied, the existing padding or
  eligibility mask. A prefix sum simultaneously counts eligible positions and
  maps sparse positions onto the dense rank domain `[0, N)`. `exclude_last`
  acts before this count, so the protected final position is not part of `N`.
- **Panel (b)** permutes those dense ranks using two Philox-derived sequence
  keys and twelve fixed swap-or-not rounds. Every round pairs ranks and gives
  both members of a pair the same swap decision, so each round is an
  involution and their composition remains a bijection.
- **Panel (c)** compares each unique permuted rank with `K`. The predicate
  `permuted_rank < K` therefore selects exactly `K` distinct positions without
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
| Program shape | `BLOCK_T = next_power_of_2(sequence_length)` | One static lane domain covers the complete row; lanes beyond `T` are masked. |
| Launch | Four warps through `BLOCK_T=512`, eight above it, one stage | Fixed launch policy with no runtime autotuner lookup. |
| Compile-time flags | Existing-mask presence, padding-mask convention, `exclude_last`, and `return_mask` are `tl.constexpr` | Unused mask reads, branches, and diagnostic stores are removed from the compiled specialization. |
| Row state | Eligibility, prefix ranks, permutation state, and selection stay in registers | No global random-score, sorting workspace, top-k workspace, or atomic counter. |
| Required I/O | Read IDs and an optional existing mask; write masked IDs | Global memory traffic is linear in the input size. |
| Optional I/O | Write one boolean per position only for `return_mask=True` | Training avoids the diagnostic allocation by default. |
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
4. Run the ordinary causal model with the original causal and padding masks.
5. Pass its hidden states and the clean labels to `linear_cross_entropy`.

MEAP is not part of the CCE loss kernel. Calling it after the model or from
inside CCE would be too late to change the contextual hidden states.

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
- Each row receives `max(1, floor(mask_ratio * eligible_tokens))` replacements
  when the ratio and eligible count are positive.
- Sampling is without replacement and independent between rows.
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
   `[0, N)`, even when eligibility contains holes.
2. Philox generates two keys per sequence, rather than one random value per
   token.
3. Each of twelve swap-or-not rounds draws a keyed pivot `p` in `[0, N)` and
   pairs rank `x` with `(p - x) mod N`.
4. A keyed hash of the pair chooses whether both partners swap or both stay.
   Because both ranks use the same pair identifier and decision, every round
   is an involution and therefore a bijection.
5. The composition of the twelve rounds is a permutation over the exact
   `[0, N)` domain. A token is selected exactly when its permuted rank is below
   `K`.

The permuted ranks are unique and exactly `K` positions satisfy `rank < K`:
there are no collisions, rejection duplicates, data-dependent retry loops,
atomic updates, or approximate Bernoulli counts. Twelve fixed rounds make the
selection work `O(T)` with the same execution structure for dense and padded
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
- MiLe: enabled, `gamma=1.0`, detached and mean-normalized weights
- mu loss: enabled, `mu_loss_lambda=1e-4`

MEAP changes the context before the Transformer. MiLe then reweights the clean
next-token losses using detached predictive entropy. Mu loss adds
`1e-4 * ||mean(classifier, dim=0)||^2`. Neither MiLe nor mu loss changes how
MEAP positions are sampled.

Validation and inference should use clean inputs (`enabled=False`). A corrupted
validation pass may be reported separately as a robustness diagnostic, but it
is not the clean language-model NLL.
