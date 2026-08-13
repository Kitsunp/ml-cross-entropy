# Graph-stable patch-level training

Patch-level training is an opt-in CCE path for predicting a fixed group of
future tokens from one hidden-state row.  It follows the patch objective from
[Beyond Next Token Prediction: Patch-Level Training for
LLMs](https://arxiv.org/abs/2407.12665) without materializing a
`rows × patch_size × vocabulary` logit tensor.

## Public contract

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

## Relationship to the paper

The paper's complete method contains model/data responsibilities in addition to
the loss implemented here:

1. Every consecutive group of K token embeddings is averaged, without adding
   projection parameters. A raw context of `K * T` tokens therefore becomes T
   patch rows before entering the otherwise unchanged sequence model.
2. A single ordinary output head predicts all K tokens of the next patch. The
   authors explicitly avoid K separate heads so the patch model remains aligned
   with the later token model.
3. Patch training runs for `N * lambda` steps and token training for
   `N * (1 - lambda)` steps. Only model parameters initialize the second stage;
   the paper resets optimizer and learning-rate-scheduler state at the boundary.

CCE implements item 2's multi-target loss and its memory-efficient backward. It
does not average token embeddings, construct patch-aligned labels, choose the
stage boundary, or reset training state. Those operations remain explicit in
the model/data/trainer pipeline, preventing a loss flag from silently changing
the architecture or optimizer.

The main experiments used `K=4` and `lambda=2/3`, giving the theoretical cost

```text
lambda / K + 1 - lambda = (2/3) / 4 + 1/3 = 0.5
```

for 180,000 total steps. Their measured eight-A100 patch stage was 3.50x faster
after reducing gradient accumulation to preserve global batch size, and the
reported total runtime was approximately 0.523x rather than the theoretical
0.5x because of data-loading and gradient-synchronization overhead.

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

The ordinary CCE vocabulary kernel computes the dense term once per row.  A
sparse Triton kernel applies the K target corrections to `dE`, `dC`, and bias.
This avoids both a K-fold vocabulary reduction and a repeated embedding tensor.
Duplicate target IDs are valid and contribute once per occurrence.

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

## Precision policy

Each correct-class logit is extracted from the exact vocabulary tile that also
feeds the LSE reduction. This preserves the accumulation and rounding identity
between both values; a separate scalar or K-target dot reduction could otherwise
produce a negative/nonzero cross-entropy for a one-class or highly concentrated
distribution. The sparse target correction is separated only in backward,
where it is algebraically independent from the dense softmax term.

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

## Measured forward feasibility

The following numbers are retained as feasibility evidence, not as final
forward+backward claims for this production patch.  They were measured on an
RTX 5070 Ti with PyTorch `2.12.1+cu130`, Triton `3.7.1`, BF16, 512 rows,
`D=512`, `V=65,536`, `K=4`, bias, `cce_exact`, MiLe `gamma=1`, μ-loss
`lambda=1e-4`, five explicit warmups and 20 requested repetitions.  The test
process—not the library—was capped at 10 GiB.

| Forward route | Median latency | Incremental peak | Delta vs one-target base |
|---|---:|---:|---:|
| Existing one-target CCE | 0.8956 ms | 41 KiB | baseline |
| Patch phase, four valid targets | 0.7960 ms | 74 KiB | −11.1% latency, +33 KiB |
| Token phase, one valid plus three ignored | 0.8036 ms | 74 KiB | −10.3% latency, +33 KiB |

In a separate algorithmic comparison using the same core geometry, one-LSE
patch CCE measured 0.8179 ms and repeating the embedding through ordinary CCE
K times measured 1.9928 ms: 2.44× speedup and 59.0% lower latency.  Incremental
peaks were 75,776 bytes and 2,150,400 bytes respectively.  The fixed K=4 target
tensor itself adds 12 KiB over K=1 at 512 rows; the measured total incremental
difference versus the one-target base was 33 KiB after including weights and
target-loss temporaries.

These are operator-forward results.  Rerun the benchmark below on the final
checkout before quoting training-step speed or memory.

## Reproduction

Forward, with the same extensions and a 10 GiB test-process ceiling:

```bash
python benchmark/cce_patch.py \
  --rows 512 --dim 512 --vocab 65536 --patch-size 4 \
  --dtype bf16 --mile --mu-loss \
  --warmup 5 --repetitions 20 --max-test-vram-gib 10
```

Forward plus backward:

```bash
python benchmark/cce_patch.py \
  --rows 512 --dim 512 --vocab 65536 --patch-size 4 \
  --dtype bf16 --mile --mu-loss --training \
  --warmup 5 --repetitions 20 --max-test-vram-gib 10
```

The benchmark clears warmup gradients and synchronizes before sampling the
memory baseline.  Its JSON records GPU, PyTorch/Triton versions, dtype, shape,
warmup/repetition settings, absolute latency, incremental peak bytes, and
deltas versus one-target CCE.

## Verification coverage

`tests/test_cce_patch.py` covers dense forward/backward agreement, bias,
duplicates, ignored/negative/out-of-range targets, MiLe, μ-loss, token-phase
equivalence, unsupported-option validation, and five compiled steps spanning
the phase transition while asserting one graph and zero graph breaks.
