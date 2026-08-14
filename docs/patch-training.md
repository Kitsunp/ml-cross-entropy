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

## Controlling the patch duration

`PatchTrainingSchedule` exposes the phase boundary to the trainer while keeping
the optimizer step out of the kernel and compiled model:

```python
from cut_cross_entropy import PatchTrainingSchedule, linear_cross_entropy

schedule = PatchTrainingSchedule(
    patch_training_steps=120_000,
    patch_size=4,
    ignore_index=-100,
)

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

## Measured operator feasibility

The final checkout was measured on an RTX 5070 Ti with PyTorch
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

These are isolated CCE operator results, not end-to-end model-step claims.
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
