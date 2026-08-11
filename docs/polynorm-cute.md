# CuTe PolyNorm and fused dropout

## Scope and FP8 boundary

The default CuTe entry point implements only `PolyNorm`. It does not absorb
`gate_proj`, `up_proj`, a learnable column multiplier, or `down_proj.linear`.
Those modules remain visible to TorchAO, including their FP8 weight/activation
scaling and amax tracking.

The resulting training flow is:

```text
gate_proj (TorchAO FP8) ──> PolyNorm + optional dropout (CuTe, FP32 math, BF16 handoff)
up_proj   (TorchAO FP8) ──> elementwise multiply
                            └─> column multiplier ─> down_proj.linear (TorchAO FP8)
```

Callers pass their existing training dropout probability directly to
`polynorm()`. A zero probability compiles dropout out; evaluation code should
pass zero. The library does not rewrite model classes or add a production flag.

Moving the mask before the elementwise product is mathematically valid:

```text
dropout(poly * up) = (mask / (1 - p)) * poly * up
                   = dropout(poly) * up
```

`up_proj`, the column multiplier, and the FP8 down projection are therefore
unchanged. In evaluation mode the effective probability is zero.

## Kernel design

The supported CuTe path uses the following design:

- BF16 or FP32 input and matching PolyNorm parameters;
- FP32 moments, normalized branches, direct derivatives, and reductions;
- an output in the input dtype, giving a BF16 handoff to the next training
  operation without a full FP32 activation tensor;
- saved FP32 inverse norms only (`rows x 3`), rather than materialized powers or
  normalized branches;
- backward recomputation of powers and dropout decisions;
- deterministic FP32 row/parameter reduction;
- an XOR shared-memory layout `Swizzle<2,4,3>` in backward. The same logical
  coordinates are used for stores and loads. The larger `Swizzle<3,4,3>` is not
  valid for this vector-four layout and is not forced onto it;
- explicit launch on `torch.cuda.current_stream()`, including CUDA Graph
  capture;
- one shape/dtype/probability-specialized CuTe compilation cached per device.

Unsupported shapes and environments use the independent PyTorch reference.
The current CuTe path requires a contiguous CUDA tensor, hidden size divisible
by four, `eps=1e-6`, and no exclusive-branch logits. Exclusive PolyNorm remains
functional through the fallback.

Inside a Dynamo-compiled model, tensors below 8 Mi elements stay as the
PyTorch expression so Inductor can fuse them with their surrounding graph.
Larger tensors use the CuTe custom op. This internal, static dispatch avoids
the fixed custom-op cost at small batch/sequence sizes without adding a user
flag. Eager execution continues to prefer CuTe because an unfused eager
reference would launch many separate kernels.

## Stateless Philox dropout

`dropout_p=0` compiles out Philox and does not advance the PyTorch RNG. A
positive probability generates four 32-bit words from PyTorch's CUDA RNG for
each invocation:

- words 0 and 1 form the 64-bit Philox key;
- words 2 and 3 occupy the high counter dimensions;
- the low and high words of the element-block index occupy the remaining
  counter dimensions.

Each thread runs one compile-time-unrolled Philox4x32-10 transform and consumes
all four results. The output is compared with an integer threshold
`ceil(p * 2^32)`, avoiding FP32 rounding in the random-number-to-uniform
conversion. No random tensor or dropout mask is written to global memory.
Backward saves the four words and reconstructs the same decisions.

This arrangement also composes with activation checkpointing: the words are
drawn from PyTorch's RNG, whose state checkpointing already preserves. Five
compiled steps produced five distinct masks and one Dynamo graph in validation.

PTX/SASS inspection on SM120 confirmed that the Philox products lower to
`mul.wide.u32` and `IMAD.WIDE.U32`; the rounds do not use a general 64-bit
multiply sequence.

## RTX 5090 measurements

### Isolated kernel boundary

Measurements below use shape `32768 x 1536`, BF16, training forward plus
backward, `dropout_p=0.1`, and a benchmark-process-only 10 GiB allocation cap.
Production code has no VRAM limit.

Environment: RTX 5090 (SM120), PyTorch 2.13.0+cu130, CUDA runtime 13.0,
nvidia-cutlass-dsl 4.6.1, Triton 3.7.1, TorchAO 0.18.0, Python 3.13.

| implementation | forward | backward | median total | relative to CuTe | incremental peak allocation |
|---|---:|---:|---:|---:|---:|
| CuTe registered API | 0.1822 ms | 0.2089 ms | 0.3903 ms | 1.00x | 202.64 MB |
| `torch.compile`, default | 0.1984 ms | 0.6681 ms | 0.8686 ms | 2.23x slower | 453.38 MB |
| `torch.compile`, max-autotune | 0.3508 ms | 0.7741 ms | 1.1254 ms | 2.88x slower | 0 MB after pool capture¹ |

The direct registered CuTe path reduces latency by about 55.1% versus the default
compiled reference and 65.3% versus max-autotune. Incremental peak allocation
is about 55.3% lower than the default compiled path.

¹ Max-autotune had already moved its live buffers into a persistent CUDA Graph
pool before the measurement baseline, so its incremental allocation reads zero.
That is not evidence of zero memory use and is not directly comparable with
ordinary allocations. Its resident reserved memory after warmup was 1.212 GB,
versus 0.757 GB for this CuTe benchmark process. End-to-end model memory must be
measured at the model boundary rather than inferred from this isolated pool.

The kernel-only measurement is 0.3381 ms and 202.25 MB for both `p=0` and
`p=0.1`. A minimal registered probe measured 0.3718 ms; the common comparison
runner above measured 0.3903 ms and is the number used for A/B claims. The
registered API includes dispatcher and RNG-word generation costs. A separately compiled microbenchmark
can copy external inputs into a CUDA Graph static pool, so it must not be used
as a proxy for the full MLP where projection outputs are graph-internal.

### Compiled model-path geometry matrix

The following comparison compiles the hybrid PolyNorm entry point itself with
`max-autotune`, matching the requested model compilation policy. Both reference
columns run in separate clean processes. "Inductor" means the internal
small-tensor dispatch intentionally left the reference expression visible to
the compiler; "CuTe" means the custom kernels were used.

| batch | sequence | PolyNorm H | rows | route | hybrid max-autotune | reference default | reference max-autotune | speedup vs same mode |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 128 | 512 | 128 | Inductor | 0.1342 ms | 0.1612 ms | 0.1357 ms | 1.01x |
| 1 | 512 | 1536 | 512 | Inductor | 0.1768 ms | 0.1224 ms | 0.1755 ms | 0.99x |
| 8 | 512 | 1536 | 4,096 | Inductor | 0.1948 ms | 0.1702 ms | 0.2231 ms | 1.14x |
| 16 | 512 | 2048 | 8,192 | CuTe | 0.2760 ms | 0.3153 ms | 0.3859 ms | 1.40x |
| 16 | 2048 | 1536 | 32,768 | CuTe | 0.6710 ms | 0.8828 ms | 1.1538 ms | 1.72x |
| 64 | 512 | 1536 | 32,768 | CuTe | 0.6750 ms | 0.8811 ms | 1.1280 ms | 1.67x |

The last row matches the current training configuration's batch and sequence
length. It is 40.2% lower latency than the same `max-autotune` reference and
23.4% lower than the default compiled reference. The two batch-1 cases do not
claim a CuTe acceleration: the internal fallback is effectively tied with the
equivalent max-autotune path, within measurement noise.

At `64 x 512 x 1536`, resident reserved memory after warmup was 0.960 GB for
the hybrid path, 0.858 GB for the default compiled reference, and 1.212 GB for
the max-autotune reference. These figures include compiler/CUDA Graph pools;
their incremental allocation is zero after capture and must not be interpreted
as zero VRAM use. Relative to the requested max-autotune mode, the isolated
hybrid process reserved about 20.8% less memory.

FP32 smoke validation at `64 x 1536` measured maximum absolute errors of
`1.43e-6` for output and `1.91e-6` for `grad_x`. BF16 uses the same FP32
accumulation and casts only at the consumer boundary. For `p=0.1`, five compiled
steps were finite, used one Dynamo graph, produced five distinct outputs, and
measured discarded fractions between 9.79% and 10.16%.

The long-run stability runner completed 5,000 compiled forward/backward steps
with 51 periodic checks. Every checked output and gradient was finite, all 51
checked outputs were distinct, Dynamo retained one graph, and the measured
discard fraction stayed between 9.99195% and 10.00875%. Its peak-memory field
includes the deliberately materialized `isfinite` validation tensors and is
not used in the performance table.

### Five-step training boundary with FP32 masters

The integrated runner uses the current training geometry: batch 64, sequence
512, model hidden 512, MLP intermediate/PolyNorm hidden 1536, fused dropout
`p=0.1`, `torch.set_float32_matmul_precision("high")`, and full-graph
`max-autotune`. Each process is capped at 10 GiB for the test only.

| projection mode | median step | min-max | incremental peak | live-weight delta | FP32-master delta |
|---|---:|---:|---:|---:|---:|
| FP32 | 5.6995 ms | 5.6940-5.8401 ms | 268.44 MB | 1.5605e-4 | 1.5605e-4 |
| BF16 | 3.1746 ms | 3.1504-3.3572 ms | 335.55 MB | 0 | 1.5712e-4 |
| TorchAO FP8 | 3.2864 ms | 3.2740-3.4685 ms | 335.55 MB | 0 | 1.5721e-4 |

All fifteen measured steps and their gradients were finite. Every mode kept a
single Dynamo graph and compiled two CuTe entries (forward and backward).
TorchAO replaced all three linears with `Float8Linear`; the compiled graph used
`aten._scaled_mm` with FP8 operands, so this was not FP8 emulation.

The update loop holds one persistent FP32 master per parameter, converts each
gradient to FP32, updates the master, and then copies it to the visible model
dtype. At this learning rate five BF16/FP8 steps did not cross a BF16 weight
quantization interval, so a zero visible delta is expected while the nonzero
master delta proves that the update was retained. The small update rule is a
precision/data-flow probe, not a replacement implementation of AdEMAMix.

An additional clean-process FP8 A/B kept all three TorchAO linears and changed
only the PolyNorm backend. CuTe measured 3.2740 ms versus 4.2491 ms for the
Inductor reference: 1.30x, or about 23.0% lower step latency. Both reached the
same 383.27 MB allocated peak. CuTe's peak reserved pool was 1.118 GB versus
1.437 GB for the reference, 318.77 MB or 22.2% lower. Reserved CUDA Graph pools
are reusable, so this difference must not be multiplied by the number of model
layers or presented as an end-to-end model saving.

The same A/B with six residual MLPs in one full graph used 18 real TorchAO
`Float8Linear` modules. The six PolyNorm calls reused two shape-specialized CuTe
entries rather than compiling per layer. CuTe measured 19.0023 ms versus
24.7726 ms: 1.30x, or 23.3% lower latency. Both reached the same 455.16 MB
allocated peak. Peak reserved memory was 3.047 GB for CuTe and 3.230 GB for the
reference, a 182.45 MB (5.65%) reduction. All five steps and gradients were
finite and Dynamo retained one graph. Six CuTe layers took 5.80x the one-layer
time, showing near-linear scaling without a per-layer compilation penalty.

Reserved-pool savings are not an invariant of the kernel. An opaque custom-op
boundary can change AOTAutograd rematerialization and buffer-lifetime choices;
a host graph with additional preprocessing may reserve more CUDA Graph memory
even when the isolated boundary reserves less. The allocated peak and reserved
pool must therefore be measured on the complete host graph. This document does
not claim an end-to-end VRAM reduction from the isolated figures above.

No fresh RTX 5070 Ti measurement has been made for this final PolyNorm-only
boundary. Do not reuse numbers from the earlier column-fused experiment; run
the commands below on that device before claiming a 5070 Ti speedup.

## Installation and use

CuTe is optional. If it cannot be imported, the PyTorch implementation remains
available:

```bash
python -m pip install ".[polynorm]"
```

For an already installed checkout, installing
`nvidia-cutlass-dsl==4.6.1` directly is equivalent.

Direct use:

```python
from cut_cross_entropy.polynorm import polynorm

y = polynorm(x, weight, bias, dropout_p=0.1 if training else 0.0)
```

## Reproducing the comparison

The reference benchmark accepts a limit only for its own process:

```bash
python -m benchmark.polynorm_profile \
  --backend cute \
  --batch 64 --sequence 512 --hidden 1536 \
  --dtype bfloat16 --output-dtype input --dropout-p 0.1 \
  --warmup 10 --iterations 100 --memory-limit-gib 10

python -m benchmark.polynorm_profile \
  --backend torch_compile --compile-mode default \
  --batch 64 --sequence 512 --hidden 1536 \
  --dtype bfloat16 --output-dtype input --dropout-p 0.1 \
  --warmup 10 --iterations 100 --memory-limit-gib 10

python -m benchmark.polynorm_profile \
  --backend torch_compile --compile-mode max-autotune \
  --batch 64 --sequence 512 --hidden 1536 \
  --dtype bfloat16 --output-dtype input --dropout-p 0.1 \
  --warmup 10 --iterations 100 --memory-limit-gib 10
```

To inspect generated code during a benchmark-only compilation:

```bash
CUTE_DSL_KEEP=ptx,cubin,sass \
CUTE_DSL_DUMP_DIR=/tmp/cute-polynorm-artifacts \
python your_polynorm_probe.py
```

These environment variables are diagnostic only and are not read or set by
the production implementation.

Long-run graph/RNG validation:

```bash
python -m benchmark.polynorm_stability \
  --rows 32768 --hidden 1536 --dtype bfloat16 --dropout-p 0.1 \
  --steps 5000 --check-every 100 --compiled --memory-limit-gib 10
```

Curated geometry matrix with each backend in a clean subprocess:

```bash
python -m benchmark.polynorm_matrix \
  --dtype bfloat16 --dropout-p 0.1 \
  --warmup 6 --iterations 30 --memory-limit-gib 10
```

Five training steps with FP32 masters and the current MLP boundary:

```bash
python -m benchmark.polynorm_training --mode fp32 \
  --batch 64 --sequence 512 --hidden 512 --intermediate 1536 \
  --dropout-p 0.1 --warmup 2 --steps 5 --compiled --memory-limit-gib 10

python -m benchmark.polynorm_training --mode bf16 \
  --batch 64 --sequence 512 --hidden 512 --intermediate 1536 \
  --dropout-p 0.1 --warmup 2 --steps 5 --compiled --memory-limit-gib 10

python -m benchmark.polynorm_training --mode fp8 \
  --polynorm-backend cute \
  --batch 64 --sequence 512 --hidden 512 --intermediate 1536 \
  --dropout-p 0.1 --warmup 2 --steps 5 --compiled --memory-limit-gib 10

python -m benchmark.polynorm_training --mode fp8 \
  --polynorm-backend reference \
  --batch 64 --sequence 512 --hidden 512 --intermediate 1536 \
  --dropout-p 0.1 --warmup 2 --steps 5 --compiled --memory-limit-gib 10

python -m benchmark.polynorm_training --mode fp8 \
  --polynorm-backend cute --layers 6 \
  --batch 64 --sequence 512 --hidden 512 --intermediate 1536 \
  --dropout-p 0.1 --warmup 2 --steps 5 --compiled --memory-limit-gib 10

python -m benchmark.polynorm_training --mode fp8 \
  --polynorm-backend reference --layers 6 \
  --batch 64 --sequence 512 --hidden 512 --intermediate 1536 \
  --dropout-p 0.1 --warmup 2 --steps 5 --compiled --memory-limit-gib 10
```
