# NeoLLM MLP in CuTe DSL

## Status

This document is the implementation contract for replacing the complete NeoLLM
training MLP.  Design 1 begins with a checked mathematical reference.  A route is
not considered implemented merely because a central PolyNorm kernel runs: it must
cover the full forward and backward contract described below.

The current reference is `cut_cross_entropy.mlp.neollm_mlp_reference`.  It returns
only the final output.  Intermediate tensors are deliberately not auxiliary outputs
because AOTAutograd may retain every returned tensor until backward.

The implementation is currently between Design 1 and Design 2. Exclusive GUPN has
a CuTe forward and a recomputed CuTe backward. The staged FP8 gate/up wrapper is an
experimental contract test, not the final kernel and not a production default.

## Exact model graph

For the current model, `D=512`, `I=1536`, `fan_ratio=0.0625` and there are twelve
decoder layers.  FAN projects `512 -> 510`; its 34 periodic channels are expanded
with cosine and sine, producing 544 channels.  The graph is:

```text
X
  -> BF16 FAN linear (512 -> 510)
  -> [cos(P), sin(P), Gf] (544)
  -> gate linear (544 -> 1536) -> gate row multiplier
  -> up linear   (544 -> 1536)
  -> exclusive PolyNorm(gate)
  -> dropout
  -> Hadamard with up
  -> down column multiplier
  -> down linear (1536 -> 512)
  -> down row multiplier
  -> Y
```

`exclusive=True` is the model default.  It introduces two trainable logits and
orthogonalizes the quadratic and cubic branches against the normalized linear
branch.  A CuTe route that does not implement this branch must reject it and use the
reference; silently dropping the logits is invalid.

## Numeric contract

The live model tensors remain BF16 and optimizer masters/moments remain FP32.  The
MLP operator neither owns nor updates optimizer state.

The current TorchAO conversion uses its default dynamic tensorwise recipe.  FAN is
not FP8-compatible because its output width is 510.  Gate, up and down are FP8
compatible.  The initial CuTe design preserves:

- independent tensorwise scales for gate and up weights;
- shared input quantization for their common FAN input;
- E4M3 forward operands and the TorchAO backward policy;
- FP32 accumulation/reductions;
- the existing logical BF16 round points, even if an intermediate stops reaching HBM;
- `fast_accum=True` only where the baseline forward uses it, and conservative
  accumulation for grad-input and grad-weight;
- deterministic Philox replay between forward and backward;
- fixed-order parameter reductions without global floating-point atomics.

The source of truth for conversion is `apply_fp8_conversion` in the training
script. It calls `convert_to_float8_training` without a custom config and filters
modules as follows:

- protect `StackMemory`, `LeviathanGenerator`, `embed_tokens`, `lm_head` and
  `token_generator`, including their descendants;
- convert only `nn.Linear` modules whose input and output widths are divisible by
  16;
- leave every other module in BF16.

For the current MLP this means FAN `512 -> 510` stays BF16 while gate
`544 -> 1536`, up `544 -> 1536` and down `1536 -> 512` use the default TorchAO
recipe. The staged dual projection has been checked bit-for-bit against two
`Float8Linear` instances for forward, `dF`, `dW_gate` and `dW_up`.

There are two unavoidable global barriers under dynamic tensorwise scaling:
`amax(F)` before gate/up and `amax(Z)` before down.  Therefore one public MLP
operator maps to a family of internal kernels rather than one grid-wide kernel.

## Design 1: safe staged operator

```text
K_FAN
  -> temporary F_BF16 + partial amax(F)
reduce_amax_F
K_GU
  -> gate/up with a shared F tile and independent weight scales
K_P
  -> gate multiplier + exclusive PolyNorm + dropout + Hadamard
  -> down column multiplier + temporary Z_BF16 + partial amax(Z)
reduce_amax_Z
K_DOWN
  -> quantize Z while loading + down GEMM + row multiplier
```

Design 1 keeps the numerical boundaries observable while establishing exact
equivalence.  It is not the final performance layout.

## Design 2: GUPN fusion

After Design 1 passes the long-training gates, `K_GU` and `K_P` become one cluster
region:

```text
TMA F/W -> dual gate/up MMA -> logical BF16 rounding
        -> exclusive PolyNorm reductions in FP32
        -> Philox -> Hadamard -> column multiplier -> Z_BF16
```

The SM120 implementation has two measured families:

- `K_REG`, when active gate/up fragments fit without spills;
- `K_SMEM_SUBTILE`, when a swizzled SMEM backing store improves occupancy.

XOR is retained as an autotuning candidate for non-MMA gate/up storage.  MMA operand
layouts follow the CuTe-provided layout for the selected SM120 instruction.  XOR is
accepted only when Nsight reports fewer serialized shared-memory wavefronts and the
end-to-end kernel is faster.

DSMEM transports only the small row reductions.  It never transports complete gate
or up tiles.  The forward reduction payload per row is the PolyNorm statistics plus
the two exclusive projection dots/norms required by the actual model.

### Implemented GUPN checkpoint

The implemented GUPN kernel fuses the gate-row multiplier, exclusive PolyNorm,
Philox dropout, Hadamard product and down-column multiplier. One CTA handles one
row in forward. Backward handles eight rows per CTA, replays Philox from four
seeds, recomputes all PolyNorm branches and produces deterministic FP32 partials.
Three small reduction kernels finalize the gate-row, down-column and six scalar
parameter gradients. It uses no global floating-point atomics.

The public training custom op saves gate/up inputs and small parameters, but does
not save PolyNorm branches, the dropout mask, Hadamard output or `Z`. This is a
completed central-stage checkpoint, not yet the final full-MLP lifetime contract.

### Experimental staged FP8 body

`fp8_gupn` currently validates the next autograd boundary:

```text
F_BF16
  -> shared dynamic cast of F
  -> two TorchAO-exact scaled_mm projections
  -> CuTe GUPN
  -> Z_BF16 only
```

Its backward saves `F`, gate/up weights, multipliers and four seeds, then recomputes
gate/up before invoking CuTe GUPN backward. A saved-tensor test rejects retention
of `(rows, 1536)` gate/up activations. This version deliberately remains staged:
because the scaled matrix multiplications execute inside an opaque custom op,
Inductor cannot fuse their casts or recomputation. It proves memory lifetime but
is not fast enough to replace the MLP.

The physical dual CuTe kernel therefore has a stricter target:

- receive two raw FP8 weights and two independent tensorwise scales;
- stage one `F` tile once;
- issue gate and up MMA operations from the same staged `F`;
- keep two FP32 accumulators and apply the two dequantization scales separately;
- never concatenate weights or construct a 3072-element scale vector;
- feed logical BF16 gate/up fragments directly into GUPN.

## Backward and activation lifetime

The final public custom op has this contract:

```text
forward(...) -> Y only
backward(dY) -> recompute F, gate, up, PolyNorm and dropout mask
```

Temporary materialization is not a saved activation.  F and Z may exist as internal
workspaces around the two global scale barriers, but they must not be returned as
auxiliary outputs or saved once per layer until backward.

Backward performs:

1. Down pre-backward: produce the `dT` scale and `dr_down` without materializing dT.
2. Down grad-input/grad-weight FP8 GEMMs with conservative accumulation.
3. Recompute FAN, gate and up using the forward scales and BF16 boundaries.
4. Two fixed PolyNorm reduction stages: moments/exclusive projections, then gradient
   dots and parameter partials.
5. Dual grad-input: accumulate gate and up contributions into one `dF` accumulator.
6. Dual grad-weight: reuse the common FAN input for `dW_gate` and `dW_up`.
7. FAN backward in BF16/FP32.

No model-level checkpoint is part of the final design.  Rematerialization belongs to
the operator's explicit autograd implementation.

## Acceptance gates

Every increasingly fused route must pass all gates before it can become the default:

1. Forward and every parameter/input gradient match the reference, including
   exclusive logits and nonzero dropout.
2. Five compiled optimizer steps run with unique masks and no graph breaks or Dynamo
   recompilation growth.
3. The test harness, not production code, caps process-visible VRAM at 10 GB.
4. No NaN/Inf and bounded error across BF16, FP32 and the TorchAO FP8 training path.
5. No local-memory spills for a winning geometry.
6. Peak allocated, peak reserved and incremental transient memory are measured after
   clearing warmup gradients.
7. Performance is measured across batches, sequence lengths and valid geometries on
   RTX 5070 Ti and RTX 5090; a microbenchmark alone cannot select the production route.
8. Long runs compare loss, gradient norm, FP8 scales, PolyNorm statistics and
   multiplier distributions before enabling the backend by default.

## SM120 implementation basis

RTX 5090 is SM120.  SM100 `tcgen05`/TMEM examples are design references, not code
templates.  Dense tensorwise E4M3 begins from the CuTe warp-level FP8 MMA supported
on SM120, with FP32 accumulators.  Block-scaled SM120 examples are relevant only to a
future recipe change and are not substituted for TorchAO tensorwise scaling.

The initial tuning space is deliberately bounded: valid MMA-derived tiles, two to
four pipeline stages, REG/SMEM residency, plain/padded/XOR layouts, and cluster sizes
that pass occupancy checks.  Each candidate is compiled once, cached by architecture
and shape, and judged by correctness before latency.

## Reproducible baseline measurements

`benchmark/mlp_profile.py` recreates the full module graph, including exclusive
PolyNorm and all three multipliers.  `--precision fp8` applies the default TorchAO
dynamic tensorwise conversion only to compatible linears; FAN remains BF16.  The
runner clears warmup gradients before its memory baseline and measures forward plus
backward.  Its 10 GB cap exists only in this test process.

RTX 5090, one layer, batch 1, length 512, five measured steps:

```bash
python benchmark/mlp_profile.py \
  --precision fp8 \
  --compile-mode max-autotune \
  --batch 1 \
  --sequence-length 512 \
  --layers 1 \
  --warmup 2 \
  --steps 5 \
  --max-test-vram-gib 10
```

Six-layer interaction check:

```bash
python benchmark/mlp_profile.py --precision fp8 --compile-mode max-autotune \
  --batch 1 --sequence-length 512 --layers 6 --warmup 2 --steps 5
```

The same command with `--precision bf16` or `--precision fp32` provides the unfused
numeric baselines.  Results must include the library and CUDA versions emitted by the
runner; tables are updated only from saved JSON outputs, not estimated values.

### RTX 5090 measured checkpoints

Environment: RTX 5090 (SM120), PyTorch `2.13.0+cu130`, CUDA `13.0`, TorchAO
`0.18.0`, NVIDIA CUTLASS DSL `4.7.0`. All memory caps below are applied only by
the benchmark process.

Exclusive GUPN forward, BF16 output:

| Rows x width | max-autotune | CuTe best | Latency reduction | CuTe scratch excluding output |
|---:|---:|---:|---:|---:|
| 512 x 1536 | 0.06634 ms | 0.02750 ms | 58.5% | 0 B |
| 32768 x 1536 | 0.60280 ms | 0.21147 ms | 64.9% | 0 B |

Exclusive GUPN forward plus backward:

| Rows x width | max-autotune | CuTe | Latency reduction | Reserved pool change |
|---:|---:|---:|---:|---:|
| 512 x 1536 | 0.31278 ms | 0.25192 ms | 19.5% | 64 -> 44 MiB |
| 32768 x 1536 | 2.45339 ms | 1.44418 ms | 41.1% | 1780 -> 724 MiB |

The one-layer full FP8 MLP at batch 1, sequence length 512 is compute-bound by the
remaining FP8 GEMMs: reference and CuTe GUPN medians are 0.88149 and 0.88102 ms.
At six layers they are 5.32379 and 5.35621 ms, so the central-only route is not
enabled as a performance replacement.

At batch 64, sequence length 512, the experimental recomputed FP8 body reduces the
isolated reserved peak from 2346 to 1796 MiB but increases step latency from
5.356 to 10.454 ms. This is the measured reason the final design requires a
physical dual MMA and a full operator backward rather than Python-level
recomputation inside an opaque custom op.

Reproduce the central training comparison in isolated processes:

```bash
python benchmark/mlp_gupn_train_profile.py --implementation max_autotune \
  --rows 32768 --hidden 1536 --warmup 20 --iterations 100
python benchmark/mlp_gupn_train_profile.py --implementation cute \
  --rows 32768 --hidden 1536 --warmup 20 --iterations 100
```

Reproduce the exact full-layer routes:

```bash
python benchmark/mlp_profile.py --precision fp8 --compile-mode max-autotune \
  --batch 64 --sequence-length 512 --layers 1 --warmup 5 --steps 20 \
  --gupn-route reference
python benchmark/mlp_profile.py --precision fp8 --compile-mode max-autotune \
  --batch 64 --sequence-length 512 --layers 1 --warmup 5 --steps 20 \
  --gupn-route fp8-cute
```
