# Leviathan Triton integration

This repository contains an optional `cut_cross_entropy.leviathan` package.
It adds the Triton LEV embedding kernel without copying NeoLLM's transformer
model or changing the existing CCE implementation.

The integration has three entry points:

- `LeviathanEmbedding` is a small consumer for tests and model integration.
- `LeviathanForCausalLM` demonstrates the LEV embedding plus the existing
  `linear_cross_entropy` loss.
- `replace_leviathan_generator` creates a state-dict-compatible adapter for a
  model that already owns `model.token_generator`.

On CUDA, supported BF16 configurations use the Triton forward/backward path.
Unsupported shapes, CPU execution, disabled use, and Triton launch failures
fall back to the differentiable dict-based reference path. It keeps the
original parameter tensors connected to autograd and honors an explicit
`knot_grid`. `LeviathanGenerator` creates its trainable tensors in
`config.dtype`, so `.cuda()` alone is sufficient to select the BF16 kernel.
When gradients are disabled—or all generator parameters are frozen—a separate
CUDA boundary runs the forward with `save_intermediates=False` and avoids
allocating the training checkpoints.

The package does not set `torch._dynamo.config.capture_dynamic_output_shape_ops`
or any other global Dynamo/`torch.compile` option. The LEV custom-op boundary
has one graph and zero breaks under the default Dynamo configuration. The
compact CCE consumer can still report a break at CCE's pre-existing
`cut_cross_entropy.cce_forward` dynamic-output operator; this integration does
not enable a config flag or modify that CCE operator.

## SM120 tensor-core specialization

The scalar implementation computes the spline basis, `phi`, and `dphi` once in
the chain kernel and recomputes them in the dDelta reduction. The SM120
specialization maps the knot-by-rank contractions to `tl.dot` and fuses
chain+dDelta. The basis and gradient factors stay live within the CTA; they
are not materialized in HBM and are not recomputed by a second kernel.

Automatic dispatch is intentionally narrow. It selects the specialization
only on CUDA compute capability 12.0 or newer for the validated production
geometry `d_seed=128`, `num_knots=16`, and `krank=64`. Other architectures and
geometries retain the deterministic scalar kernels. No public API flag or
global `torch.compile` setting is added. `LEV_DOT=0/1` remains a
developer-only A/B override.

For long-token workloads the fused backward uses `BLOCK_M=128`,
`BLOCK_D=1`, `BLOCK_R=64`, four warps, and one software stage. Short inputs
keep `BLOCK_M=32`. The forward dot kernel uses BM32/w4/s1 with a split-head
grid on SM120 and unrolls its per-seed loop by four. Larger tiles, gather
fusion, TF32x3, and a cuBLAS replacement were measured and rejected because
they were slower or used more memory.

The backward keeps `dM` and `M` independent and multiplies them after loading
both as FP32. An in-place BF16 premultiplication was slightly faster in an
isolated dDelta profile but produced no measurable end-to-end gain on RTX 5090
and increased relative gradient error from roughly 0.03--0.07% to
0.24--0.30%. It is therefore disabled by default. `LEV_PREMUL_DMM=1` remains
available only for developer diagnostics and does not affect the reference
fallback or public API.

The dot operations use `input_precision="ieee"` for the spline chain. The
surrounding model keeps its normal precision policy; the measurements below
used `torch.set_float32_matmul_precision("high")`, BF16 model representation,
and FP32 optimizer masters.

### Prior SM120 kernel measurements

The first optimization stage was measured on an RTX 5070 Ti with a 10 GB test
cap. At `D=2048, N=4096`, the candidate-only profile was approximately 1.1 ms
forward, 3.0 ms backward, and 0.14 GB peak, with one synchronization event and
no N-scaled dDelta workspace. Splitting the forward grid by head reduced its
kernel from about 0.925 to 0.776 ms without adding a persistent buffer. Hoisting
the token-block `dM` and `M` loads out of the seed loop improved backward or
forward+backward medians by roughly 4--14% across four tested shapes, with
unchanged 0.115/0.142/0.333/0.435 GB peaks. Those measurements established the
geometry and liveness design; the RTX 5090 work below validates the second
stage inside the full training graph.

## RTX 5090 full-training measurement

The end-to-end integration measurement used the complete training step rather
than a Leviathan-only microbenchmark: 12 transformer layers, batch 64,
sequence length 512, 32,768 tokens/step, MXFP8 TorchAO linears, BF16 model
representation, CCE, MEAP, MiLe, mu-loss, TWEO/NITP, the fused gradient
stabilizer, AdEMAMix FP32 optimizer masters, `torch.compile`, and matmul
precision `high`. Only the Leviathan kernel route changed between the first
two rows.

| Embedding route | steps/s | GPU ms/step | forward ms | backward ms | peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Leviathan scalar control | 4.206 | 237.71 | 58.02 | 167.58 | 2.798 GB |
| Leviathan SM120 automatic | **5.727** | **174.58** | **45.86** | **116.63** | 2.798 GB |
| Dense embedding control | 6.042 | 165.47 | 42.93 | 106.64 | 3.154 GB |

The specialization improves Leviathan throughput by 36.2% and reduces its
gap to the dense control from 72.24 to 9.11 ms/step. It does not increase
allocated peak memory and recorded zero allocation retries and zero OOMs.
At batch 32 and sequence length 512, the automatic route measured 10.094
steps/s (165,377 tokens/s) and 2.250 GB peak allocated. Compiling that new
shape under max-autotune took 491.8 seconds; compilation time is excluded from
steady-state throughput.

These figures describe an external full-model integration and are not
presented as a standalone repository benchmark. The repository-owned harness
below reproduces the kernel-level numerical and latency comparison without
requiring that model:

```bash
python benchmark/leviathan_backward_compare.py \
  --tokens 31 257 4096 32768 \
  --warmup 2 \
  --steps 5 \
  --include-reference
```

The harness fixes the seed, keeps BF16 parameters and activations, uses FP32
reductions and matmul precision `high`, compares every trainable gradient, and
reports non-finite counts, cosine similarity, relative L2 error, and latency.
The automatic-dispatch regression can be run independently with:

```bash
pytest tests/test_leviathan_runtime_policy.py -q
```
