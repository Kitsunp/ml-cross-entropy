# Fused Triton REPO-GRAPE

This experimental operator implements the positional REPO-GRAPE region as an
independent Triton custom op. It lives in the CCE repository for distribution
and testing convenience; it does not call, modify, or fuse CCE.

## Mathematical contract

For raw REPO assignments `z`, runtime positions `i`, and one learned FP32
coefficient per REPO head, the active coordinate is

```text
u[b,h,i] = i + alpha[h] * (z[b,h,i] - i)
theta[h,r] = inv_freq[r] * exp(log_scale[h,r])
phi[b,h,i,r] = u[b,h,i] * theta[h,r]
```

Queries use their own `(cos(phi), sin(phi))` rotation. Under GQA, one key is
shared by several query heads. Its rotation is the normalized circular mean of
those query rotations. If the circular resultant is numerically zero, the
operator deterministically uses the identity rotation; it does not divide by a
near-zero norm.

IHA with `P=2` is represented by `sequence_pseudo_factor=2`:

```text
u[b,h,i,p] = 2 * u[b,h,i] + p,  p in {0, 1}
```

The validated production scope is IHA `P=1` (no sequence expansion) and `P=2`.
No performance or stability claim is made for `P>2`.

`momentum_gamma` is an optional model extension, not part of base REPO-GRAPE.
When nonzero, it applies

```text
y[t] = x[t] + gamma * (x[t] - x[t-1])
```

after the positional action. Setting it to zero specializes out all previous-
token loads and temporal gradient terms.

## Fused region and numerical policy

The fast path fuses:

1. optional Q/K RMS normalization;
2. linear-blend coordinate construction;
3. scaled GRAPE frequencies and trigonometry;
4. per-query rotation and the GQA circular projection;
5. the optional temporal extension;
6. the corresponding backward, including RMS gradients;
7. compact FP32 parameter reductions.

It intentionally does not fuse attention or CCE.

Trigonometry, coordinates, norms, circular means, parameter gradients, and
reduction partials use FP32. BF16 inputs preserve the reference's pathwise BF16
rounding in backward; regrouping the algebra in FP32 was rejected because it
changes long-run BF16 dynamics. FP32 inputs retain FP32 gradients. Live output
dtype is selected explicitly and no FP32 master policy is imposed by this op.

The implementation registers an autograd formula for a `torch.library.triton_op`
boundary. It is compatible with `torch.compile(..., mode="max-autotune",
fullgraph=True)` without changing Dynamo limits, enabling CUDA-graph flags, or
adding a compile-mode flag to the library.

## Geometry and data movement

Dispatch is shape based:

- row kernels handle very short or non-temporal forms;
- stream kernels carry the preceding rotated row in registers and amortize
  frequencies/parameters across sequence tiles;
- optimized stream specializations cover MHA and GQA ratios 2:1 and 4:1;
- IHA `P=2` uses four effective rows per tile, preserving pseudo-slot locality;
- batch `<=2`, sequence `<=128` uses two-row forward tiles;
- larger backward tiles amortize parameter partials, while D=128 and long
  sequences use smaller occupancy-preserving variants;
- small backward workloads fuse alpha/log-scale and both RMS-weight reductions
  into one epilogue launch; larger reductions retain the scalable multi-stage
  path.

No intermediate phase, cosine, sine, normalized Q/K, or momentum tensor is
materialized in global memory by the fused path. The source does not explicitly
allocate a shared-memory staging tile; Triton remains free to use shared memory
while lowering reductions. Nsight Compute 2025.3.1 was present on the validation
host, but hardware counters were disabled by the provider
(`ERR_NVGPUCTRPERM`), so bank-conflict, register, and spill counter claims are
intentionally omitted.

## API

```python
from cut_cross_entropy.repo_grape import repo_grape

q_out, k_out = repo_grape(
    q,                         # [B, Hq, S_effective, D]
    k,                         # [B, Hk, S_effective, D]
    z,                         # [B, Hrepo, S_base]
    position_ids,              # [B, S_base] or None
    inv_freq,                  # [rotary_dim / 2], FP32
    alpha,                     # [Hrepo], FP32
    log_scale,                 # [Hrepo, >= rotary_dim / 2], FP32
    attention_scaling,
    sequence_pseudo_factor=2,  # 1 or validated IHA P=2
    momentum_gamma=0.1,        # optional extension; use 0 for base behavior
    output_dtype=torch.bfloat16,
    q_norm_weight=q_weight,    # omit both weights to disable fused RMSNorm
    k_norm_weight=k_weight,
    rms_norm_eps=1e-6,
)
```

Validated inputs are CUDA BF16 or FP32 Q/K, BF16 or FP32 `z`, head dimensions
up to 256, non-contiguous positive-stride Q/K views, MHA, GQA 2:1/4:1, and IHA
`P=1/2`. `repo_grape_supported(...)` can be used before optional dispatch.

## RTX 5090 results

Environment: RTX 5090 (SM120), Python 3.13.12, PyTorch 2.13.0+cu130,
CUDA 13.0.2, and Triton 3.7.1. The reference is the same PyTorch formulation
compiled with `mode="max-autotune"` and `fullgraph=True`. Times below are the
sum of CUDA time inside compiled FX graphs after warmup, not Python wall time.

### Training forward + backward

| Shape / route | Reference (µs) | Fused (µs) | Speedup | Incremental peak |
|---|---:|---:|---:|---:|
| GQA 2:1, B1 S64 D64 | 20.08 | 7.95 | 2.53x | 96 KiB / 96 KiB |
| GQA 2:1, B8 S128 D64 | 30.83 | 14.38 | 2.14x | 1.5 MiB / 1.5 MiB |
| GQA 2:1, B8 S512 D64 | 53.27 | 20.24 | 2.63x | 6 MiB / 6 MiB |
| GQA 2:1, B8 S2048 D64 | 166.67 | 80.06 | 2.08x | 24 MiB / 24 MiB |
| GQA 2:1, B64 S512 D64 | 435.88 | 149.05 | 2.92x | 48 MiB / 48 MiB |
| MHA, B8 S512 D64 | 41.19 | 15.25 | 2.70x | 4 MiB / 4 MiB |
| GQA 4:1, B8 S512 D64 | 52.87 | 25.17 | 2.10x | 5 MiB / 5 MiB |
| GQA 2:1, B8 S512 D128 | 87.75 | 39.92 | 2.20x | 12 MiB / 12 MiB |
| FP32 input/output, B8 S512 D64 | 60.25 | 22.12 | 2.72x | 6 MiB / 6 MiB |
| no fused RMS, B8 S512 D64 | 37.90 | 12.38 | 3.06x | 6 MiB / 6 MiB |
| IHA P=2, B8 S512 base, D64 | 94.57 | 36.57 | 2.59x | 12 MiB / 12 MiB |
| no momentum, B8 S512 D64 | 55.40 | 21.48 | 2.58x | 6 MiB / 6 MiB |

Across this matrix the minimum CUDA speedup was 2.08x, the maximum 3.06x, and
the unweighted mean 2.52x. Incremental allocation was unchanged; the gain comes
from fewer launches, fewer global intermediates, and fused recomputation, not
from reserving additional VRAM.

### Inference

| Shape / route | Reference (µs) | Fused (µs) | Speedup |
|---|---:|---:|---:|
| GQA 2:1, B1 S64 D64 | 5.83 | 2.23 | 2.62x |
| GQA 2:1, B8 S512 D64 | 20.01 | 7.35 | 2.72x |
| MHA, B8 S512 D64 | 13.40 | 5.37 | 2.50x |
| GQA 4:1, B8 S512 D64 | 17.46 | 7.51 | 2.33x |
| IHA P=2, B8 S512 base, D64 | 37.29 | 12.67 | 2.94x |
| GQA 2:1, B8 S2048 D64 | 67.33 | 22.48 | 3.00x |
| no momentum, B8 S512 D64 | 19.89 | 9.35 | 2.13x |
| FP32 input/output, B8 S512 D64 | 22.40 | 7.65 | 2.93x |

The minimum inference CUDA speedup was 2.13x, the maximum 3.00x, and the
unweighted mean 2.65x. Steady-state inference reported zero incremental
allocation for both variants because the compiled graph pools already owned
their output buffers.

## Precision and stability evidence

- 23 focused CUDA tests cover BF16/FP32 forward and backward, full-graph
  compilation, MHA/GQA, IHA `P=2`, sequence lengths 1/3/5/63/65, reset and
  non-monotonic positions, zero momentum, circular-resultant fallback, and
  `torch.no_grad()` inference.
- Across magnitude probes (Q/K scaled by 256 and 2^-10, alpha in
  `[-1.5, 2.5]`, log-scale in `[-2, 2]`, and positions near 32,700 with a
  reset), all outputs and gradients were finite. Maximum relative L2 error
  against the compiled reference was 0.311%.
- A paired 500-step IHA `P=2` AdamW run had zero non-finite checks, final loss
  3.85985 reference vs 3.85997 fused, and FP32-master relative drift 0.0408%.
- The paired 5,000-step run had zero non-finite checks, final loss 2.37051 vs
  2.37049, maximum gradient norm 0.444697 vs 0.444695, and FP32-master relative
  drift 0.111%.
- The stability runner held both models and optimizers simultaneously, peaked
  at about 131 MiB, and was capped below 10 GiB. The cap belongs only to the
  benchmark process.

These synthetic trajectories validate the operator boundary and accumulated
gradient behavior; they are not a substitute for a complete model pretraining
ablation.

## Reproduction

Run correctness and compilation tests:

```bash
python -m pytest -q tests/test_repo_grape.py
```

Profile the normal GQA training shape:

```bash
python benchmark/repo_grape_profile.py \
  --batch 8 --query-heads 8 --key-heads 4 \
  --sequence 512 --head-dim 64 --rot-half 16 \
  --fuse-norm --profile-steps 30 --event-repeats 30 --compact
```

Profile inference or IHA `P=2` by adding `--forward-only` or
`--sequence-pseudo-factor 2`, respectively. Geometry probe arguments affect
only this benchmark process and are not production flags.

Reproduce the long stability comparison:

```bash
python benchmark/repo_grape_stability.py \
  --steps 5000 --pseudo-factor 2 --memory-limit-gib 10
```
