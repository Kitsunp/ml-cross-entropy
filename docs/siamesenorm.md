# Paired SiameseNorm kernel candidate

This is a general paired-normalization operator, not a JToK, Leviathan or
JToK-M kernel. It has no tokenizer, routing, attention, MLP or dataset dependency.
It replaces one normalization boundary of the existing
[SiameseNorm topology](https://arxiv.org/abs/2602.08064), not the topology itself.

## Definition

For each row of width D, define

```
r(x) = (sum_d x_d^2 / D + eps)^(-1/2)
a_d = w_x,d * x_d * r(x)
b_d = w_y,d * y_d * r(y)
z_d = s_d * a_d + b_d             # s=1 when omitted
o_d = w_out,d * z_d * r(z)        # o=z when outer norm is omitted
```

These expressions describe real arithmetic. The implementation preserves the
native Torch dtype boundaries: each affine RMSNorm returns in input dtype;
the separate scale multiply rounds before the subsequent add; the sum rounds
before the outer RMSNorm. FP32 reductions do not authorize removal of these
BF16 boundaries. In CuTe, explicit round-to-nearest conversion prevents LLVM
contraction from silently replacing a rounded multiply and add with one FMA.

For an affine RMSNorm `u=w*x*r`, with upstream gradient g, its row VJP is

```
q = g * w
dx = r*q - x*r^3*mean_d(q_d*x_d)
dw_d = sum_rows(g_d*x_d*r)
```

Apply this first to the outer norm, then to the two branches. The scale has
`ds_d=sum_rows(dz_d*a_d)`, and the X branch receives `da=dz*s`.
The dtype-specific gradient casts of the separate scale multiplication are
also preserved before broadcast reduction. There are no parameter atomics:
row-group partials are accumulated in FP32 and reduced in a fixed hierarchy.
Before affine weights, the RMSNorm Jacobian has radial eigenvalue `eps*r^3`
and orthogonal eigenvalue `r`. With positive epsilon it is not an exact
tangent projection. A double-precision regression checks the full paired VJP,
including optional scalar output scaling, against native autograd.

## Status and use

The default remains the original Torch reference until the real-model gate
passes. The explicit candidates are:

- `backend="triton"`: Triton forward and backward.
- `backend="cute"`: independent CuTe forward, backward and parameter reductions.
  No Triton kernel is called inside this route.
- `backend="auto"`: CuTe when installed and eligible, otherwise Triton on
  eligible CUDA inputs, otherwise Torch. Use explicit candidates for benchmarks.

The upstream integration keeps its static `SIAMESENORM_KERNEL_BACKEND`
selector at `"reference"`, letting Inductor compile the original Torch
expression. Selecting `"triton"` or `"cute"` before model construction is
explicit opt-in; installing this package does not switch the model backend.
The private upstream source is not distributed in this repository.

```python
from cut_cross_entropy.siamesenorm import siamesenorm

h = siamesenorm(x, y, weight_x, weight_y, weight_out, scale,
                eps=1e-6, backend="triton")
```

Current custom paths require contiguous CUDA FP32/BF16 streams and parameter
vectors on one device, positive explicit epsilon and width at most 4096.
Streams and affine parameters may have different supported dtypes: FP32
residuals with BF16 weights do not require hidden casts. Each branch RMSNorm
returns its stream dtype; scale multiplication and addition use Torch type
promotion. The outer RMSNorm returns the promoted sum dtype. Backward casts
each branch gradient to its native dtype before its VJP, and parameter gradient
outputs retain their corresponding parameter dtype. CuTe also
requires width divisible by 32 and at most 1024 to bound shared memory in its
backward. Offset contiguous views need no hidden aligned
clone. Forced unsupported routes raise; auto uses the independent reference.
Autograd currently covers first derivatives, not higher-order derivatives.

The opaque path has fake tensor and autograd registrations and uses the
current device/stream. External JIT entries must be prewarmed before CUDA Graph
capture. No per-step backend, environment-variable or Python phase changes are
part of the operator. Residual updates stay with the caller; fusing across
attention/MLP would change dependency and tensor lifetime contracts.

### Optional consumer epilogue (candidate)

`post_scale` is an optional **non-trainable scalar tensor** on the native norm
device and dtype. `out_dtype` optionally selects FP32 or BF16 after that scale.
Their defaults preserve the original API. Invalid shape, trainable post-scale
or implicit mixed-dtype scalar promotion is rejected rather than detached.
The branch scale remains an independent trainable hidden-width vector.

Let `Q_t` mean rounding to dtype t, n the native mixed/norm dtype and c the
consumer dtype. The contract, including rounding boundaries, is

```text
h = Q_n(RMS_out(z))             # or z without outer norm
v = Q_n(h * post_scale)         # or h without post_scale
o = Q_c(v)
g_v = Q_n(g_o)
g_h = Q_n(g_v * post_scale)     # or g_v without post_scale
```

Apply the existing norm VJP to `g_h`. Neither the native mixed value nor its
VJP may use c in place of n. A FP32 consumer does not restore precision lost
at earlier BF16 boundaries; correctness tolerances must account for that
path, not only the final tensor dtype. Tests separately compare fused versus
unfused operations within the same backend, avoiding concealment of fusion
errors by the native-versus-backend reduction tolerance.

The scalar's **value** is tensor data, not a Python compilation argument.
Its shape/dtype, presence and consumer dtype remain fixed after prewarm.
This supports an existing fixed depth factor without learning a new scale or
changing optimizer parameters. Both phases of each backend implement this
epilogue independently; there is no Torch pointwise kernel inside its VJP.

Fuse the cast only where an identical consumer cast already exists. An
autocast Linear can qualify; a router, positional module, activation tap or
memory consumer of the original activation may not. Do not cast residual
streams, skip regularizers or move a shared router state to buy bandwidth.
Returning both native and consumer outputs would need a separate multi-output
VJP that sums both incoming gradients; it is not implemented here.

For a solely consumed FP32 norm result followed by a scalar scale and BF16 cast,
fusion removes a 4-byte write and 4-byte read per element (`8*R*D` bytes before
caches). It also avoids materializing the corresponding full-width native
upstream-gradient intermediate. These are traffic estimates, not measured
DRAM bytes or an accepted training-speed improvement. Real 100+100 validation
and stable compiler counters remain required.

## Acceptance and profiling

Correctness checks compare outputs and all parameter/input gradients with native
Torch, including optional components, zeros, small amplitudes, odd widths and
offset views. The benchmark records actual execution traces, not source hashes
as a substitute for profiling:

```bash
python -m benchmark.siamesenorm_profile --backend triton --output PRIVATE_RESULTS
python -m benchmark.siamesenorm_profile --backend cute --output PRIVATE_RESULTS
python -m benchmark.siamesenorm_profile --backend reference --output PRIVATE_RESULTS
```

Operator numbers alone cannot establish a model-step improvement. The real
training comparison uses 100 optimizer steps and 100 validation batches,
identical data and policy, with table/no-Siamese, Leviathan/no-Siamese,
Leviathan/Torch-Siamese, Leviathan/Triton-Siamese and Leviathan/CuTe-Siamese
conditions. Compare each custom backend against Leviathan/Torch-Siamese to
isolate replacement of the kernel. Cold compilation, profiling steps,
stable compute, allocated memory and reserved CUDA Graph pools are separated.
No end-to-end speedup is claimed yet.

The consumer-epilogue experiment produced the following **observed**, aggregate
full-training-step medians in one test cohort. Each candidate completed 100
optimizer steps and 100 validation batches; the existing control was reused.
No private model source, configuration, provenance or other-component timings
are included here:

| Normalization route | Full-step median | Steps/s |
| --- | ---: | ---: |
| Compiled Torch control | 216.13 ms | 4.627 |
| Triton with consumer epilogue | 213.72 ms | 4.679 |
| CuTe with consumer epilogue / identity-cast cleanup | 214.00 ms | 4.673 |
| CuTe with consumer epilogue / unified row guard | 214.12 ms | 4.670 |

Both candidates kept post-warmup compiler/graph snapshots unchanged and had
zero guarded upstream reference fallbacks. Relative to their own unfused
custom boundaries, aggregate validation allocated peaks fell about 96 MiB;
stable training allocated peaks did not fall. Reserved pools are not a proxy
for required working set. Conditional within-run timing intervals include
zero, so these roughly 1% observed differences do **not** establish a robust
win or change the default. Removing redundant identity casts did not resolve
the CuTe BF16 row-backward regression. A subsequent unified-row-guard change
recovered most of that specialization's observed time: 235.18 to 92.75 us,
versus 88.72 us in the earlier compact variant (width 512; two profiled calls
per variant). Full-step performance stayed effectively unchanged. These
profiles support a local correction, not a robust whole-training speed win.
Full private traces and failure/correction records are retained separately.

### Engineering checks before attributing a slowdown

A passing numerical test and zero Dynamo graph breaks are not proof that the
rest of the model kept its fast path. The real runner forbids actual upstream
Leviathan reference forward/backward fallbacks and validates integer tile
overrides before compilation. In reports, `auto` means an unset tile override,
not the literal environment value `auto`. Retain misconfigured measurements
as failures of the experiment; exclude them from kernel-speed comparisons.

CUDA Graph tree counters need separate interpretations. The runner records
`cuda_graph_new_recording_events` from the tree manager's `debug_fail_counter`:
despite that internal name, it counts replay misses that lead to new recordings,
not capture failures. Compare its post-warmup delta alongside tree node count
and unexpected rerecord count. Historical JSON used the misleading key
`cuda_graph_recording_failures`; retain the raw records but apply this corrected
interpretation, rather than claiming a nonzero value proves a capture failure.

Compare identical graph boundaries. Inductor may fuse paired normalization
with residual updates, output casts and adjacent elementwise work. A custom
boundary can introduce saved tensors and extra full-width writes/reads. For
R rows of width D, writing and reading one extra FP32 intermediate costs
`8*R*D` bytes before caches. This estimate is not a measured DRAM count or a
latency prediction. Kernel names containing RMSNorm can include other work;
their entire duration must not be reported as normalization-only latency.

The current [PyTorch Triton integration guide](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
distinguishes an opaque `custom_op` from a compiler-visible `triton_op` with
`wrap_triton` launches. Compiler visibility is an integration option to test,
not proof of automatic fusion or a speedup. The existing independent backend
and native-dtype numerical contracts still apply.

On Torch versions exposing this API, the Triton path now registers both phases
as `triton_op` and wraps all four launch sites with `wrap_triton`. CuTe retains
its independent opaque boundary. Older Torch retains the compatibility path.
The compiled regression forbids use of the opaque Triton compatibility route;
passing it does not establish an end-to-end speedup.

For CuTe, [NVIDIA's framework integration guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/framework_integration.html)
and [TVM FFI guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/tvm_ffi_compilation.html)
describe direct Torch tensor invocation and stream/device handling. The
candidate already uses TVM FFI; switching the invocation mechanism alone is
not evidence that register pressure or memory traffic improved. Inspect
emitted register/shared-memory resources and hardware spill/traffic counters
before attributing a bottleneck. A high register count does not prove spills.

CuTe compacts affine-gradient scratch space to `P = 2 + outer + scale` planes,
with absent optional parameters removed rather than initialized and ignored.
For `G = ceil(rows/8)`, the row partials occupy `4*P*G*width` bytes and the
row kernel's shared accumulator occupies `4*4*P*width` bytes. For width 512
with an outer norm but no scale, this reduces shared memory from 32 to 24 KiB;
at 32768 rows, partial storage falls from 32 to 24 MiB. The two-stage reduction
and each active plane's sum order stay unchanged. These resource savings are
not themselves a claim of reduced full-step latency.

### Work layout, not just a register count

The regular one-warp CuTe row VJP encloses its live vectors and reductions in
one row-valid region. Each warp owns one row, so that predicate is uniform
within the warp. Parameter scratch is zeroed before the region; the final CTA
barrier and shared reduction remain outside it. Absent final rows therefore
contribute zero without carrying live zero-filled vectors through three
separately masked regions. The cooperative experiment must not use a
row-dependent guard around its CTA barriers: its enclosing condition is
statically true, keeping every row owner at the barriers.

In the BF16 width-512 compile-only comparison, consolidating the guard reduced
PTX instruction lines from 2397 to 1964 and branches from five to three.
Registers increased from 197 to 251; both variants had zero reported static
spills/local storage and the same potential occupancy. PTX counts are static
code, not dynamic executed instruction counts. The observed recovery despite
more registers motivates inspecting control flow and live ranges, not tuning
the register count in isolation. [CuTe's control-flow documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.rst)
distinguishes emitted runtime regions from compile-time specialization.

The experimental CuTe cooperative backward uses two warps per row for widths
at least 512 and divisible by 64; other supported widths retain one warp.
The regular CuTe planner keeps one warp: the cooperative variant improved
resource-limited occupancy but not measured full-step latency, and its row
kernel regressed. Tests explicitly select the experimental layout before
compilation; it is not a supported per-step runtime toggle.
Four row owners per block each process
two rows, so `ceil(rows/8)` parameter-partial groups stay unchanged. Each lane
owns half as many stream/VJP elements on the two-warp path. Values are retained
rather than reloaded or repeatedly recomputed from global memory.
Forward retains its separate one-warp-per-row layout.

Three separate scratch planes combine the partner-warp row dots, adding 96
bytes of shared memory. CTA barriers are outside the valid-row predicate, even
for an incomplete final block. Without the outer norm, an explicit final
barrier protects reuse of the branch-dot scratch in the next row iteration.
With the outer norm, the next outer-dot barrier provides that protection.
Within-row FP32 sum trees change; branch rounding and the VJP formula do not.
The width-544 regression ensures a legal width not divisible by 64 is not
silently truncated. Widths 32/1024 and partial row groups are also tested.

The Triton 16-warp width-512 experiment also did not improve the full step;
the regular planner retains eight warps. The experiment kept eight rows per
partial group. Its compiler owns the lane layout; it does not call the CuTe
implementation. Inspect aligned and offset pointer
specializations: alignment can change Triton's vectorization, register use
and cross-warp reduction layout, even with the same shape and math.

[CUDA's occupancy API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__OCCUPANCY.html)
can check the maximum resident blocks for a compiled function, block size and
actual dynamic shared-memory size without launching the kernel. Compare
`blocks * warps_per_block`, not just blocks or registers per thread. This is
resource-limited potential occupancy, not measured achieved occupancy.
More resident warps may help latency hiding, but additional barriers or an
already-saturated memory path can erase that benefit. Full-step profiling
remains the acceptance gate.

[NVIDIA's debugging guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/debugging.html)
documents `remarks{ptx}` and retained PTX/CUBIN for compile-time spill/resource
inspection. A register cap can induce spills; do not apply one just to lower
the reported register count. Compile-time spill diagnostics and runtime
hardware traffic measurements are different kinds of evidence.

Private source, checkpoint, full-model traces and private provenance are not
public benchmark artifacts. Kernel-only traces may be exported separately.
