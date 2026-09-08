# Leviathan-JTok / JTok-M integration

This change keeps the existing Leviathan custom operation intact and adds an
explicit JTok boundary for the continuous token modulation used by NeoLLM.
It does not change the training compiler policy, tokenizer, configuration
defaults, or the legacy Leviathan path.

## Dispatch contract

`backend="torch"` is the reference implementation. `backend="triton"` is a
strict opt-in and requires the compatible CCE JTok adapter plus a CUDA shape
supported by the Triton surface. Unsupported geometries raise an error instead
of silently falling back to a different implementation. `backend="auto"` is
reserved for library callers that explicitly request capability dispatch.

On Torch 2.14, the JTok operations use `torch.library.triton_op` and
`torch.library.wrap_triton` when available. This exposes the Triton launch as
an explicit operator boundary to `torch.compile`, while leaving the router and
surrounding model operations available for Inductor. Older Torch versions keep
the compatibility custom-op registration.

## Leviathan connection and gradients

The legacy `leviathan_embedding_compiler_safe` entry point remains available
for callers that only need the embedding. When JTok/JTok-M is active,
`leviathan_embedding_with_seed_compiler_safe` uses a multi-output Leviathan
operation instead:

```text
ids -> codebook gather -> z -> Leviathan embedding
                         \\-> JTok/JTok-M
```

For supported CUDA geometries, the gather is fused into Leviathan's stage-A
kernel (`FUSE_GATHER=True`). The kernel writes `z` once and returns that exact
tensor to JTok/JTok-M; the old CUDA path that called
`_leviathan_seed_from_codebooks` a second time is not used. In backward, the
Leviathan gradient and the JTok/JTok-M seed gradient are added before the one
codebook scatter. The CPU/reference path retains the differentiable bridge for
model-independent tests and non-CUDA callers.

`LEV_FUSE_GATHER=0` disables the automatic fused-gather policy for the ordinary
Leviathan path. The JTok/JTok-M multi-output operation requests the safe fused
path explicitly because it must expose the shared seed. Base Leviathan training
still saves `z` when its backward needs it; base inference keeps `SAVE_Z=False`
when no consumer requests the seed.

The integration tests compare the returned embedding and seed with
`leviathan_forward_ref`, verify that the CUDA path does not call the duplicate
seed bridge, and check finite codebook gradients through both output routes.

## NeoLLM activation

The companion NeoLLM training script keeps the default off for checkpoint
compatibility and selects the mode without editing `config.json`:

```bash
# JTok, one continuous surface per decoder layer
NEOLLM_JTOK_MODE=jtok \
NEOLLM_JTOK_KERNEL_BACKEND=triton \
python train.py

# JTok-M, routed continuous surfaces with Top-K experts
NEOLLM_JTOK_MODE=jtokm \
NEOLLM_JTOK_KERNEL_BACKEND=triton \
python train.py
```

Use `NEOLLM_JTOK_KERNEL_BACKEND=torch` for the reference comparison. The
model configuration still requires `use_token_generator=True`; JTok-M also
requires `use_jtok=True`. The existing `jtok_num_modes`, `jtok_num_experts`,
`jtok_top_k`, and `jtok_aux_loss_weight` fields control the geometry and
load-balancing objective.

## Verification

On the CUDA environment used for validation (Torch 2.14.0+cu132, CUDA 13.2,
RTX 5090 / SM120), the focused suite passed:

```text
32 passed, 4 warnings
```

The warnings were Triton `AnnAssign` deprecations. A full-suite run was
started but intentionally stopped at approximately 6% when the experiment was
closed; it is therefore not reported as a full-suite pass.

The actual two-layer compiled training step used BF16, batch 1, sequence 64,
hidden 256, and an optimizer update. `torch.compile(mode="max-autotune")`
was used for both variants:

| JTok surface | Full step median | Incremental CUDA allocation |
| --- | ---: | ---: |
| Torch reference | 5.219 ms | 33,963,520 bytes (32.39 MiB) |
| Triton | 6.281 ms | 33,779,200 bytes (32.21 MiB) |

The current result is a memory tie (about 0.6% less incremental allocation for
Triton) but a slower Triton backward in this small geometry. The implementation
is therefore an integration and correctness slice, not a claim that the
current Triton surface has already met the final speed target.

## Reproducible integration probes

The repository now contains two deliberately separate runners:

* `benchmark/leviathan_jtok_integration.py` is model-free. It constructs the
  repository's own `LeviathanGenerator`, calls the differentiable seed bridge,
  applies several JTok/JTok-M layers, runs backward, and can execute AdamW.
  It never imports `modeling_neollm.py`, a tokenizer, or a checkpoint. This is
  the required test for kernel correctness and for odd geometries when the
  downstream model source is unavailable.
* `benchmark/neo_llm_jtok.py` is optional and must receive explicit model and
  configuration source paths. It is the integration probe for the complete
  NeoLLM graph and the real CCE loss; its source files are not a dependency of
  the repository tests.

Both runners use seed `1729`, report CUDA memory, compile counters, and finite
loss/parameter checks. `--inductor-cache-dir` records an isolated cache path;
use a new path for a cold-cache reproduction. Reusing a generated Inductor
graph can hide a CUDA-Graph capture failure.

The model-free smoke matrix used on the RTX 5090/SM120 remote environment
(Torch `2.14.0+cu132`, CUDA `13.2`, Triton `3.8.0`) included:

| case | result |
| --- | --- |
| JTok, batch 2 × 37, hidden 256, 5 modes/9 knots, 73% valid mask, eager Triton | finite forward/backward, AdamW optional path passed |
| JTok-M, batch 4 × 37, hidden 256, 5 experts/Top-2, 61% valid mask, eager Triton | finite forward/backward and AdamW passed |
| JTok, 4 layers, `fullgraph=True`, Triton | passed with one graph |
| JTok, 4 layers, `fullgraph=False`, `triton.cudagraphs=False` per call | passed without a capture error and with finite gradients |
| JTok-M, 4 layers, `fullgraph=True`, 5 experts/Top-2 | passed with one graph and AdamW |

### Hidden-size dispatch boundary

The Triton dispatcher keeps the single-tile kernel for `hidden < 256` and
routes the boundary (`hidden == 256`) through the compact wide path. This is
not a model-specific geometry override: both forward and backward share the
named `_SINGLE_TILE_HIDDEN_LIMIT` so the dispatch cannot diverge. The reason
is measured kernel work: the old single-tile backward recalculated the
B-spline/mode derivative per hidden lane, while the wide path caches the mode
products per token and route. Larger rows were already using that path.

On the replacement RTX 5090 environment, with seed `1729`, BF16, AdamW, four
layers, `batch=4`, `sequence=37`, five experts/Top-2, and fresh
`torch.compile(mode="max-autotune")` caches:

| JTok-M route at hidden 256 | step median | stable allocated peak |
| --- | ---: | ---: |
| old single-tile Triton | 4.650 ms | ~36.5 MiB |
| compact wide Triton | 2.169 ms (10 stable steps) | ~36.5 MiB |
| Torch reference oracle | 1.806 ms | ~36.8 MiB |

The same change measured 6.803 ms eager for Triton versus 7.698 ms before it.
The external kernel remains about 20% above the compiled Torch oracle in this
case, so this result is a measured improvement and not a claim of final
parity. The boundary has a dedicated forward/backward numerical test,
including invalid rows; the remote focused suite passed 16 tests.

The same case was repeated with five warmup steps and 30 measured steps to
avoid mixing short samples with the comparison above. The compact Triton path
measured `2.1518 ms` median (`2.1528 ms` mean) with `38,244,864` bytes of
stable allocated peak. The clean checkout measured `4.5889 ms`, and the
compiled Torch oracle measured `1.7272 ms` with `38,631,936` bytes. Thus the
boundary change reduced the Triton median by 53.1% versus the old external
path and used 384 KiB less stable allocation than the oracle, but remained
24.6% slower than Torch in this reproducible 30-step sample. The +/-5% target
is therefore still open; these numbers are stored under
`remote_runs/2026-09-07-jtok-wheel/`.

Commands for those model-free checks are reproducible without NeoLLM:

```bash
python benchmark/leviathan_jtok_integration.py \
  --compiled --fullgraph --variant jtokm --backend triton \
  --batch 2 --sequence 37 --hidden 256 --layers 4 \
  --d-seed 32 --generator-knots 8 --generator-rank 16 \
  --modes 5 --knots 9 --experts 5 --top-k 2 \
  --valid-ratio 0.61 --seed 1729 --expected-status pass
```

### Complete NeoLLM result and current blocker

With `/modeling_neollm.py` and `/configuration_neollm.py` supplied explicitly,
the same remote environment was tested with batch 2, sequence 128, hidden 512,
12 layers, BF16, the Triton Leviathan path, JTok enabled, AdamW, and the real
CCE training loss. A fresh `TORCHINDUCTOR_CACHE_DIR` made the result stable:

| complete graph case | result |
| --- | --- |
| Legacy baseline, JTok disabled, default partitioned compile | passed; cold ~77.1 s, stable ~105.6 ms |
| JTok + Triton, default partitioned compile, cold cache | `cudaStreamCaptureInvalidated` during `_jtok_project_fused_kernel_0` capture |
| JTok + Torch reference surface, same compile/cold-cache setup | the same CUDA-Graph capture error |
| JTok + Triton, `fullgraph=True`, real CCE loss | the same capture error |
| JTok + Triton, `--disable-cudagraphs` benchmark isolation | passed; cold ~80.95 s, stable ~15.25 ms, peak ~594 MiB |

This means the isolated Triton kernel, the seed bridge, and the JTok backward
are functioning; the remaining defect is the composition of the JTok-enabled
NeoLLM/CCE graph with CUDA Graph Trees. It is not evidence that the production
training command should disable CUDA Graphs. `--disable-cudagraphs` is only an
isolation control in the benchmark, while the real training code still uses
its normal `torch.compile(max-autotune)` policy and
`cudagraph_mark_step_begin()` boundary.

### Complete-model inference after benchmark output fix

The benchmark now clones scalar outputs immediately outside the compiled
boundary. This is required because retaining a detached view of a CUDA-Graph
output across the next replay can trigger Torch 2.14's intentional
"output ... overwritten" diagnostic; that was a bug in the measurement harness,
not in JTok. With a fresh Inductor cache and no memory guard, the corrected
runner produced these complete-model results on the same RTX 5090:

| variant/backend | mode | batch x sequence | layers | valid ratio | cold compile+run | stable forward | peak allocated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JTok / Torch | inference | 2 x 128 | 12 | 1.00 | 39.87 s | 1.54 ms | 113.8 MiB |
| JTok / Triton | inference | 2 x 128 | 12 | 1.00 | 36.55 s | 4.13 ms | 113.8 MiB |
| JTok-M / Triton | inference | 2 x 128 | 12 | 0.61 | 36.77 s | 6.96 ms | 118.6 MiB |

All three runs used seed `1729`, Torch `2.14.0+cu132`, CUDA `13.2`, Triton
`3.8.0`, `max-autotune`, one compiled graph, finite outputs, and the real
Leviathan Triton producer. The JTok-M case exercised five experts with Top-2
routing and a partial validity mask. These are inference/pre-fill results;
they do not repair the independent training+CCE capture failure above, and
the Triton surface is not yet at the required +/-5% speed parity with the
Torch oracle in this configuration.

### Full training profile: vectorized mode evaluation (2026-09-08)

The next optimization was restricted to the wide JTok/JTok-M path. The old
mode evaluator launched one Triton program per `(token, route, mode)` and
rebuilt the same normalized B-spline basis for every mode. The new guarded
path launches one program per `(token, route)`, evaluates a padded mode vector,
and reuses the basis across modes. It is enabled only when
`d_seed <= 128`, `mode_pad <= 32`, and
`d_seed * knot_pad * mode_pad <= 8192`; larger or unusual surfaces retain the
previous evaluator.

The valid full-flow runs used the supplied NeoLLM source, seed `1729`, BF16,
batch 64, sequence 512, 12 Transformer layers, the real CCE loss,
`torch.compile(mode="max-autotune")`, AdEMAMix, and MXFP8 inactive. The
external package provenance was checked before execution:

```text
/workspace/codex_ml_cross_entropy_jtok_wide256_exp/cut_cross_entropy/__init__.py
/workspace/codex_ml_cross_entropy_jtok_wide256_exp/cut_cross_entropy/leviathan/jtok.py
```

The first run made without `PYTHONPATH` was discarded for this comparison,
because the editable install pointed at the older clone under
`/root/src/cut-cross-entropy`. This is intentionally recorded so an installed
package cannot be mistaken for the checkout under test.

Stable step medians were taken from steps 4--11 of the same profiled process;
the first compiled step and the profiler step were excluded:

| full training flow | stable step | steps/s | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| Leviathan base | 223.31 ms | 4.478 | 19.53 GiB | 20.86 GiB |
| JTok before mode fusion | 380.21 ms | 2.630 | 19.95 GiB | 21.75 GiB |
| JTok with vectorized modes | 321.94 ms | 3.106 | 19.95 GiB | 21.75 GiB |
| JTok-M before mode fusion | 560.72 ms | 1.783 | 20.43 GiB | 22.27 GiB |
| JTok-M with vectorized modes | 441.51 ms | 2.265 | 20.43 GiB | 22.27 GiB |

Relative to the immediately preceding implementation, the complete JTok step
improved by 15.32% and JTok-M by 21.26%, without an observed VRAM increase.
Relative to the original full-flow measurements, the cumulative reductions are
31.38% for JTok and 38.97% for JTok-M. The base Leviathan flow is still faster:
the remaining time gap is 98.63 ms for JTok and 218.20 ms for JTok-M per step.

The clean traces confirmed the intended external route. JTok used
`_jtok_modes_kernel_vectorized` 24 times for the profiled window, plus the
existing Leviathan and CCE kernels; JTok-M used the same vectorized mode kernel
and its routed auxiliary-loss path. No `jtok_reference` event was present.
For JTok, the mode kernel fell from 81.84 ms to 25.26 ms in the trace. For
JTok-M it fell from 164.89 ms to 50.55 ms. The remaining largest JTok-specific
kernel is `_jtok_backward_multi_tile_fast_kernel` at 36.98 ms (72.83 ms for
JTok-M), so that is the next optimization target. No change to that kernel has
been claimed yet.

The profile harness synchronizes before and after each measured step, so its
`cudaStreamSynchronize` duration is not attributed to JTok without a separate
launch-boundary experiment. Full-flow JSON results and traces are retained
outside the source distribution rather than committed with model data.

### Rejected experiment: vectorized backward mode contraction (2026-09-08)

The next candidate added a padded `modes x hidden-tile` load to
`_jtok_backward_multi_tile_fast_kernel`. It was guarded by a geometry budget and
kept the previous scalar loop for larger surfaces. Focused CUDA tests passed,
but the full-flow profiles did not justify keeping it:

| variant | previous stable step (4--11) | candidate stable step (4--11) | multi-tile kernel in trace |
| --- | ---: | ---: | ---: |
| JTok | 321.94 ms | 321.77 ms | 36.98 -> 36.70 ms |
| JTok-M | 441.51 ms | 448.60 ms | 72.83 -> 81.13 ms |

The real geometry uses four modes, so the scalar loop was already small. The
vectorized version kept a larger output-weight tile live, increased register
pressure, and did not remove launches, atomics, or hidden-tile reads. The
candidate was removed before commit. The mode-evaluation fusion remains; the
shared Leviathan seed boundary is now implemented, so JTok and JTok-M do not
repeat the codebook gather already performed by Leviathan.

### Shared seed / fused gather full-flow result (2026-09-08)

The shared-seed change was measured in the complete NeoLLM training flow with
seed `1729`, BF16, batch `64`, sequence `512`, 12 Transformer layers, the real
CCE loss, AdEMAMix, `torch.compile(mode="max-autotune")`, and MXFP8 inactive.
Each row is one profiled process; stable medians use steps 4--11 and exclude
the cold compiled step and profiler step.

| full flow | stable step | steps/s | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| Leviathan base, before | 223.311 ms | 4.478 | 19.53 GiB | 20.86 GiB |
| Leviathan base, fused gather | 223.479 ms | 4.475 | 19.54 GiB | 20.86 GiB |
| JTok, before shared seed | 321.939 ms | 3.106 | 19.95 GiB | 21.75 GiB |
| JTok, shared seed | 322.924 ms | 3.097 | 19.94 GiB | 21.22 GiB |
| JTok-M, before shared seed | 441.515 ms | 2.265 | 20.43 GiB | 22.27 GiB |
| JTok-M, shared seed | 442.074 ms | 2.262 | 20.42 GiB | 21.76 GiB |

The result is not a claim of a complete-step speedup: the gather is small
relative to the Transformer, CCE, backward, and optimizer. The change keeps
base within measurement noise, while reducing the allocator's peak reservation
by about `0.52--0.53 GiB` for both JTok variants. Live allocated memory changes
only slightly because the large model and optimizer buffers remain. The
profile traces contain `cut_cross_entropy::leviathan_forward_with_seed`,
`_lev_fused_dot`, the JTok kernels, and no `jtok_reference` event.

### Geometry-aware JTok autotune (2026-09-08)

The hidden-tile planner remains a correctness decision: a complete row is
processed by one tile only when its padded work fits the safe budget. For the
full NeoLLM geometry (`hidden=512`, `d_seed=128`, `modes=4`, `top_k=2`) the
backward uses two 256-lane multi-tile programs per token. Triton autotune does
not select between the single-tile and multi-tile algorithms.

Autotune now selects only launch parameters for the chosen geometry:

- the wide backward benchmarks a small set of warp/stage configurations;
- the projection-gradient reduction benchmarks `BLOCK_M` and launch
  resources independently from hidden-tile ownership;
- the cache key includes the effective tensor geometry, mask state, dtypes,
  and tile width;
- the reusable `surface` activation is restored between candidates because
  the kernel consumes it as input and writes its gradient to the same buffer.

The first occurrence of a geometry pays the candidate benchmark; later calls
reuse Triton's per-process result. This does not change the Torch compile
policy and does not add a Torch fallback.

The corrected full-flow JTok-M run used seed `1729`, BF16, batch `64`,
sequence `512`, 12 Transformer layers, real CCE/AdEMAMix and auxiliary losses,
MXFP8 inactive, and one profile inside a 12-step run. Recomputing stable
steps 4--11 from the recorded steps gives a median of `373.271 ms`
(`2.679 steps/s`) and mean `372.201 ms`; peak allocation was `20.418 GiB`
and peak reservation `21.756 GiB`. Two validation steps completed.

The trace totals were approximately `243.64 ms` for
`cut_cross_entropy::leviathan_backward`, `43.28 ms` for
`_jtok_backward_multi_tile_fast_kernel`, `28.07 ms` for
`_jtok_backward_projection_grad_kernel`, and `0.69 ms` for the global
normalization-dot kernel. The earlier route-cached full-flow profile is kept
as historical diagnostic data, not as a causal A/B baseline, because it
predates the global hidden-axis normalization-dot correction.

### Rejected experiment: block reduction of the shared scaler gradient (2026-09-08)

An isolated candidate added a Triton kernel that reduced the global
`grad_scaler` contribution in token blocks before the atomic update in the
multi-tile backward. The main multi-tile kernel skipped that contribution and
received a temporary hidden-sized FP32 workspace, so the numerical equation
and the legacy/single-tile routes were unchanged.

The candidate passed the focused Triton and multi-tile tests (16 passed) and
completed the full JTok-M training/evaluation flow with seed `1729`. It was
not kept because the complete-step result was indistinguishable from the
previous implementation:

| JTok-M full flow | previous | candidate | change |
| --- | ---: | ---: | ---: |
| stable median, steps 4--11 | 373.271 ms | 372.998 ms | -0.073% |
| stable mean, steps 4--11 | 372.201 ms | 372.773 ms | +0.154% |
| steps/s from median | 2.6790 | 2.6810 | +0.073% |
| peak allocated | 20.418 GiB | 20.418 GiB | unchanged |
| peak reserved | 21.756 GiB | 21.756 GiB | unchanged |

The trace showed the added kernel taking about `0.65 ms` in twelve calls. It
reduced the measured multi-tile kernel total from `43.28 ms` to `42.76 ms`,
but the enclosing Leviathan backward increased from `243.64 ms` to
`245.26 ms`; no complete-step speedup was established. The candidate was
removed before commit. The next optimization must target the larger
token-owned compact reductions in the multi-tile and projection-gradient
paths, not add another small standalone launch.

### Rejected experiment: route-major projection-gradient reduction (2026-09-08)

The next candidate changed only the wide JTok-M projection-gradient ownership.
The existing expert-major kernel scans each token block once per expert and
uses a route mask. The candidate instead launched one program per token block
and selected Top-K slot, then used scattered two-dimensional atomics to write
the selected expert's grad_spline_out and grad_residual_out. The geometry
guard was restricted to sparse routing (top_k * 2 <= experts), so the real
five-expert/Top-2 case selected it while plain JTok did not.

The candidate passed the focused model-free numerical tests (10 passed),
including the full hidden=512, d_seed=128, modes=4, top_k=2 backward and the
no-reference-fallback check. It was nevertheless rejected by the complete
NeoLLM training profile. Both runs used seed 1729, BF16, batch 64, sequence
512, 12 layers, the real CCE/AdEMAMix flow, MXFP8 inactive, and the same Torch
2.14 max-autotune training policy:

| JTok-M full flow | expert-major baseline | route-major candidate |
| --- | ---: | ---: |
| stable median, steps 4--11 | 373.271 ms | 452.615 ms |
| steps/s from median | 2.6790 | 2.2094 |
| peak allocated | 20.418 GiB | 20.418 GiB |
| peak reserved | 21.756 GiB | 21.756 GiB |

The trace explains the regression: the projection-gradient kernel increased
from 28.071 ms in 12 calls to 122.434 ms. The scattered atomics were much
more expensive than the expert-major scan, despite doing fewer logical route
checks. This candidate was removed from source and remote execution; the
expert-major path remains authoritative. The result rules out a naive
route-major atomic design, including for unbalanced routing. A future
reduction must aggregate by expert in a coalesced way before updating the
parameter tensors, without introducing a dense expert-expanded workspace.

### Accepted experiment: forward mode-cache reuse in wide backward (2026-09-08)

The next change targets a different source of repeated work. Before this
change, the wide JTok backward evaluated the selected B-spline mode product a
second time, even though the forward had already produced the same compact
`[tokens, top_k, modes]` activation. The new registered forward operation has
two outputs: the normal JTok/JTok-M result and that compact mode buffer. The
autograd context retains the buffer and the wide backward consumes it directly
for the projection and token-local gradient reductions. No dense
`[tokens, experts, modes, hidden]` activation is introduced, and the global
scaler-gradient path is unchanged.

The public wrappers still return exactly the previous public values: JTok
returns one tensor and JTok-M returns `(output, stats)`. The second registered
output is internal to the external-kernel/autograd boundary. `grad_modes` is
ignored intentionally because the mode buffer is an implementation cache, not
a differentiable model output.

The change was first checked without importing NeoLLM: six CUDA tests passed,
including odd/single-tile geometry, the full `hidden=512, d_seed=128,
modes=4` geometry, custom-op `opcheck`, compilation, and the no-reference
fallback assertion. It was then measured in the complete training flow with
seed `1729`, BF16, batch `64`, sequence `512`, 12 Transformer layers, CCE,
AdEMAMix, `torch.compile(mode="max-autotune")`, MXFP8 inactive, two validation
steps, and a profile in step 12.

| full flow | previous stable median (4--11) | mode-cache stable median (4--11) | change | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| JTok | 307.876 ms | 293.657 ms | -4.62% | 19.943 -> 19.946 GiB | 21.219 -> 21.221 GiB |
| JTok-M | 373.271 ms | 358.556 ms | -3.95% | 20.418 -> 20.424 GiB | 21.756 -> 21.760 GiB |

The corresponding stable throughput changed from `3.248` to `3.405`
steps/s for JTok and from `2.679` to `2.789` steps/s for JTok-M. The
additional retained compact buffer is approximately 3 MiB over the 12-layer
JTok flow and 6 MiB over the JTok-M flow for this geometry.

The traces provide the causal check: JTok's
`_jtok_modes_kernel_vectorized` fell from 24 calls / about 25.30 ms to 12
calls / 12.70 ms. JTok-M's
`_jtok_modes_kernel_route_vectorized` fell from 24 calls / about 22.11 ms to
12 calls / 11.02 ms. The other large backward kernels remained on their
existing expert-major/token-tile routes. No `jtok_reference` event appeared,
and the JSON reported `jtok_kernel_backend="triton"`.

This is distinct from the rejected `_jtok_backward_scaler_grad_kernel`
experiment: that candidate only changed the reduction of the global
`grad_scaler` and did not eliminate mode recomputation. The mode-cache change
is kept because it removes a complete backward mode-evaluation launch per
layer in the real flow while adding only the compact activation workspace.

### Accepted experiment: complete-row tile for the wide backward (2026-09-08)

The next bottleneck was not the shared scaler reduction.  The wide backward
was splitting the real hidden row into two 256-lane programs even when the
complete row fit the existing work budget.  For the real geometry
`hidden=512`, `d_seed=128`, `modes=4`, the budget is `67584` work items for
JTok (`top_k=1`) and `69632` for JTok-M (`top_k=2`), both below the
`131072` limit.

The planner now permits a complete 512-lane row only when both conditions are
true: the padded hidden width is at most the resource guard and the combined
seed/mode work fits the budget.  Otherwise it keeps the 256-lane multi-tile
path.  This is a dispatch change in `_backward_hidden_tile_plan`, not a
model-specific special case and not a change to the Leviathan base path.

With one complete row, `_jtok_backward_multi_tile_fast_kernel` computes the
normalization dot product locally and writes the token-local mode, residual,
and route-weight gradients directly.  It therefore removes the separate
`_jtok_backward_norm_dot_kernel` launch and the cross-tile atomics for those
token-local workspaces.  The shared `grad_scaler` still uses its inline FP32
atomic accumulation across tokens; the rejected standalone scaler-reduction
kernel was not reintroduced.  No dense expert-expanded workspace is added.

The model-free CUDA comparison used `n=8192`, BF16, seed `1729`, and the
full `hidden=512, d_seed=128, knots=16, modes=4` geometry.  The output was
bitwise equal between the tiled and complete-row partitions.  The largest
absolute gradient differences were `2.44e-4` for the seed and `1.95e-3` for
the residual surface in JTok; JTok-M remained finite with maximum absolute
differences no larger than `9.77e-4` for the residual surface.  These are
the expected FP32 accumulation-order differences, not a mathematical path
change.  The isolated backward median changed as follows, with no change in
allocated or reserved peak memory:

| isolated backward | two 256-lane tiles | complete 512-lane row | change |
| --- | ---: | ---: | ---: |
| JTok | 1.600 ms | 1.310 ms | -18.1% |
| JTok-M | 3.455 ms | 2.972 ms | -14.0% |

The complete NeoLLM training flow was then rerun with the same seed `1729`,
BF16, batch `64`, sequence `512`, 12 Transformer layers, CCE, AdEMAMix,
`torch.compile(mode="max-autotune")`, MXFP8 inactive, Delta inactive, two
validation steps, and one profile at step 12.  The comparison is against the
previous accepted mode-cache run, using stable steps 4--11:

| full flow | previous median | complete-row median | change | previous steps/s | new steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| JTok | 293.657 ms | 277.506 ms | -5.50% | 3.405 | 3.604 |
| JTok-M | 358.556 ms | 330.163 ms | -7.92% | 2.789 | 3.029 |

Peak memory was unchanged within the recorded precision:

| full flow | previous allocated/reserved | complete-row allocated/reserved |
| --- | ---: | ---: |
| JTok | 19.946 / 21.221 GiB | 19.946 / 21.221 GiB |
| JTok-M | 20.424 / 21.760 GiB | 20.424 / 21.760 GiB |

Validation did not regress: JTok changed from `131.269` to `131.392 ms`,
and JTok-M from `137.645` to `137.532 ms`.  The traces provide the causal
evidence.  The wide backward kernel fell from `22.534` to `9.854 ms` for
JTok and from `43.311` to `18.347 ms` for JTok-M, while the normalization-dot
kernel disappeared.  The mode, projection-gradient, and token-projection
kernels remained on their Triton paths, and no `jtok_reference` event was
observed.

The focused model-free JTok suite passed `44` tests with one unrelated
selection.  The full package run passed `3432` tests; its `38` failures were
the existing multi-process FSDP/vocab-parallel tests attempting to assign
multiple NCCL ranks to the single available GPU, reporting
`Multiple Ranks are using the same GPU/Partition`.  The complete-row test
itself passes independently and restores the planner limit after overriding
it, so unusual geometries retain deterministic dispatch behavior.

### Rejected experiment: masking zero-support spline atomics (2026-09-08)

The token-projection backward evaluates a compact quadratic B-spline, whose
support is at most three knots per seed coordinate for the usual uniform
grid.  A candidate added an explicit `distance < 1.5` mask to coefficient
loads and `grad_coeff` atomics, and also masked invalid rows.  It preserved
the dense equation and changed no public interface.

The candidate was measured after the complete-row tile was enabled, using the
same `n=8192`, BF16, `hidden=512`, `d_seed=128`, `knots=16`, `modes=4`
geometry.  It was slower in both variants:

| isolated backward | accepted path | masked-support candidate | change |
| --- | ---: | ---: | ---: |
| JTok | 1.310 ms | 1.428 ms | +9.0% |
| JTok-M | 2.972 ms | 3.306 ms | +11.2% |

Peak allocated and reserved memory were unchanged.  The dynamic predication
and masked vector transactions cost more than the zero-valued atomics saved
on this RTX 5090/Triton geometry.  The candidate was removed before a full
training run; the unmasked token-projection path remains authoritative.

### Rejected experiment: expert-fused projection-gradient kernel (2026-09-08)

The remaining JTok-M projection-gradient kernel reloads the same
`grad_surface` tile once per expert.  A candidate instead loaded one
token/hidden tile and looped over a bounded expert set, preserving
expert-major FP32 atomics and avoiding any dense expert-expanded workspace.
The dispatch guard limited it to small expert pools and bounded
`experts * (d_seed + modes)` work.

The candidate passed the model-free CUDA suite (`44 passed`) and preserved
the forward and backward equations, but it was slower in the isolated full
geometry (`n=8192`, BF16, `hidden=512`, `d_seed=128`, `knots=16`, `modes=4`,
`top_k=2`):

| isolated backward | accepted expert-major path | fused-expert candidate | change |
| --- | ---: | ---: | ---: |
| JTok | 1.285 ms | 1.285 ms | no applicable change |
| JTok-M | 2.972 ms | 3.561 ms | +19.8% |

Peak memory was unchanged.  Reusing the surface load did not compensate for
the longer expert loop and its register/atomic pressure on the RTX 5090.
The candidate was removed before a full-flow run; the expert-major kernel
remains authoritative, including for unbalanced routing.

### Accepted experiment: autotuned token-to-spline backward launch (2026-09-08)

The remaining token-local backward kernel,
`_jtok_backward_token_projection_grad_block_kernel`, was still using a fixed
`num_warps=4, num_stages=1` launch.  This kernel owns the derivative of the
compact B-spline projection: one program handles one `(token, top-k route)`
and vectorizes the `d_seed` coordinates while accumulating `grad_z` and
`grad_coeff`.  It is therefore independent from the global `grad_scaler`
reduction, which remains fused in `_jtok_backward_multi_tile_fast_kernel`.

The accepted change puts only this block kernel behind Triton's autotuner.
The candidate set is deliberately small and geometry-keyed:

```text
num_warps=2, num_stages=1
num_warps=4, num_stages=1
num_warps=8, num_stages=1
num_warps=4, num_stages=2
```

The key includes `N`, `D_SEED`, `NUM_KNOTS`, `NUM_MODES`, `TOP_K`,
`BLOCK_D`, and the mask flag.  `grad_z` and `grad_coeff` are reset between
autotune candidates, so benchmarking a candidate cannot accumulate its
partial gradients into the selected run.  The equations, launch grid,
workspace sizes, public API, and legacy Leviathan path are unchanged.  The
first encounter with a new geometry pays Triton's compile/benchmark cost;
subsequent calls use the cached configuration.

The model-free CUDA check used seed `1729`, BF16, `n=8192`,
`hidden=512`, `d_seed=128`, `knots=16`, and `modes=4`.  Compared with the
fixed launch, using the same forward+backward measurement and excluding the
separate auxiliary-statistics calculation, the median changed as follows:

| isolated backward | fixed launch | autotuned launch | change |
| --- | ---: | ---: | ---: |
| JTok | 1.310 ms | 1.235 ms | -5.8% |
| JTok-M | 2.972 ms | 2.939 ms | -1.1% |

Peak allocated/reserved memory stayed within measurement noise (JTok-M was
about `196 MiB` allocated and `428 MiB` reserved in the isolated run).  The
focused model-free suite passed `44` tests with one unrelated selection.

The full NeoLLM training flow was measured separately for JTok and JTok-M
with seed `1729`, BF16, batch `64`, sequence `512`, 12 Transformer layers,
CCE, AdEMAMix, `torch.compile(mode="max-autotune")`, MXFP8 inactive, Delta
inactive, MEAP/MiLe/MU/NITP enabled, two validation steps, and a profile at
step 12.  Both runs reported `jtok_kernel_backend="triton"` and did not
emit a `jtok_reference` event.  Stable steps `4--11` changed as follows:

| full flow | fixed-launch median | autotuned median | change | fixed steps/s | autotuned steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| JTok | 277.506 ms | 276.548 ms | -0.35% | 3.604 | 3.616 |
| JTok-M | 330.163 ms | 328.567 ms | -0.48% | 3.029 | 3.044 |

Peak memory did not change:

| full flow | allocated | reserved |
| --- | ---: | ---: |
| JTok | 19.946 GiB | 21.221 GiB |
| JTok-M | 20.424 GiB | 21.760 GiB |

The trace attributes the improvement to the target kernel itself.  Its total
time over the 12 profiled layers fell from `11.990` to `11.411 ms` for JTok
and from `20.708` to `18.895 ms` for JTok-M.  The wide backward, projection
gradient, and mode kernels remained on their existing Triton paths; the
autotuner did not alter the complete-row dispatch or introduce a dense expert
workspace.  Validation also remained stable (`130.667 ms` for JTok and
`137.435 ms` for JTok-M on the second validation step).

The first training step remains a cold compilation event (about six minutes
in this complete max-autotune probe) and is not used as the steady-state
comparison.  No global compile policy, optimizer flag, MXFP8 setting, model
configuration, tokenizer, or checkpoint was changed.
