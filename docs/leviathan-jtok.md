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
