# PyTorch 2.14 CUDA integration

This integration is deliberately limited to runtime coordination around the
existing CCE, Leviathan, and PolyNorm kernels. It does not change their math,
launch geometry, autograd formulas, allocator implementation, or the caller's
`torch.compile` policy.

## Version gate

`cut_cross_entropy.torch_2_14` exports these booleans:

- `TORCH_2_14_OR_NEWER`: the installed Torch package is at least 2.14.
- `TORCH_2_14_CUDA_INTEGRATION`: the version requirement is met and the
  integration has not been disabled.
- `TORCH_2_14_GRAPH_ANNOTATIONS`: CUDA Graph profiling labels were explicitly
  requested.
- `TORCH_2_14_CUDA_MEMORY_POOL`: allocations are routed through the shared
  PyTorch 2.14 CUDA pool.
- `TORCH_2_14_CUDA_KERNEL_CONTEXT`: either allocation routing or graph
  annotation is active, so compiler boundaries enter the integration context.
- `TORCH_2_14_MEMORY_ANNOTATIONS`: allocation annotations were explicitly
  requested.

The integration is enabled automatically on Torch 2.14 or newer. For an A/B
comparison in the same environment, set
`CUT_CROSS_ENTROPY_TORCH_2_14_INTEGRATION=0` before Python starts. On Torch
2.13 and older all the helpers are no-ops, even if the environment variable is
set.

## What changes

Set `CUT_CROSS_ENTROPY_GRAPH_ANNOTATIONS=1` to make the three compiler
boundaries use `torch.cuda.graph_annotations.mark_kernels` to identify forward,
inference, and backward work in annotated CUDA Graph captures. It is opt-in so
ordinary training does not enter a Python profiling context on each unsafe
custom-op invocation. Every region passes `backward=False` because CCE,
Leviathan, and PolyNorm already have separately registered backward operators.
This prevents the profiler from attributing a backward kernel twice.

The metadata is attached at the existing Python/compiler boundaries; it does
not register a second CUDA extension or replace any kernel. The mapping is:

| Boundary | Component | Phase | Metadata operator |
| --- | --- | --- | --- |
| `cce.forward` / `cce.backward` | CCE | forward / backward | `cut_cross_entropy::cce_forward` / `cut_cross_entropy::cce_backward` |
| `leviathan.forward`, `.inference`, `.backward` | Leviathan | forward / inference / backward | `cut_cross_entropy::leviathan_forward`, `..._inference`, `..._backward` |
| `polynorm.forward`, `.inference`, `.backward` | PolyNorm | forward / inference / backward | `cut_cross_entropy::polynorm_forward`, `..._inference`, `..._backward` |

The annotations are diagnostic labels for CUDA Graph/profiler inspection. The
actual custom-op registration and the separately registered backward operators
remain the ones already present in the project.

After PolyNorm's first successful launch of each distinct device, dtype, shape,
and dropout configuration, its graph-safe boundary calls
`torch.compiler.cudagraph_mark_warmup_incomplete()`. During CUDA Graph Trees
warmup this requests one additional eager iteration, so a graph is not recorded
immediately after CuTe first specializes a new geometry. A process-local set
removes the hook from the steady-state path after the first launch. CCE and
Leviathan deliberately do not call this hook because their custom operators
remain tagged `cudagraph_unsafe` and are excluded from graph capture.

Optional allocation labels are available through
`CUT_CROSS_ENTROPY_MEMORY_ANNOTATIONS=1`. They use the PyTorch 2.14
`torch.cuda.memory._annotate_tensor()` diagnostic API and only become visible
when CUDA memory-history recording is active. They are off by default because
they are diagnostic metadata, not a throughput optimization.

By default on Torch 2.14+, allocations made while the three external-kernel
boundaries execute are routed through `torch.cuda.use_mem_pool()` when the
caller is eager or uses a CUDA-graph path that does not belong to Inductor.
The pool is created lazily, one per CUDA device, and shared by CCE, Leviathan,
and PolyNorm so released blocks can be reused across components. It uses
`torch.cuda.MemPool(use_on_oom=True)`, allowing the general allocator to use
its released blocks under memory pressure. Set
`CUT_CROSS_ENTROPY_TORCH_2_14_MEMORY_POOL=0` before Python starts to disable
only pool routing while retaining other 2.14 integration features.

The boundary automatically skips this external pool while Inductor CUDA Graph
Trees owns the current device. In PyTorch 2.14, Inductor validates returned
storages against its own graph pool; nesting a second `MemPool` around a
custom-op implementation makes those outputs look foreign to
`cudagraph_trees.check_memory_pool` and aborts the compiled step. The guard
uses the read-only private `get_manager(..., create_if_none_exists=False)`
probe because PyTorch 2.14 does not expose a public predicate for the current
Graph Trees phase. If that compatibility probe changes or fails, the optional
external pool is skipped for safety. This does not disable `torch.compile`,
change CUDA Graph capture policy, or change any kernel.

## Deliberate exclusions

`torch.Tag.cudagraph_unsafe` remains on CCE and Leviathan. The new hooks do not
prove that their data-dependent saved tensors are safe to replay, and changing
that tag would change the training compiler policy.

No pool is created on Torch 2.13 or older. The implementation also avoids one
pool per kernel, which would unnecessarily isolate released blocks. Pool memory
can increase `memory_reserved()` even when `memory_allocated()` is unchanged;
both values must therefore be reported in full-model validation.

The pool is intentionally not a hard-coded 10-GiB allocator limit. A benchmark
may use `--memory-limit-gib 10` as a resource guard, but production memory
capacity remains controlled by the process and PyTorch allocator. PyTorch 2.14
builds without both `MemPool` and `use_mem_pool` fail the capability probe and
automatically leave pool routing disabled. The external pool is also bypassed
for an active Inductor Graph Tree, even when the environment variable requests
pool routing.

## Regression fixed: Inductor Graph Trees cross-pool storage

The failure was reproduced on Torch `2.14.0+cu132` with
`torch.compile(mode="max-autotune")`: a minimal custom CUDA op succeeded on
the first warmup call and then failed on the next call with
`These storage data ptrs are not allocated in pool (0, 1) but should be ...`.
The traceback ended in Inductor's `cudagraph_trees.check_memory_pool`, after
Triton's `triton_mm` autotune. The failure was allocator ownership bookkeeping,
not a Triton matmul correctness or autotune failure. With the guard active, the
same four-call reproduction completes successfully while the shared pool is
still used by eager calls. The regression test is
`test_kernel_region_does_not_nest_pool_inside_inductor_graph_tree`.

## Reproducible comparison

Run every case in a fresh process so allocator and compiler caches do not leak
between variants. The validation environment used for this change was Torch
2.14.0+cu132, Triton 3.8.0, CUDA 13.2, and an NVIDIA GeForce RTX 5090.

```bash
export PYTHONPATH=/workspace/ml-cross-entropy
export PYTHONHASHSEED=20260905
python benchmark/torch_2_14_cuda_integration.py \
  --trials 3 --warmup 10 --iterations 50 --seed 20260905 \
  --memory-limit-gib 10 \
  --output benchmark/results/torch214_cuda_integration/ab_3x50.json
```

The runner starts a fresh process for every component and A/B state, alternates
the order of the enabled and disabled states, and records raw samples plus
medians. The seed is fixed to `20260905`; it controls the tensor construction
and `PYTHONHASHSEED`, but it does not eliminate compiler/driver noise, so the
reported numbers are medians over independent processes.

## Isolated validation result

The following result was obtained remotely on 2026-09-05 with the environment
listed above. It exercises the compiler-safe CCE path, CuTe PolyNorm path, and
Leviathan forward/training path. “On” means the 2.14 integration and shared
CUDA pool were enabled; “off” disables the integration in a fresh process.

| Case | Off | On | Change |
| --- | ---: | ---: | ---: |
| CCE training step | 2.507 ms | 2.733 ms | +9.03% |
| PolyNorm total forward+backward | 0.489 ms | 0.641 ms | +30.93% |
| Leviathan forward, 1,024 tokens | 0.316 ms | 0.338 ms | +7.15% |
| Leviathan training, 1,024 tokens | 1.550 ms | 1.640 ms | +5.74% |
| Leviathan forward, 4,096 tokens | 0.558 ms | 0.571 ms | +2.45% |
| Leviathan training, 4,096 tokens | 3.371 ms | 3.261 ms | -3.27% |

Peak live allocation was unchanged in these isolated cases: CCE was about
290 MiB, PolyNorm about 80 MiB, and Leviathan was 41/56 MiB at 1,024 tokens
and 60/102 MiB at 4,096 tokens for forward/training respectively. CCE's
`memory_reserved()` was about 294 MiB off versus 360 MiB on; PolyNorm was
138 MiB versus 140 MiB. This is expected pool behavior: reserved memory is not
the same as live tensor memory.

The full subprocess wall time, including import, compilation, warmup, and
measurement, changed by less than 2% in the 3×50 run. The steady-state isolated
latency is not yet a speed win for every small geometry; the purpose of this
change is better 2.14 allocator/compiler integration and component attribution.
The full NeoLLM training run remains the deciding validation because its larger
and repeated allocations can amortize the boundary cost. These isolated
results are retained in
`benchmark/results/torch214_cuda_integration/memory_pool_compile_ab_3x50.json`
and the longer 3×200 run is in
`benchmark/results/torch214_cuda_integration/memory_pool_ab_3x200.json`.
