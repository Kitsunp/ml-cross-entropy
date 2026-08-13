# WIP CuTe MLP experiment — DO NOT MERGE

This branch is a preservation point for an unfinished CuTe DSL training MLP.
It is intentionally not a production proposal and must not be merged as-is.

## Frozen scope

- Base commit: `9f4ad72` (`Fix PolyNorm dispatch-aware checkpoint routing`).
- GPU used for the measurements: NVIDIA RTX 5090, SM120.
- Software: PyTorch `2.13.0+cu130`, CUDA `13.0`, TorchAO `0.18.0`,
  NVIDIA CUTLASS DSL `4.7.0`, pytest `9.1.1`.
- The model and training scripts outside this repository were not modified.
- No production flag, Torch compile mode, Dynamo limit or VRAM limit was added.
  The 10 GiB limit belongs only to benchmark/test processes.

The detailed mathematical, numerical, lifetime and reproduction contract is in
`docs/neollm-mlp-cute.md`.

## Exact TorchAO conversion contract found in the training script

The training script calls `convert_to_float8_training` without a custom config,
therefore the default `Float8LinearConfig()` applies.  Its filter:

1. excludes `StackMemory` and `LeviathanGenerator` subtrees;
2. excludes names containing `embed_tokens`, `lm_head` or `token_generator`;
3. converts only `nn.Linear` modules whose input and output widths are both
   divisible by 16.

For the measured 512/1536 MLP this yields:

| Projection | Shape | Training precision |
|---|---:|---|
| FAN | 512 -> 510 | BF16 |
| gate | 544 -> 1536 | TorchAO default FP8 |
| up | 544 -> 1536 | TorchAO default FP8 |
| down | 1536 -> 512 | TorchAO default FP8 |

The experimental operator therefore does not assume that the complete MLP is
quantized. It receives the BF16 FAN result and preserves independent gate/up
weight scales and TorchAO's forward/backward policies.

## Implemented and verified

- Exact mathematical reference for FAN, exclusive GUPN and the complete MLP.
- Exclusive CuTe GUPN forward: gate-row multiplier, FP32 PolyNorm reductions,
  Philox dropout replay, Hadamard product and down-column multiplier.
- CuTe GUPN backward that recomputes the forward internals, uses deterministic
  fixed-order reductions and does not use global floating-point atomics.
- Public custom autograd boundary that returns only `Z` and does not retain the
  PolyNorm branches, dropout mask, Hadamard tensor or `Z` for backward.
- Experimental TorchAO-exact gate/up wrapper that shares the dynamic cast of the
  common FAN input. It is a correctness/lifetime prototype, not a fast kernel.
- Benchmarks that clear warm-up gradients before the memory baseline and report
  allocated, reserved and transient memory independently.

Frozen focused verification:

```text
32 passed, 14 deprecation warnings in 11.21 s
git diff --check: passed
```

Command:

```bash
PYTHONPATH=. /venv/main/bin/python -m pytest \
  tests/test_mlp_reference.py tests/test_mlp_gupn.py \
  tests/test_mlp_fp8.py tests/test_polynorm.py -q
git diff --check
```

The suite includes five full-graph compiled training steps with
`mode="max-autotune"`, exact comparison with two default TorchAO FP8 linears,
gradient checks and a saved-tensor lifetime assertion.

## Measured RTX 5090 results

Exclusive GUPN:

| Workload | max-autotune | CuTe | Latency reduction | Reserved pool |
|---|---:|---:|---:|---:|
| forward, 512 x 1536 | 0.06634 ms | 0.02750 ms | 58.5% | CuTe scratch 0 B |
| forward, 32768 x 1536 | 0.60280 ms | 0.21147 ms | 64.9% | CuTe scratch 0 B |
| forward+backward, 512 x 1536 | 0.31278 ms | 0.25192 ms | 19.5% | 64 -> 44 MiB |
| forward+backward, 32768 x 1536 | 2.45339 ms | 1.44418 ms | 41.1% | 1780 -> 724 MiB |

Full one-layer TorchAO FP8 MLP:

| Workload | Reference | Experimental route | Result |
|---|---:|---:|---|
| batch 1, length 512 | 0.88149 ms | 0.88102 ms | effectively tied |
| six layers, batch 1, length 512 | 5.32379 ms | 5.35621 ms | experimental 0.61% slower |
| batch 64, length 512 | 5.356 ms | 10.454 ms | experimental slower |

At batch 64 the recomputed experimental body reduced the isolated reserved peak
from 2346 to 1796 MiB, but its opaque staged FP8 GEMMs execute again in backward.
This proves the lifetime opportunity while also proving that this Python-level
composition cannot be the final performance path.

## Why this branch is not mergeable

- Gate/up and down FP8 GEMMs are not yet physical CuTe kernels.
- The staged FP8 custom op saves memory but regresses full-layer latency at real
  row counts because Inductor cannot fuse operations inside the opaque boundary.
- FAN and down have not yet been connected into the final `forward -> Y only`
  custom-autograd contract.
- The bounded REG/SMEM/XOR geometry search for the physical dual MMA is pending.
- Nsight Compute is installed, but hardware counters were denied by
  `ERR_NVGPUCTRPERM`; no bank-conflict claim is made.
- No current RTX 5070 Ti run was available, so no 5070 Ti numbers are invented.
- Long 500/5000-step stability gates and full-training loss/scale comparisons
  remain pending.

## Next engineering slice

Build a physical SM120 dual gate/up FP8 CuTe kernel that stages one BF16/FP8 FAN
tile, reuses it for two independent MMA streams, keeps separate FP32 accumulators
and scales, and feeds logical-BF16 fragments directly into GUPN without writing
gate/up tensors to HBM.  Only after correctness and five-step compilation should
down projection and the final full-body autograd boundary be added.

Weight concatenation was measured as a weak ceiling only: it improved the two
projection pair by about 0.12% at 512 rows and 2.52% at 32768 rows while changing
the accumulation geometry. It is deliberately not the selected implementation.
