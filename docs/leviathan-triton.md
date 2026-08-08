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
fall back to the differentiable dict-based reference path.  It keeps the
original parameter tensors connected to autograd and honors an explicit
`knot_grid`.  `LeviathanGenerator` creates its trainable tensors in
`config.dtype`, so `.cuda()` alone is sufficient to select the BF16 kernel.
When gradients are disabled—or all generator parameters are frozen—a separate
CUDA boundary runs the forward with `save_intermediates=False` and avoids
allocating the training checkpoints.

The package does not set `torch._dynamo.config.capture_dynamic_output_shape_ops`
or any other global Dynamo/`torch.compile` option.  The LEV custom-op boundary
has one graph and zero breaks under the default Dynamo configuration.  The
compact CCE consumer can still report a break at CCE's pre-existing
`cut_cross_entropy.cce_forward` dynamic-output operator; this integration does
not enable a config flag or modify that CCE operator.

## Measured kernel point

The directed sweep was run on an RTX 5070 Ti (SM120, BF16, 10 GB test cap) with
`LEV_DOT=1`.  For long-token workloads the backward dDelta selector uses
`BLOCK_M=128, BLOCK_D=1, BLOCK_R=64, num_warps=4, num_stages=1`.  Short inputs
keep `BLOCK_M=32`.  The forward dot kernel uses BM32/w8/s1 and unrolls its
per-seed dimension loop by four; larger tiles, gather fusion, TF32x3, and a
cuBLAS replacement were measured and rejected because they were slower or
used more memory.

Representative candidate-only results at D=2048,N=4096 were approximately
1.1 ms forward, 3.0 ms backward, 0.14 GB peak, one synchronization event,
and no N-scaled dDelta workspace.  The authoritative harness enforces a 10 GB
peak budget; use the same harness for a target-GPU comparison before enabling
architecture-specific overrides.

The backward dot path also precomputes the seed-independent `dM * M` term
in-place (`LEV_PREMUL_DMM=1`, the default) before the per-seed loop.  This
preserves the dDelta liveness optimization: the CUDA profile reduced the
dominant dDelta kernel from about 3.50 ms to 2.75 ms without increasing peak
memory.  Set `LEV_PREMUL_DMM=0` only for a diagnostic comparison; it does not
change the reference fallback or the public API.
