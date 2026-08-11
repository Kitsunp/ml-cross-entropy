# CCE and `torch.compile`

## Problem

Tracing the Python and Triton implementation of CCE directly through Dynamo is
not a stable compiler boundary. The implementation legitimately uses Python
metadata and values that are not tensors: the installed PyTorch version, BF16
capability, matmul-precision strings, Triton launch metadata, and a
data-dependent list of valid labels. Dynamo therefore split and specialized the
caller even though the mathematical CCE operation itself was unchanged.

Disabling compilation around the model's loss wrapper hid those internals, but
it still forced one graph before CCE and another after it. It also made the
model responsible for knowing an implementation detail of this package.

## Compiler boundary

The supported CUDA training path now enters CCE through registered
`torch.library.custom_op` forward and backward operations while Dynamo is
compiling. Fake implementations describe their output shapes to Dynamo, and an
explicit autograd formula delegates to the same `LinearCrossEntropyFunction`
forward and backward implementations used by eager execution. The Triton
mathematics, saved tensors, MiLe weighting, μ-loss, MEAP metrics, bias handling,
and gradient filters are therefore not duplicated.

The optimized boundary currently covers CUDA inputs whose effective compute
dtype is BF16/FP16 with `mean` reduction, positive label shift, no returned
LSE, and no vocabulary-parallel group. This includes FP32 storage inputs under
CUDA BF16/FP16 autocast, as produced by operations such as RMSNorm. Other
public configurations retain the existing eager/traced fallback.
The public CCE shape, objective-parameter, reduction, and device-capability
validations run before compiler dispatch. When CUDA autocast is active,
`filter_eps="auto"` is resolved from the autocast compute dtype rather than the
input storage dtype, matching eager CCE. The same dtype is passed explicitly to
the fake witness used by AOTAutograd, so the compiled backward and runtime
backward agree when FP16 inputs run under BF16 autocast or vice versa. If the
resolved epsilon is `None`, both gradient-filter flags are disabled before
dispatch, as they are in eager CCE.

The boundary also transports whether CUDA autocast was active and recreates
that context inside both the opaque forward and reused backward. Inductor is
not required to preserve the caller's ambient autocast state while invoking a
custom operator. Relying on that ambient state can otherwise run the compiled
forward in FP32 while its metadata witness and backward assume BF16/FP16,
silently changing loss and gradients.
The compiler boundary exposes saved valid-label tensors at the static capacity
implied by the input shape. Only their compact prefixes are populated; backward
reconstructs the valid indices inside the opaque operator and slices those
prefixes before launching the existing kernels. Callers therefore do not need
to enable dynamic-output tracing or mutate global Dynamo configuration.

The operations are tagged `cudagraph_unsafe`: their Triton drivers allocate
dynamic temporary tensors whose ownership is not compatible with Inductor's
CUDA-graph pool. This prevents unsafe CUDA-graph capture of the opaque kernel;
it does not introduce an FX graph break around CCE.

## Validation

The four objective combinations below were tested through the public API with
tensor work both before and after CCE:

| Objective | Disabled wrapper: graphs / breaks | Compiler boundary: graphs / breaks |
|---|---:|---:|
| CCE | 2 / 1 | 1 / 0 |
| CCE + MiLe | 2 / 1 | 1 / 0 |
| CCE + μ-loss | 2 / 1 | 1 / 0 |
| CCE + MiLe + μ-loss | 2 / 1 | 1 / 0 |

Changing the number of valid labels between calls did not compile another
graph. The regression test uses FP32 storage under both BF16 and FP16 CUDA
autocast and ten distinct valid-label counts, crossing Dynamo's default
eight-recompile threshold without producing another caller graph. Losses,
metrics, and FP32 gradients were compared against eager execution for all four
CCE/MiLe/μ-loss combinations. `torch.library.opcheck` also passes schema, autograd
registration, FakeTensor metadata, and dynamic AOT-dispatch validation. This
caught and prevented aliasing between absent optional-output placeholders.
Focused metadata cases also cover non-contiguous embeddings and partial
gradient ownership: fake embedding gradients advertise the contiguous layout
returned at runtime, and `logit_avg` is vocabulary-sized whenever any input or
bias needs gradients and either gradient-filter flag is enabled.

At the training-loss shape `B=64`, `S=512`, `D=512`, `V=64,402` with 27,310
valid labels, paired alternating steady-state measurements produced:

| Objective | Disabled wrapper (ms) | Compiler boundary (ms) | Change |
|---|---:|---:|---:|
| CCE | 82.182 | 82.113 | -0.08% |
| CCE + MiLe | 84.142 | 83.814 | -0.39% |
| CCE + μ-loss | 82.832 | 82.766 | -0.08% |
| CCE + MiLe + μ-loss | 84.966 | 84.671 | -0.35% |

Each row contains 30 samples per implementation. A separate 40-sample
MiLe + μ-loss run measured 85.341 versus 85.258 ms (-0.10%), confirming the
direction but also that the sub-millisecond magnitude is sensitive to normal
run-to-run noise. Peak allocated memory was equal. These differences should not
be generalized as universal kernel-speed gains; the material result is
eliminating the graph split without adding steady-state cost.

Allocator stability should be evaluated after one complete warm-up over the
expected valid-label range. Compilation and Triton autotuning legitimately
raise the initial peak. In a focused repeated-range diagnostic, reserved memory
was identical across the two post-warm-up cycles; the regression criterion is
no cumulative graph-pool or reservation growth, not identical instantaneous
workspace usage inside every kernel.

All performance numbers in this investigation were measured on the available
RTX 5070 Ti. Correctness and dispatch do not depend on that model name. Other
GPU families should run the focused correctness tests and an A/B benchmark;
`CCE_AUTOTUNE=1` remains available when the cold-start tuning cost is acceptable.

The shared-memory fallback correction was also measured with
`cce_kahan_full_c`, MiLe, μ-loss, and metrics enabled. The table compares the
previous `32 x 128 x 32` fallback with the capability-selected
`128 x 128 x 32`, four-warp, three-stage schedule:

| Scale case | Previous total (ms) | Corrected total (ms) | Change |
|---|---:|---:|---:|
| `B=64,S=512,D=512,V=64,402` | 92.76 | 83.97 | -9.48% |
| vocabulary `V=128,000` | 181.39 | 158.60 | -12.56% |
| hidden size `D=1,024` | 177.34 | 153.84 | -13.25% |
| context `S=1,024` | 184.23 | 167.77 | -8.93% |
| batch `B=128` | 184.50 | 166.74 | -9.63% |

Peak allocated memory was unchanged in every pair. The gain came from fewer,
larger backward programs; forward time was approximately unchanged.
