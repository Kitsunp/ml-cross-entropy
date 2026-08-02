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

The optimized boundary currently covers BF16/FP16 CUDA inputs with `mean`
reduction, positive label shift, no returned LSE, and no vocabulary-parallel
group. Other public configurations retain the existing eager/traced fallback.
Callers that continue tensor work after CCE must enable
`torch._dynamo.config.capture_dynamic_output_shape_ops = True`, because the
number of saved valid-label rows is data dependent. The NeoLLM integration sets
that option when CCE is available.

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
graph. Losses, metrics, and gradients were compared against eager execution for
all four combinations. `torch.library.opcheck` also passes schema, autograd
registration, FakeTensor metadata, and dynamic AOT-dispatch validation. This
caught and prevented aliasing between absent optional-output placeholders.

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
