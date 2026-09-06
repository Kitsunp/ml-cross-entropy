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

The legacy `leviathan_embedding_compiler_safe` call still produces the
embedding through the existing Leviathan kernel. JTok additionally needs the
compositional seed `z`. The new
`leviathan_embedding_with_seed_compiler_safe` helper repeats only the cheap
base-`k` codebook lookup/sum in a compiler-visible differentiable bridge. It
does not repeat the projection, spline, or output stages. This is necessary
because the legacy custom op saves its internal seed for its own backward and
does not expose that saved tensor as a differentiable output.

The bridge is tested against `leviathan_forward_ref`: its seed and embedding
match the reference, and a loss through the returned seed produces finite
codebook gradients.

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
