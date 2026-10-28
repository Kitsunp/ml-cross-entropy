# Copy-Paste language-model loss

`cut_cross_entropy.copy_paste_loss` trains a causal vocabulary/copy mixture
without materializing a `[batch, predicted_tokens, vocabulary]` logit tensor.
It is a CUDA operation and supports `torch.compile`; callers should keep a
separate dense reference when CPU execution is required.

For vocabulary logits `z`, `L = logsumexp(z)`, copy-gate logit `g`, and a
pointer softmax over the causal context plus its sink, define

```
c = sigmoid(g - L)
a = 1 - c + c * alpha_sink
p(v) = a * softmax(z)[v] + c * sum(alpha[j] for j where input[j] == v)
```

The gate is equivalent to adding a `<copy>` row to the vocabulary classifier;
it does not require adding a tokenizer ID. The sink sends its mass back to the
ordinary vocabulary distribution. Repeated source IDs contribute to one class.

The wrapper accepts predicted hidden states and queries `[B,T,D]`, classifier
`[V,D]`, pointer keys `[B,S,D]`, source IDs and their validity mask `[B,S]`,
and aligned target IDs `[B,T,K]`. Invalid target IDs are ignored. For patch
size `P`, a query for patch `t` sees source positions through `(t+1)*P-1`;
the caller constructs the repeated patch keys. `gate` and `sink` are `[B,T]`
logits; the caller applies the `D**-0.5` scale to the sink dot product. IDs in
the source must be valid vocabulary indices, including masked positions.

The return tuple is `(loss, ce, mile_delta, mu, topk_counts)`. `loss` is the
mean over valid targets after optional detached MiLe weighting plus optional
mu-loss. `ce`, `mile_delta`, and `mu` are detached scalars. When requested,
`topk_counts` stores hits at 1 and 3, an inactive zero Top-5 slot, the number
of valid targets, and the nonfinite count. Top-k measures slot zero only in the
token phase (`patch_size=1`); remaining padded target slots can be ignored.
MiLe group fractions use valid-target counts, including partially filled patches.
Ignored rows do not set the exponent's numerical scale. When groups are
provided, each population is shifted and normalized independently, avoiding
underflow between populations with very different entropies or large gamma.

With `return_copy_metrics=True`, the tuple has a sixth detached tensor with six
**sums**, ordered as follows. Divide the first five by the last after summing
over all batches and ranks; omit means when the count is zero.

1. `gate`: `c`, the probability of opening the copy route.
2. `sink`: `alpha_sink`, the sink probability inside the pointer softmax.
3. `effective_gate`: `c * (1 - alpha_sink)`, the probability assigned to
   the non-sink copy distribution.
4. `target_responsibility`: `c * sum(alpha[j] for input[j] == y) / p(y)`,
   the fraction of the target's predicted probability contributed by copying.
5. `target_available`: one if the causal, unmasked source contains `y`.
6. `valid_targets`: the number of valid target slots.

Gate and sink are weighted by the number of valid target slots in their row.
This gives partially filled patches the same token-level interpretation as
ordinary predictions. The sink mean is not weighted conditionally on the gate.
No IDs, text, activation vectors, or example-level records leave the device.
Diagnostics are detached and never participate in the loss or gradients.

CCE supplies the base log-normalizer, weighted-logit moment, target logits,
and optionally the three best base classes. The Triton sparse reduction groups
source tokens by ID, adds their copy mass, and corrects the base entropy. For
ranking, the base top three plus copied classes suffice: an uncopied class
outside the base top three cannot overtake all of those base classes. Gradients
flow through the normalizer, target logits, gate, pointer scores, and sink;
MiLe entropy and diagnostic counts do not carry gradients.

Before the vocabulary kernels, valid rows are stably compacted on the GPU
into a fixed-size index buffer padded with a sentinel. Empty CCE tiles skip
their matrix products; results are scattered back to their original rows and
the same indices are reused in backward. The output and compiler shapes remain
static, and neither the number of valid rows nor the diagnostics require a host
synchronization. Savings depend on the fraction and arrangement of ignored
targets; an entirely valid batch still pays the small compaction overhead.

The largest pointer and candidate tensors scale with `B*T*S`, rather than
`B*T*V`. Other model activations, optimizer state, CCE workspaces, and compiler
buffers still contribute to total VRAM; timing and peak memory should be
measured on the complete training step and target GPU. Vocabulary parallelism
and softcapped logits are not part of this API.
