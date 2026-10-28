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

CCE supplies the base log-normalizer, weighted-logit moment, target logits,
and optionally the three best base classes. The Triton sparse reduction groups
source tokens by ID, adds their copy mass, and corrects the base entropy. For
ranking, the base top three plus copied classes suffice: an uncopied class
outside the base top three cannot overtake all of those base classes. Gradients
flow through the normalizer, target logits, gate, pointer scores, and sink;
MiLe entropy and diagnostic counts do not carry gradients.

The largest pointer and candidate tensors scale with `B*T*S`, rather than
`B*T*V`. Other model activations, optimizer state, CCE workspaces, and compiler
buffers still contribute to total VRAM; timing and peak memory should be
measured on the complete training step and target GPU. Vocabulary parallelism
and softcapped logits are not part of this API.
