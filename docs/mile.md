# MiLe Loss in Cut Cross Entropy

[MiLe Loss](https://aclanthology.org/2024.findings-naacl.18/) reweights the
next-token cross-entropy of each valid token using the entropy of the model's
current prediction. The CCE implementation keeps the released method's two
important operational details: the weights are normalized to mean one and are
detached from autograd.

![MiLe kernel data flow](../assets/mile_kernel.svg)

## How to read the diagram

`N` is the number of valid tokens after shift, padding, and `ignore_index`
filtering; `V` is the vocabulary size. Subscript `n` selects a token block and
subscript `v` selects a vocabulary tile.

- **Panel (a)** is still the ordinary blockwise CCE forward. Each temporary
  `z_nv` tile lives in registers. While vocabulary tiles stream through the
  program, CCE reduces them into three FP32 values per valid token: `LSE_n`,
  `m_n = E_p[z_n]`, and `NLL_n`.
- **Panel (b)** is the MiLe-specific post-processing. `LSE_n - m_n` gives the
  predictive entropy, the entropy produces a raw weight `a_n`, and the global
  mean makes the final weights average to one. The two blue boxes are execution
  strategies for the same formula, not two different objectives.
- **Panel (c)** is CCE backward. It reconstructs one logit tile, forms the usual
  cross-entropy gradient, loads one saved scalar `w_n`, and multiplies the whole
  token row by it before accumulating `dE` and `dC`.

The long `NLL_n` route in panel (b) is intentionally separate from weight
normalization: NLL does not determine the weight. It meets the detached weight
only at the final product `w_n * NLL_n`.

## Objective

For valid token `i`, logits `z_i`, probabilities `p_i`, and target `y_i`:

```text
H_i = logsumexp(z_i) - sum_j p_ij z_ij
a_i = (1 + H_i) ** gamma
w_i = stop_gradient(a_i / mean_valid(a))
L_MiLe = mean_valid(w_i * -log p_i,y_i)
```

The normalization preserves `mean_valid(w) = 1`, so MiLe changes the allocation
of gradient across tokens rather than deliberately increasing the average loss
scale. The stop-gradient does **not** remove the effect of the weight:

```text
dL/dtheta = mean_valid(w_i * dCE_i/dtheta)
```

It only removes the additional `CE_i * dw_i/dtheta` term. Weights are recomputed
from the current model on every forward pass and then treated as constants for
that backward pass.

For `gamma > 0`, higher-entropy tokens receive a larger weight relative to
lower-entropy tokens in the same valid-token batch. Because the weight is
detached, this is an allocation rule for the CE gradient; it is not a direct
entropy-minimization gradient. At `gamma=0`, every normalized weight is one and
the objective reduces to ordinary CCE.

## CCE execution path

MiLe is integrated into the existing CCE pipeline and never materializes the
`tokens x vocabulary` logit matrix in global memory.

1. CCE compacts `shift`, padding, and `ignore_index` into the valid-token domain.
2. The blockwise LSE forward kernel computes `logsumexp(z_i)`, the correct logit,
   and the probability-weighted mean logit `sum_j p_ij z_ij`.
3. `cce_mile_forward_kernel` constructs the detached, mean-normalized weights and
   weighted token losses.
4. The ordinary CCE reduction produces `mean`, `sum`, or unreduced output.
5. The CCE backward kernel reconstructs each logit block, forms
   `p_i - one_hot(y_i)`, loads one scalar MiLe weight per token, and scales the
   block before accumulating classifier and embedding gradients.

The entropy identity in step 2 is exact for the logits processed by CCE:

```text
H(p_i) = logsumexp(z_i) - E_p[z_i]
```

No probability or full-logit matrix is stored.

## Fused MiLe weight kernels

The token-wise post-processing has two execution paths:

- **Up to 16,384 valid tokens:** one Triton program computes entropy, raw
  weights, the global weight reduction, normalization, and weighted NLL.
- **More than 16,384 valid tokens:** a bounded-block kernel writes raw weights
  and contributes one atomic partial sum per program; a second kernel normalizes
  the weights in place and multiplies NLL.

The valid-token count is a runtime value, so different padding counts do not
compile one kernel per exact token count. Only the power-of-two block shape and
the specialized gamma mode affect compilation. `gamma=0` and `gamma=1` avoid a
general power operation; other values use `exp(gamma * log(1 + H))`.

## Memory contract

Besides the state already required by CCE, MiLe retains compact FP32 vectors:

```text
mean_logit:  O(valid tokens)
mile_weight: O(valid tokens), saved for backward
token_loss:  O(valid tokens), transient until reduction
```

The multi-block path also uses one FP32 scalar for the raw-weight sum. It does
not allocate `O(tokens x vocabulary)` storage. Setting `mile_enabled=False`
keeps the ordinary CCE path and does not request the MiLe mean-logit statistic.

## Usage

```python
from cut_cross_entropy import linear_cross_entropy

loss = linear_cross_entropy(
    embeddings,
    classifier,
    labels,
    shift=1,
    impl="cce_kahan_full_c",
    mile_enabled=True,
    mile_gamma=1.0,
)
```

MiLe can be combined with mu loss by setting both explicit flags. Padding and
`ignore_index` entries do not participate in the weight mean. MiLe is supported
by CCE implementations and is intentionally rejected by `impl="torch_compile"`.

## Precision and gradient filtering

When MiLe requests the probability-weighted logit moment, the forward logit dot
products use IEEE precision. The backward uses aligned precision so that the
reconstructed probabilities correspond to the forward LSE. For pretraining,
`cce_kahan_full_c` keeps the full classifier gradient while retaining CCE's
embedding-gradient filtering; `cce_exact` is the audit path when exact gradient
comparison is required.

## Source and validation

- Weight kernel: [`cut_cross_entropy/cce_mile.py`](../cut_cross_entropy/cce_mile.py)
- CCE integration: [`cut_cross_entropy/cce.py`](../cut_cross_entropy/cce.py)
- Weighted backward: [`cut_cross_entropy/cce_backward.py`](../cut_cross_entropy/cce_backward.py)
- Dense and kernel equivalence tests: [`tests/test_cce_mile.py`](../tests/test_cce_mile.py)

The tests cover gamma values `0`, `0.5`, `1`, and `2`; both kernel-size paths;
bias, soft-capping, shift, padding/ignore entries, reductions, returned LSE, and
forward/backward agreement with a dense formulation.
