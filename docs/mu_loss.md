# Mu Loss in Cut Cross Entropy

[Output Embedding Centering](https://arxiv.org/abs/2601.02031) identifies the
common component of output embeddings as a source of logit growth. Mu loss adds
a differentiable penalty on that common component without explicitly replacing
or centering the classifier during the forward pass.

![Mu-loss kernel data flow](../assets/mu_loss_kernel.svg)

## How to read the diagram

`C` is the output classifier, `V` is its number of vocabulary rows, and `D` is
the embedding width. `C_v` denotes one vocabulary row, not a token activation.

- **Panel (a)** streams `C` in `BLOCK_V x BLOCK_D` tiles and reduces over the
  vocabulary axis in FP32. Its output is one local vector of length `D`, not a
  second `V x D` tensor.
- **Panel (b)** synchronizes that vector and the row count only when vocabulary
  parallelism is active. The finalize kernel divides by global `V`, saves `mu`,
  and returns the scalar penalty.
- **Panel (c)** receives `mu` during backward, constructs one length-`D` vector
  `r`, and adds the same `r` to every row of the already allocated `dC` tensor.
  The blue rectangle illustrates an in-place tile update; it is not an extra
  dense gradient allocation.

The dashed box is therefore conditional distributed communication, whereas the
solid dark boxes are Triton computation. With an unsharded vocabulary, the
forward goes directly from the local reduction to finalization.

## Objective

For output classifier `C` with vocabulary size `V` and embedding width `D`:

```text
mu = mean(C, dim=0) = (1 / V) * sum_v C_v
L_mu = lambda * ||mu||_2 ** 2
```

The derivative is identical for every vocabulary row:

```text
dL_mu/dC_v = (2 * lambda / V) * mu
```

This is the classifier's common-row direction. It is complementary to ordinary
cross-entropy and detached MiLe gradients, which act primarily through
differences between vocabulary rows.

Geometrically, each classifier row can be decomposed as:

```text
C_v = mu + delta_v,    with sum_v delta_v = 0
```

The direct mu-loss gradient acts only on the shared `mu` component. It does not
directly change the centered row differences `delta_v`, which continue to learn
from CCE or MiLe. Repeated optimizer steps tend to shrink the common component,
but other loss gradients can rebuild it, so the result is a soft correction
rather than an exact projection onto `mu=0`.

## Triton forward

The forward uses two Triton kernels:

1. `_output_embedding_sum_kernel` traverses classifier rows in vocabulary tiles
   and accumulates an FP32 sum for every embedding dimension.
2. `_mu_loss_finalize_kernel` divides by the global vocabulary size, stores the
   compact vector `mu`, and produces the scalar `lambda * sum(mu ** 2)`.

With vocabulary parallelism, each rank first computes its local row sum. The
local sums and vocabulary sizes are then combined with distributed
`all_reduce` operations before the finalize kernel. Every rank therefore uses
the same global `mu` and global `V`.

The scalar mu loss is added to the mean-reduced CCE or MiLe loss:

```text
L_total = L_CCE_or_MiLe + L_mu
```

Mu loss currently requires `reduction="mean"`; this keeps its scale independent
of the number of valid training tokens.

## Triton backward

The CCE backward first computes the ordinary classifier gradient `dC`. The
`_add_mu_loss_gradient_kernel` then visits `dC` in `32 x 32` tiles and adds:

```text
dOut * (2 * lambda / V) * mu
```

to every vocabulary row in place. There is no separate dense gradient tensor.
The direct mu-loss term does not modify the hidden-state gradient `dE` or the
bias gradient because its objective depends only on `C`.

When input and output embeddings are tied, the shared parameter can still
receive other gradients through its input-embedding role. The statement above
describes only the direct derivative of `L_mu`.

## Memory contract

Mu loss does not materialize logits. Its persistent and temporary state is:

```text
embedding_sum: O(D), FP32
mu:            O(D), FP32 and saved for backward
loss:          one FP32 scalar
vocab_size:    one FP32 scalar
```

The backward adds directly into the existing `dC` allocation. Compute scales as
`O(V * D)`, which is the minimum work required to read the classifier and apply
the row-wise regularizer, while auxiliary memory remains `O(D)`.

## Mu loss is not hard centering

Mu loss penalizes the center but does not execute:

```python
classifier = classifier - classifier.mean(dim=0)
```

Consequently, `mu` approaches zero according to the optimizer, learning rate,
coefficient, and training duration; it is not forced to zero after every step.
This also means that a small coefficient such as `1e-4` can act slowly in short
training runs.

## Usage

```python
from cut_cross_entropy import linear_cross_entropy

loss = linear_cross_entropy(
    embeddings,
    classifier,
    labels,
    shift=1,
    impl="cce_kahan_full_c",
    mu_loss_enabled=True,
    mu_loss_lambda=1e-4,
)
```

Combined with MiLe:

```python
loss = linear_cross_entropy(
    embeddings,
    classifier,
    labels,
    shift=1,
    impl="cce_kahan_full_c",
    mile_enabled=True,
    mile_gamma=1.0,
    mu_loss_enabled=True,
    mu_loss_lambda=1e-4,
)
```

Mu loss is supported by CCE implementations and is intentionally rejected by
`impl="torch_compile"`. Setting `mu_loss_enabled=False` keeps the ordinary CCE
path and does not allocate the mean-embedding state.

## Source and validation

- Kernels: [`cut_cross_entropy/mu_loss.py`](../cut_cross_entropy/mu_loss.py)
- CCE integration: [`cut_cross_entropy/cce.py`](../cut_cross_entropy/cce.py)
- Tests: [`tests/test_cce_mu_loss.py`](../tests/test_cce_mu_loss.py)

The tests compare loss and gradients with a dense FP32 reference, cover MiLe
composition and vocabulary parallelism, verify the explicit disabled path, and
check that invalid coefficients and unsupported reductions are rejected.
