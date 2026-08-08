"""Small language-model consumer for the integrated Leviathan layer.

The model deliberately keeps the integration surface narrow:

* ``LeviathanEmbedding`` is the only LEV entry point and calls the existing
  autograd wrapper, so CUDA uses the Triton forward/backward when the config is
  supported and the verified reference path remains available as a fallback.
* The classifier is consumed by :func:`linear_cross_entropy`, which avoids
  allocating a ``[batch, sequence, vocab]`` logits tensor.
* ``shift=1`` lets the CCE implementation use its compiler-safe custom-op
  boundary when the surrounding model is compiled with ``torch.compile``.

This is intentionally a compact integration model rather than a full
transformer.  It is useful for correctness, graph, memory, and end-to-end
kernel tests, and can be inserted as the embedding/loss portion of a larger
causal LM.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from cut_cross_entropy.linear_cross_entropy import linear_cross_entropy

from .compiler import leviathan_embedding_compiler_safe
from .core import LeviathanConfig, LeviathanGenerator


class LeviathanEmbedding(nn.Module):
    """Embedding module backed by the integrated LEV kernel entry point."""

    def __init__(
        self,
        config: LeviathanConfig,
        *,
        use_reference: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.generator = LeviathanGenerator(config)
        self.use_reference = use_reference

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.use_reference:
            return self.generator(input_ids)

        params: Mapping[str, torch.Tensor] = {
            "codebooks": self.generator.codebooks,
            "head_proj_weight": self.generator.head_proj_weight,
            "head_norm_weight": self.generator.head_norm_weight,
            "head_norm_bias": self.generator.head_norm_bias,
            "head_spline_delta": self.generator.head_spline_delta,
            "head_out_weight": self.generator.head_out_weight,
        }
        return leviathan_embedding_compiler_safe(
            input_ids,
            dict(params),
            self.config,
            self.generator.knot_grid,
        )


class LeviathanForCausalLM(nn.Module):
    """Compact causal-LM head using LEV embeddings and CCE loss.

    When ``labels`` is supplied, the return value is a scalar loss.  With no
    labels, the method returns the hidden states so callers can attach their
    own transformer blocks or inspect the LEV output.
    """

    def __init__(
        self,
        config: LeviathanConfig | None = None,
        *,
        loss_impl: str = "cce",
        use_reference_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or LeviathanConfig()
        self.loss_impl = loss_impl
        self.embed = LeviathanEmbedding(
            self.config,
            use_reference=use_reference_embedding,
        )
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.lm_head = nn.Parameter(
            torch.empty(self.config.vocab_size, self.config.hidden_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.lm_head,
            mean=0.0,
            std=self.config.initializer_range,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.norm(self.embed(input_ids))
        if labels is None:
            return hidden

        return linear_cross_entropy(
            hidden,
            self.lm_head,
            labels,
            reduction="mean",
            shift=1,
            impl=self.loss_impl,
        )


__all__ = ["LeviathanEmbedding", "LeviathanForCausalLM"]
