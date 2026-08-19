# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from cut_cross_entropy.cce_utils import LinearCrossEntropyImpl
from cut_cross_entropy.linear_cross_entropy import (
    LinearCrossEntropy,
    linear_cross_entropy,
)
from cut_cross_entropy.meap import (
    MEAPEmbeddingOverride,
    apply_meap_embedding_override,
    meap_attention_diagnostics,
    meap_mask_inputs,
)
from cut_cross_entropy.mile import linear_mile_loss
from cut_cross_entropy.patch import PatchTrainingPhase, PatchTrainingSchedule
from cut_cross_entropy.vocab_parallel import VocabParallelOptions

__all__ = [
    "LinearCrossEntropy",
    "LinearCrossEntropyImpl",
    "linear_cross_entropy",
    "linear_mile_loss",
    "MEAPEmbeddingOverride",
    "apply_meap_embedding_override",
    "meap_attention_diagnostics",
    "meap_mask_inputs",
    "PatchTrainingPhase",
    "PatchTrainingSchedule",
    "VocabParallelOptions",
]


__version__ = "25.9.3"
