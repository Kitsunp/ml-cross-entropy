"""Fused Triton implementation of REPO-GRAPE positional attention."""

from .triton import repo_grape, repo_grape_supported

__all__ = ["repo_grape", "repo_grape_supported"]
