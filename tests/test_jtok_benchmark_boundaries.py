"""Static guardrails for the model-free JTok integration probes."""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_FREE_RUNNER = REPOSITORY_ROOT / "benchmark" / "leviathan_jtok_integration.py"
MODEL_RUNNER = REPOSITORY_ROOT / "benchmark" / "neo_llm_jtok.py"


def _imported_module_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def test_model_free_runner_does_not_import_downstream_model_sources():
    tree = ast.parse(MODEL_FREE_RUNNER.read_text(encoding="utf-8"))
    imported = _imported_module_names(tree)

    assert "modeling_neollm" not in imported
    assert "configuration_neollm" not in imported
    assert "transformers" not in imported


def test_cudagraph_disable_switch_is_scoped_to_benchmark_runner():
    source = MODEL_FREE_RUNNER.read_text(encoding="utf-8")
    assert "--disable-cudagraphs" in source
    assert '"triton.cudagraphs": False' in source

    core_sources = list((REPOSITORY_ROOT / "cut_cross_entropy").rglob("*.py"))
    assert all(
        "triton.cudagraphs" not in path.read_text(encoding="utf-8")
        for path in core_sources
    )


def test_model_runner_snapshots_outputs_before_graph_replay():
    source = MODEL_RUNNER.read_text(encoding="utf-8")

    assert "def _snapshot_scalar" in source
    assert "return _snapshot_scalar(loss)" in source
    assert "_snapshot_scalar(callable_model(input_ids, attention_mask, labels))" in source
    assert "_snapshot_scalar(loss)," in source
