"""Regression tests for the split-N Leviathan kernel contract."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_DOT_KERNEL_PATH = _ROOT / "cut_cross_entropy" / "leviathan" / "backward_dot_kernels.py"
_TRACE_ANALYZER_PATH = _ROOT / "benchmark" / "analyze_leviathan_kernel_trace.py"
if not _DOT_KERNEL_PATH.is_file():
    _DOT_KERNEL_SPEC = importlib.util.find_spec(
        "cut_cross_entropy.leviathan.backward_dot_kernels"
    )
    assert _DOT_KERNEL_SPEC is not None and _DOT_KERNEL_SPEC.origin is not None
    _DOT_KERNEL_PATH = Path(_DOT_KERNEL_SPEC.origin)


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_split_kernel_and_deterministic_reducer_exist() -> None:
    tree = ast.parse(_DOT_KERNEL_PATH.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    kernel = functions["_lev_bwd_ddelta_dot_kernel"]
    argument_names = {argument.arg for argument in kernel.args.args}

    assert {"ddelta_partial_ptr", "NUM_SPLIT_BLOCKS", "N_SPLITS"} <= argument_names
    assert "_lev_bwd_ddelta_split_reduce_kernel" in functions


def test_trace_analyzer_ignores_trace_metadata(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("trace_analyzer", _TRACE_ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    trace = tmp_path / "trace.json"
    trace.write_text(
        '{"host_name":"must-not-be-emitted",'
        '"traceEvents":[{"name":"_lev_bwd_ddelta_dot_kernel",'
        '"ph":"X","dur":12000},{"name":"other",'
        '"ph":"X","dur":4}]}',
        encoding="utf-8",
    )

    result = module.analyze(trace)

    assert result["trace_event_count"] == 2
    assert result["targets"]["_lev_bwd_ddelta_dot_kernel"]["median_us"] == 12000
    assert "host_name" not in result
    assert "must-not-be-emitted" not in str(result)
