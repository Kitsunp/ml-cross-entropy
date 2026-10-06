"""CPU-only regressions for reusing profiles without leaking unrelated events."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "lev_trace", Path(__file__).parents[1] / "benchmark/analyze_leviathan_kernel_trace.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TARGET = "_lev_bwd_ddelta_dot_kernel"


def test_kernel_trace_export_keeps_only_gpu_times_and_resources(tmp_path):
    gpu = {"name": TARGET, "cat": "kernel", "ph": "X", "ts": 1000, "dur": 12,
           "pid": 500, "tid": 999,
           "args": {"registers per thread": 227, "shared memory": 20480,
                    "private_source": "do-not-export", "path": "private"}}
    cpu = dict(gpu, cat="cpu_op", dur=99)
    unrelated = dict(gpu, name="unrelated_model_component", dur=77)
    trace = tmp_path / "raw.json"
    trace.write_text(json.dumps({"traceEvents": [gpu, cpu, unrelated],
                                 "private_metadata": "do-not-export"}))
    summary = module.analyze(trace, (TARGET,))
    assert summary["targets"][TARGET]["count"] == 1
    assert summary["targets"][TARGET]["median_us"] == 12
    exported = module.kernel_trace(trace, (TARGET,))
    assert len(exported["traceEvents"]) == 1
    event = exported["traceEvents"][0]
    assert event["ts"] == 0 and event["pid"] == 1 and event["tid"] == 1
    assert event["args"] == {"registers per thread": 227, "shared memory": 20480}
    assert "private" not in json.dumps(exported)
    assert exported["gaps_are_not_gpu_idle"] is True


def test_kernel_trace_export_rejects_unrelated_target(tmp_path):
    with pytest.raises(ValueError, match="investigated Leviathan"):
        module.kernel_trace(tmp_path / "unused", ("unrelated_model_component",))


def test_unknown_resources_are_not_reported_as_zero():
    event = {"name": TARGET, "cat": "kernel", "ph": "X", "dur": 12}
    assert module._summary([event], TARGET)["resources"] == []
    partial = dict(event, args={"registers per thread": 227})
    assert module._summary([partial], TARGET)["resources"] == []
    known = dict(event, args={"registers per thread": 40, "shared memory": 0})
    assert module._summary([known], TARGET)["resources"] == [
        {"registers_per_thread": 40, "shared_memory_bytes": 0}]


def test_kernel_trace_export_supports_investigated_jtok_only(tmp_path):
    target = "_jtok_backward_token_projection_grad_block_kernel"
    trace = tmp_path / "raw.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"name": target, "cat": "kernel", "ph": "X", "ts": 400, "dur": 20,
         "args": {"private_source": "do-not-export", "registers per thread": 48}},
        {"name": "outside_scope", "cat": "kernel", "ph": "X", "ts": 450, "dur": 90}]}))
    result = module.kernel_trace(trace, (target,))
    assert len(result["traceEvents"]) == 1
    assert result["traceEvents"][0]["name"] == target
    assert result["traceEvents"][0]["ts"] == 0
    assert "private_source" not in json.dumps(result)
