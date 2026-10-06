"""Summarize Leviathan kernel events from a PyTorch Chrome trace.

This is a diagnostic-only analyzer.  By default it reads an existing trace
and emits only aggregate timings for the investigated kernel; it does not
print trace arguments, process metadata, paths, or connection information.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

DEFAULT_TARGETS = ("_lev_bwd_ddelta_dot_kernel",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--kernel-trace", type=Path,
                        help="Export a minimal Chrome trace of the selected GPU kernels only")
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="kernel event name to summarize; may be repeated",
    )
    return parser.parse_args()


def _summary(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    selected = [event for event in events
                if event.get("name") == name and event.get("cat") == "kernel"
                and event.get("ph") == "X"
                and isinstance(event.get("dur"), (int, float))
                and float(event["dur"]) >= 0.0]
    durations = [
        float(event["dur"])
        for event in selected
    ]
    if not durations:
        return {"count": 0, "dur_us": []}
    return {
        "count": len(durations),
        "dur_us": durations,
        "median_us": statistics.median(durations),
        "total_us": sum(durations),
        "max_us": max(durations),
        "resources": [
            dict(zip(("registers_per_thread", "shared_memory_bytes"), resources))
            for resources in sorted({
                (int(event["args"]["registers per thread"]),
                 int(event["args"]["shared memory"]))
                for event in selected
                if isinstance(event.get("args"), dict)
                and all(isinstance(event["args"].get(key), (int, float))
                        for key in ("registers per thread", "shared memory"))
            })
        ],
    }


def analyze(path: Path, targets: tuple[str, ...] = DEFAULT_TARGETS) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("trace does not contain a traceEvents list")
    typed_events = [event for event in events if isinstance(event, dict)]
    return {
        "schema": "leviathan-kernel-trace-summary-v2",
        "targets": {name: _summary(typed_events, name) for name in targets},
    }


def kernel_trace(path: Path, targets: tuple[str, ...]) -> dict[str, Any]:
    """Keep GPU timing/resources, never source paths or unrelated trace metadata.

    Stream identifiers are remapped, and time starts at the first selected
    event. Gaps represent work outside this selected scope, not GPU idleness.
    """
    if not targets or any(not name.startswith("_lev_") for name in targets):
        raise ValueError("kernel-only export is restricted to investigated Leviathan kernels")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = [event for event in payload["traceEvents"]
                if isinstance(event, dict) and event.get("name") in targets
                and event.get("cat") == "kernel" and event.get("ph") == "X"
                and isinstance(event.get("ts"), (int, float))
                and isinstance(event.get("dur"), (int, float))]
    start = min((event["ts"] for event in selected), default=0)
    streams: dict[tuple, int] = {}
    events = []
    for event in selected:
        stream = (event.get("pid"), event.get("tid"))
        stream_number = streams.setdefault(stream, len(streams) + 1)
        args = event.get("args", {})
        events.append({"name": event["name"], "cat": "kernel", "ph": "X",
                       "ts": event["ts"] - start, "dur": event["dur"],
                       "pid": 1, "tid": stream_number,
                       "args": {key: args[key] for key in ("registers per thread", "shared memory")
                                if isinstance(args.get(key), (int, float))}})
    return {"displayTimeUnit": "ms", "traceEvents": events,
            "publication_scope": "investigated_kernel_timing_and_resources_only",
            "gaps_are_not_gpu_idle": True}


def main() -> int:
    args = _parse_args()
    targets = tuple(args.targets) if args.targets else DEFAULT_TARGETS
    if args.kernel_trace is not None:
        args.kernel_trace.parent.mkdir(parents=True, exist_ok=True)
        args.kernel_trace.write_text(json.dumps(kernel_trace(args.trace, targets), indent=2)
                                     + "\n", encoding="utf-8")
    print(json.dumps(analyze(args.trace, targets), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
