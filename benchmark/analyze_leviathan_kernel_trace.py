"""Summarize Leviathan kernel events from a PyTorch Chrome trace.

This is a diagnostic-only analyzer.  It reads an existing trace and emits
only aggregate event names and timings; it does not print trace arguments,
process metadata, paths, or connection information.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

_TARGETS = (
    "_lev_fused_dot",
    "_lev_bwd_ddelta_dot_kernel",
    "_lev_bwd_ln_kernel",
    "_lev_bwd_stats_kernel",
    "cut_cross_entropy::leviathan_forward",
    "cut_cross_entropy::leviathan_backward",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    return parser.parse_args()


def _summary(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    durations = [
        float(event["dur"])
        for event in events
        if event.get("name") == name
        and event.get("ph") == "X"
        and isinstance(event.get("dur"), (int, float))
        and float(event["dur"]) >= 0.0
    ]
    if not durations:
        return {"count": 0, "dur_us": []}
    return {
        "count": len(durations),
        "dur_us": durations,
        "median_us": statistics.median(durations),
        "total_us": sum(durations),
        "max_us": max(durations),
    }


def analyze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("trace does not contain a traceEvents list")
    typed_events = [event for event in events if isinstance(event, dict)]
    return {
        "schema": "leviathan-kernel-trace-summary-v1",
        "trace_event_count": len(typed_events),
        "targets": {name: _summary(typed_events, name) for name in _TARGETS},
    }


def main() -> int:
    args = _parse_args()
    print(json.dumps(analyze(args.trace), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
