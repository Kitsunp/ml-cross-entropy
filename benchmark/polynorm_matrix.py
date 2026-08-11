"""Run isolated PolyNorm A/B processes across curated training geometries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    name: str
    batch: int
    sequence: int
    hidden: int


GEOMETRIES = (
    Geometry("latency", 1, 128, 512),
    Geometry("batch1", 1, 512, 1536),
    Geometry("medium", 8, 512, 1536),
    Geometry("wide", 16, 512, 2048),
    Geometry("long", 16, 2048, 1536),
    Geometry("large_batch", 64, 512, 1536),
)


def _run(
    geometry: Geometry,
    backend: str,
    compile_mode: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "benchmark.polynorm_profile",
        "--backend",
        backend,
        "--batch",
        str(geometry.batch),
        "--sequence",
        str(geometry.sequence),
        "--hidden",
        str(geometry.hidden),
        "--dtype",
        args.dtype,
        "--output-dtype",
        "input",
        "--dropout-p",
        str(args.dropout_p),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--memory-limit-gib",
        str(args.memory_limit_gib),
    ]
    if backend == "torch_compile":
        command.extend(("--compile-mode", compile_mode))
    elif backend == "cute":
        command.extend(("--compile-cute", "--compile-mode", compile_mode))
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--memory-limit-gib", type=float, default=10.0)
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    for geometry in GEOMETRIES:
        runs = {
            "cute": _run(geometry, "cute", "max-autotune", args),
            "default": _run(geometry, "torch_compile", "default", args),
            "max_autotune": _run(
                geometry, "torch_compile", "max-autotune", args
            ),
        }
        cute_ms = float(runs["cute"]["total_ms"]["median"])
        default_ms = float(runs["default"]["total_ms"]["median"])
        max_autotune_ms = float(runs["max_autotune"]["total_ms"]["median"])
        results.append(
            {
                "name": geometry.name,
                "batch": geometry.batch,
                "sequence": geometry.sequence,
                "hidden": geometry.hidden,
                "rows": geometry.batch * geometry.sequence,
                "cute_ms": cute_ms,
                "default_ms": default_ms,
                "max_autotune_ms": max_autotune_ms,
                "cute_speedup_vs_default": default_ms / cute_ms,
                "cute_speedup_vs_max_autotune": max_autotune_ms / cute_ms,
                "cute_incremental_peak_allocated_bytes": runs["cute"]["memory"][
                    "incremental_peak_allocated_bytes"
                ],
                "cute_resident_reserved_bytes": runs["cute"]["memory"][
                    "resident_reserved_after_warmup_bytes"
                ],
                "default_incremental_peak_allocated_bytes": runs["default"][
                    "memory"
                ]["incremental_peak_allocated_bytes"],
                "default_resident_reserved_bytes": runs["default"]["memory"][
                    "resident_reserved_after_warmup_bytes"
                ],
                "max_autotune_resident_reserved_bytes": runs["max_autotune"][
                    "memory"
                ]["resident_reserved_after_warmup_bytes"],
            }
        )

    print(
        json.dumps(
            {
                "dtype": args.dtype,
                "dropout_p": args.dropout_p,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "test_memory_limit_gib": args.memory_limit_gib,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
