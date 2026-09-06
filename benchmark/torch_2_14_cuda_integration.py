#!/usr/bin/env python3
"""A/B benchmark for the optional PyTorch 2.14 CUDA integration hooks."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--memory-limit-gib", type=float)
    parser.add_argument("--cce-rows", type=int, default=1024)
    parser.add_argument("--cce-hidden", type=int, default=512)
    parser.add_argument("--cce-vocab", type=int, default=65536)
    parser.add_argument("--leviathan-tokens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--polynorm-batch", type=int, default=4)
    parser.add_argument("--polynorm-sequence", type=int, default=2048)
    parser.add_argument("--polynorm-hidden", type=int, default=1024)
    args = parser.parse_args()
    if args.trials < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("trials and iterations must be positive; warmup must be non-negative")
    return args


def _run(command: list[str], *, enabled: bool, seed: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONHASHSEED"] = str(seed)
    env["CUT_CROSS_ENTROPY_TORCH_2_14_INTEGRATION"] = "1" if enabled else "0"
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    process_wall_seconds = time.perf_counter() - started
    payload = completed.stdout[completed.stdout.find("{") :]
    result = json.loads(payload)
    result["process_wall_seconds"] = process_wall_seconds
    return result


def _commands(args: argparse.Namespace) -> dict[str, list[str]]:
    memory = (
        ["--memory-limit-gib", str(args.memory_limit_gib)]
        if args.memory_limit_gib is not None
        else []
    )
    return {
        "cce": [
            sys.executable,
            "benchmark/cce_compile_extreme_profile.py",
            "--rows", str(args.cce_rows),
            "--dim", str(args.cce_hidden),
            "--vocab", str(args.cce_vocab),
            "--warmup", str(args.warmup),
            "--steps", str(args.iterations),
            "--seed", str(args.seed),
            *memory,
        ],
        "polynorm": [
            sys.executable,
            "benchmark/polynorm_profile.py",
            "--backend", "cute",
            "--compile-cute",
            "--batch", str(args.polynorm_batch),
            "--sequence", str(args.polynorm_sequence),
            "--hidden", str(args.polynorm_hidden),
            "--dtype", "bfloat16",
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--seed", str(args.seed),
            *memory,
        ],
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for component in ("polynorm",):
        component_summary: dict[str, Any] = {}
        for metric in ("forward_ms", "backward_ms", "total_ms"):
            by_state = {
                state: _median(
                    [
                        record["result"][metric]["median"]
                        for record in records
                        if record["component"] == component and record["state"] == state
                    ]
                )
                for state in ("off", "on")
            }
            component_summary[metric] = {
                **by_state,
                "delta_percent": (by_state["on"] / by_state["off"] - 1.0) * 100.0,
            }
        summary[component] = component_summary

    cce_by_state = {
        state: _median(
            [
                record["result"]["latency_ms_median"]
                for record in records
                if record["component"] == "cce" and record["state"] == state
            ]
        )
        for state in ("off", "on")
    }
    summary["cce"] = {
        "training_step_ms": {
            **cce_by_state,
            "delta_percent": (cce_by_state["on"] / cce_by_state["off"] - 1.0) * 100.0,
        }
    }
    for metric in (
        "incremental_peak_bytes",
        "peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
        "peak_reserved_bytes",
    ):
        summary["cce"][metric] = {
            state: int(
                _median(
                    [
                        record["result"][metric]
                        for record in records
                        if record["component"] == "cce" and record["state"] == state
                    ]
                )
            )
            for state in ("off", "on")
        }

    summary["polynorm"]["memory"] = {}
    for metric in (
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ):
        summary["polynorm"]["memory"][metric] = {
            state: int(
                _median(
                    [
                        record["result"]["memory"][metric]
                        for record in records
                        if record["component"] == "polynorm" and record["state"] == state
                    ]
                )
            )
            for state in ("off", "on")
        }
    for component in ("cce", "polynorm"):
        summary[component]["process_wall_seconds"] = {
            state: _median(
                [
                    record["result"]["process_wall_seconds"]
                    for record in records
                    if record["component"] == component and record["state"] == state
                ]
            )
            for state in ("off", "on")
        }

    leviathan: dict[str, Any] = {}
    token_counts = sorted(
        {
            case["tokens"]
            for record in records
            if record["component"] == "leviathan"
            for case in [record["result"]]
        }
    )
    for tokens in token_counts:
        variants: dict[str, Any] = {}
        for variant in ("forward_fused_ms", "training_fused_ms"):
            by_state = {}
            for state in ("off", "on"):
                values = []
                for record in records:
                    if record["component"] != "leviathan" or record["state"] != state:
                        continue
                    case = record["result"]
                    if case["tokens"] == tokens:
                        values.append(case[variant]["median"])
                by_state[state] = _median(values)
            variants[variant] = {
                **by_state,
                "delta_percent": (by_state["on"] / by_state["off"] - 1.0) * 100.0,
            }
        leviathan[str(tokens)] = variants
        for metric in ("forward_peak_bytes", "training_peak_bytes"):
            leviathan[str(tokens)][metric] = {
                state: int(
                    _median(
                        [
                            record["result"]["memory"][metric]
                            for record in records
                            if record["component"] == "leviathan"
                            and record["state"] == state
                            and record["result"]["tokens"] == tokens
                        ]
                    )
                )
                for state in ("off", "on")
            }
        leviathan[str(tokens)]["process_wall_seconds"] = {
            state: _median(
                [
                    record["result"]["process_wall_seconds"]
                    for record in records
                    if record["component"] == "leviathan"
                    and record["state"] == state
                    and record["result"]["tokens"] == tokens
                ]
            )
            for state in ("off", "on")
        }
    summary["leviathan"] = leviathan
    return summary


def main() -> None:
    args = _parse_args()
    commands = _commands(args)
    records: list[dict[str, Any]] = []
    for trial in range(args.trials):
        states = (False, True) if trial % 2 == 0 else (True, False)
        trial_commands = dict(commands)
        for tokens in args.leviathan_tokens:
            trial_commands[f"leviathan:{tokens}"] = [
                sys.executable,
                "benchmark/leviathan_meap_profile.py",
                "--tokens", str(tokens),
                "--warmup", str(args.warmup),
                "--steps", str(args.iterations),
                "--seed", str(args.seed),
                "--json",
                *(
                    ["--vram-limit-gib", str(args.memory_limit_gib)]
                    if args.memory_limit_gib is not None
                    else []
                ),
            ]
        for component_key, command in trial_commands.items():
            component = "leviathan" if component_key.startswith("leviathan:") else component_key
            for enabled in states:
                records.append(
                    {
                        "trial": trial,
                        "component": component,
                        "state": "on" if enabled else "off",
                        "result": _run(command, enabled=enabled, seed=args.seed),
                    }
                )

    report = {
        "benchmark": "torch_2_14_cuda_integration",
        "seed": args.seed,
        "trials": args.trials,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "memory_limit_gib": args.memory_limit_gib,
        "summary": _summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
