"""Summarize physical forward counters from a patch-training NCU report."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path

TENSOR_FLOPS = "sm__ops_path_tensor_src_bf16_dst_fp32.sum"
FADD = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
FFMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
FMUL = "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
HADD = "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum"
HFMA = "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"
HMUL = "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"
DRAM_BYTES = "dram__bytes.sum"
DURATION_NS = "gpu__time_duration.sum"


def _counter(row: Mapping[str, str], name: str) -> int:
    value = row.get(name, "")
    return int(Decimal(value)) if value else 0


def summarize_rows(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    counters = {
        name: 0
        for name in (TENSOR_FLOPS, FADD, FFMA, FMUL, HADD, HFMA, HMUL, DRAM_BYTES, DURATION_NS)
    }
    kernel_count = 0
    for row in rows:
        if not row.get("ID"):
            continue
        kernel_count += 1
        for name in counters:
            counters[name] += _counter(row, name)

    fp32_flops = counters[FADD] + 2 * counters[FFMA] + counters[FMUL]
    # NVIDIA's half-precision SASS counters represent packed two-lane operations.
    fp16_flops = 2 * counters[HADD] + 4 * counters[HFMA] + 2 * counters[HMUL]
    tensor_flops = counters[TENSOR_FLOPS]
    return {
        "kernel_count": kernel_count,
        "tensor_bf16_fp32_flops": tensor_flops,
        "cuda_core_fp32_flops": fp32_flops,
        "cuda_core_fp16_flops": fp16_flops,
        "measured_floating_point_ops": tensor_flops + fp32_flops + fp16_flops,
        "dram_bytes": counters[DRAM_BYTES],
        "summed_kernel_duration_ns": counters[DURATION_NS],
    }


def summarize_csv(csv_text: str) -> dict[str, int]:
    return summarize_rows(csv.DictReader(io.StringIO(csv_text)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Nsight Compute .ncu-rep file")
    parser.add_argument("--ncu", default="ncu", help="Nsight Compute CLI executable")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    completed = subprocess.run(
        [
            args.ncu,
            "--import",
            str(args.report),
            "--csv",
            "--page",
            "raw",
            "--print-units",
            "base",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(json.dumps(summarize_csv(completed.stdout), indent=2))


if __name__ == "__main__":
    main()
