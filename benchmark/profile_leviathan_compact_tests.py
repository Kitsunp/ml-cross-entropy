"""Run focused local compact-spline tests with a private userspace profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEST_FILES = (
    "test_leviathan_spline_support.py",
    "test_leviathan_runtime_policy.py",
    "test_leviathan_compact_kernels.py",
    "test_leviathan_candidate_provenance.py",
    "test_remote_real_10x10_validation.py",
    "test_publication_boundary.py",
    "test_leviathan_kernel_trace.py",
    "test_leviathan_kernel_split.py",
    "test_jtok_sparse_updates.py",
    "test_jtok_projection.py",
    "test_jtok_compact.py",
    "test_jtok_route_vjp.py",
    "test_leviathan_jtok.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--last-failed", action="store_true")
    parser.add_argument("--test-name", help="Select only a new/changed test expression")
    parser.add_argument("--test-file", choices=TEST_FILES, action="append")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Profile CPU-only contract tests without starting CUDA profiling")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import pytest
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if not args.cpu_only and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    profiler = torch.profiler.profile(activities=activities, record_shapes=False)
    try:
        with profiler:
            arguments = [
                "-q", "-k", args.test_name or "compact or candidate or critical_cce_hashes",
                *(str(root / "tests" / name) for name in (args.test_file or TEST_FILES[:5])),
            ]
            if args.last_failed:
                arguments.append("--lf")
            return int(pytest.main(arguments))
    finally:
        profiler.export_chrome_trace(str(args.profile))


if __name__ == "__main__":
    raise SystemExit(main())
