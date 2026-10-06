"""Check public kernel sources before accepting a candidate execution.

The manifest includes only this repository's Leviathan implementation. Private
training sources, dataset information, machine paths and connection details
are never part of this format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CANDIDATE_FILES = (
    "leviathan/autograd_fn.py",
    "leviathan/backward_dot_kernels.py",
    "leviathan/backward_kernels.py",
    "leviathan/compiler.py",
    "leviathan/forward_impl.py",
    "leviathan/runtime_policy.py",
    "leviathan/spline_support.py",
    "leviathan/spline_support_kernels.py",
)
SCHEMA = "leviathan-public-candidate-v1"


def source_manifest(package_root: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "files": {
            name: hashlib.sha256((package_root / name).read_bytes()).hexdigest()
            for name in CANDIDATE_FILES
        },
    }


def verify_candidate(package_root: Path, expected: dict[str, Any]) -> None:
    if set(expected) != {"schema", "files"} or expected["schema"] != SCHEMA:
        raise ValueError("invalid public Leviathan candidate manifest")
    files = expected["files"]
    if not isinstance(files, dict) or set(files) != set(CANDIDATE_FILES):
        raise ValueError("candidate manifest must contain exactly the public kernel files")
    for name, value in files.items():
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("invalid candidate source digest")
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError("invalid candidate source digest")
        path = package_root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != value:
            raise RuntimeError(f"loaded candidate source mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = source_manifest(args.source_root / "cut_cross_entropy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
