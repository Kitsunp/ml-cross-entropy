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
JTOK_SOURCE = "leviathan/jtok.py"
JTOK_PROJECTION_SOURCE = "leviathan/jtok_projection.py"
JTOK_COMPACT_SOURCE = "leviathan/jtok_compact.py"


def source_manifest(
    package_root: Path, *, include_jtok: bool = False, include_projection: bool = False,
    include_compact_jtok: bool = False,
) -> dict[str, Any]:
    include_projection = include_projection or include_compact_jtok
    sources = (*CANDIDATE_FILES, JTOK_SOURCE) if include_jtok or include_projection else CANDIDATE_FILES
    if include_projection:
        sources = (*sources, JTOK_PROJECTION_SOURCE)
    if include_compact_jtok:
        sources = (*sources, JTOK_COMPACT_SOURCE)
    return {
        "schema": SCHEMA,
        "files": {
            name: hashlib.sha256((package_root / name).read_bytes()).hexdigest()
            for name in sources
        },
    }


def verify_candidate(
    package_root: Path, expected: dict[str, Any], *, require_jtok: bool = False,
    require_projection: bool = False,
    require_compact_jtok: bool = False,
) -> None:
    if set(expected) != {"schema", "files"} or expected["schema"] != SCHEMA:
        raise ValueError("invalid public Leviathan candidate manifest")
    files = expected["files"]
    permitted_sets = (set(CANDIDATE_FILES), set(CANDIDATE_FILES) | {JTOK_SOURCE},
                      set(CANDIDATE_FILES) | {JTOK_SOURCE, JTOK_PROJECTION_SOURCE},
                      set(CANDIDATE_FILES) | {JTOK_SOURCE, JTOK_PROJECTION_SOURCE, JTOK_COMPACT_SOURCE})
    if not isinstance(files, dict) or set(files) not in permitted_sets:
        raise ValueError("candidate manifest must contain exactly the public kernel files")
    if require_jtok and JTOK_SOURCE not in files:
        raise ValueError("native JTok candidate requires its public source fingerprint")
    if require_projection and JTOK_PROJECTION_SOURCE not in files:
        raise ValueError("projection candidate requires its public source fingerprint")
    if require_compact_jtok and JTOK_COMPACT_SOURCE not in files:
        raise ValueError("compact JTok candidate requires its public source fingerprint")
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
    parser.add_argument("--include-jtok", action="store_true")
    parser.add_argument("--include-projection", action="store_true")
    parser.add_argument("--include-compact-jtok", action="store_true")
    args = parser.parse_args()
    manifest = source_manifest(args.source_root / "cut_cross_entropy", include_jtok=args.include_jtok,
                               include_projection=args.include_projection,
                               include_compact_jtok=args.include_compact_jtok)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
