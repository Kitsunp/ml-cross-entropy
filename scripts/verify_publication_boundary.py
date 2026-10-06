"""Fail closed before publishing private training sources or profile traces.

The remote NeoLLM files are used to run experiments but are not part of this
repository. Their SHA-256 fingerprints belong only in ignored private records,
never in publication manifests. This guard checks the staged index (or the complete tracked tree)
without printing file contents, credentials, paths, or connection details.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Optional

PROTECTED_SOURCE_NAMES = (
    "train.py",
    "modeling*.py",
    "configuration*.py",
)
# This pre-existing public utility is unrelated to the private experiment
# sources. No other protected filename becomes publishable just by committing it.
LEGACY_PUBLIC_SOURCE_PATHS = frozenset(("training/train.py",))
RAW_PROFILE_SUFFIXES = (
    ".trace.json",
    ".trace.json.gz",
    ".nsys-rep",
    ".qdrep",
    ".prof",
)
PRIVATE_ARTIFACT_PARTS = frozenset((".codex-tmp", "wandb", "runs"))
PUBLICATION_PREFIXES = ("docs/", "benchmark-results/")

_CONNECTION_PATTERNS = (
    re.compile(r"(?i)\b(?:ssh|scp)\s+-p\s+\d+\b"),
    re.compile(r"(?i)\b(?:root|ubuntu|admin)@[a-z0-9.-]+\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"(?i)BEGIN\s+(?:OPENSSH|RSA|EC|DSA)\s+PRIVATE\s+KEY"),
    re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+[^\s\\/]+"),
    re.compile(r"(?i)/(?:home|root)/[^\s/]+"),
    re.compile(
        r"(?i)\b(?:password|passwd|token|secret|api[_-]?key|private[_-]?key)"
        r"[ \t]*[:=][ \t]*[^\s,;]+"
    ),
    # A prose line ending in "token:" is not a credential assignment. Explicit
    # credential names still reject values split over multiple lines.
    re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|private[_-]?key|"
        r"(?:access|auth|hf)[_-]?token)\s*[:=]\s*[^\s,;]+"
    ),
)
_PRIVATE_SOURCE_FINGERPRINT = re.compile(
    r"(?i)\b(?:train(?:_[\w-]+)?|modeling[\w-]*|configuration[\w-]*)\.py"
    r"[^\n]*\b[0-9a-f]{64}\b"
)


def _normalise(path: str) -> str:
    normalised = path.replace("\\", "/")
    while normalised.startswith("./"):
        normalised = normalised[2:]
    return normalised


def _is_protected_source(path: str) -> bool:
    name = PurePosixPath(_normalise(path)).name.lower()
    return name.endswith(".py") and (
        name == "train.py"
        or name.startswith("train_")
        or name.startswith("modeling")
        or name.startswith("configuration")
    )


def _is_raw_profile(path: str) -> bool:
    normalised = _normalise(path).lower()
    return normalised.endswith(RAW_PROFILE_SUFFIXES) or any(
        part in PRIVATE_ARTIFACT_PARTS for part in PurePosixPath(normalised).parts
    )


def _is_publication_artifact(path: str) -> bool:
    normalised = _normalise(path).lower()
    return normalised.startswith(PUBLICATION_PREFIXES)


def _git_output(arguments: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _git_paths(mode: str) -> list[str]:
    if mode == "staged":
        output = _git_output(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"])
    else:
        output = _git_output(["ls-files", "-z"])
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def _staged_content(path: str) -> bytes:
    return _git_output(["show", f":{_normalise(path)}"])


def _tree_content(path: str) -> bytes:
    return Path(_normalise(path)).read_bytes()


def _head_content(path: str) -> Optional[bytes]:
    completed = subprocess.run(
        ["git", "show", f"HEAD:{_normalise(path)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout if completed.returncode == 0 else None


def _json_policy_errors(path: str, content: bytes) -> list[str]:
    normalised = _normalise(path).lower()
    if not _is_publication_artifact(normalised) or not normalised.endswith(".json"):
        return []
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [f"{path}: invalid JSON publication artifact"]
    if not isinstance(payload, dict):
        return [f"{path}: publication manifest must be a JSON object"]

    errors: list[str] = []
    def contains_raw_trace(value: object) -> bool:
        if isinstance(value, dict):
            return any(str(key) in {"traceEvents", "stackFrames"}
                       or contains_raw_trace(child) for key, child in value.items())
        if isinstance(value, list):
            return any(contains_raw_trace(child) for child in value)
        return False

    if contains_raw_trace(payload):
        errors.append(f"{path}: raw profile payloads are prohibited from publication")

    def contains_private_hash(value: object, private_source: bool = False) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key) == "module_hashes" and isinstance(child, dict) and any(
                    _is_protected_source(str(name)) for name in child
                ):
                    return True
                protected = private_source or _is_protected_source(str(key))
                if protected and str(key).lower() in {"sha256", "sha", "hash", "digest"}:
                    return True
                if contains_private_hash(child, protected):
                    return True
        elif isinstance(value, list):
            return any(contains_private_hash(child, private_source) for child in value)
        elif private_source and isinstance(value, str):
            return re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
        return False

    if contains_private_hash(payload):
        errors.append(f"{path}: private source fingerprints are prohibited from publication")

    if "trace_summary" in payload:
        profile_policy = payload.get("profile_publication_policy")
        if not isinstance(profile_policy, dict):
            errors.append(f"{path}: trace summary requires profile_publication_policy")
        elif profile_policy.get("raw_trace_included") is not False:
            errors.append(f"{path}: raw trace publication is prohibited")
        scope = str(profile_policy.get("published_scope", "")) if isinstance(profile_policy, dict) else ""
        if "investigated_kernel" not in scope:
            errors.append(f"{path}: trace summary scope must name the investigated kernel")
    return errors


def _content_policy_errors(path: str, content: bytes) -> list[str]:
    errors: list[str] = []
    if _is_publication_artifact(path):
        text = content.decode("utf-8", errors="replace")
        for pattern in _CONNECTION_PATTERNS:
            if pattern.search(text):
                errors.append(f"{path}: connection or secret pattern detected")
                break
        if not path.lower().endswith(".json") and _PRIVATE_SOURCE_FINGERPRINT.search(text):
            errors.append(f"{path}: private source fingerprints are prohibited from publication")
    errors.extend(_json_policy_errors(path, content))
    return errors


def validate_paths(
    paths: Iterable[str],
    content_loader: Callable[[str], bytes],
    baseline_loader: Callable[[str], Optional[bytes]],
) -> list[str]:
    """Validate paths and contents; returns safe, content-free diagnostics."""

    errors: list[str] = []
    for raw_path in paths:
        path = _normalise(raw_path)
        if _is_raw_profile(path):
            errors.append(f"{path}: raw profile artifacts are not publishable")
            continue

        content = content_loader(path)
        baseline = baseline_loader(path)
        # Git commonly checks out LF Python sources as CRLF on Windows. This
        # does not authorize any source edit beyond newline representation.
        unchanged_source = baseline is not None and (
            content.replace(b"\r\n", b"\n") == baseline.replace(b"\r\n", b"\n")
        )
        if _is_protected_source(path) and (
            path not in LEGACY_PUBLIC_SOURCE_PATHS or not unchanged_source
        ):
            errors.append(f"{path}: protected training source must not be added or modified")
        errors.extend(_content_policy_errors(path, content))
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true", help="validate the staged index")
    group.add_argument("--tree", action="store_true", help="validate all tracked files")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mode = "staged" if args.staged else "tree"
    try:
        paths = _git_paths(mode)
        if mode == "staged":
            loader = _staged_content
        else:
            loader = _tree_content
        errors = validate_paths(paths, loader, _head_content)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"publication boundary check failed to inspect git state: {type(error).__name__}")
        return 2

    if errors:
        print("Publication boundary violations:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Publication boundary OK ({mode}; {len(paths)} tracked paths checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
