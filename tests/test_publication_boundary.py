"""Regression tests for the public/private publication boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_SCRIPT = _ROOT / "scripts" / "verify_publication_boundary.py"
_SPEC = importlib.util.spec_from_file_location("publication_boundary", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_GUARD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GUARD)


def _validate(path: str, content: bytes, baseline: bytes | None = None) -> list[str]:
    return _GUARD.validate_paths(
        [path],
        content_loader=lambda _path: content,
        baseline_loader=lambda _path: baseline,
    )


def test_remote_training_sources_are_rejected() -> None:
    assert _validate("modeling_neollm.py", b"class Model: pass\n")
    assert _validate("configuration_neollm.py", b"class Config: pass\n")
    assert _validate("train.py", b"def main(): pass\n")


def test_unchanged_existing_training_utility_is_allowed() -> None:
    source = b"existing public utility\n"
    assert _validate("training/train.py", source, baseline=source) == []


def test_protected_source_allows_only_newline_representation_change() -> None:
    source = b"existing public utility\nsecond line\n"
    assert _validate("training/train.py", source.replace(b"\n", b"\r\n"), source) == []
    assert _validate("training/train.py", b"changed source\r\n", source)


def test_committed_private_source_is_still_rejected_from_tracked_tree() -> None:
    source = b"private experiment source\n"
    for path in ("train.py", "modeling_neollm.py", "nested/configuration_neollm.py"):
        assert _validate(path, source, baseline=source)


def test_token_prose_is_not_a_multiline_credential_assignment() -> None:
    assert _validate("docs/math.md", b"Three values per valid token:\n  $LSE_n$.\n") == []
    for assignment in (b"token = forbidden", b"access_token:\n forbidden",
                       b"password:\n forbidden", b"api_key = forbidden"):
        assert _validate("docs/math.md", assignment)


def test_raw_profile_artifact_is_rejected() -> None:
    assert _validate("benchmark-results/private.trace.json", b"{}")


def test_raw_profile_payload_cannot_hide_behind_a_manifest_name() -> None:
    for payload in ({"traceEvents": []}, {"nested": [{"stackFrames": {}}]}):
        assert _validate("benchmark-results/run.json", json.dumps(payload).encode())


def test_personal_paths_are_rejected_from_publication_documents() -> None:
    for content in (b"C:/Users/example/run.json", b"/home/example/run.json"):
        assert _validate("docs/results.md", content)


def test_private_dot_directory_is_rejected_without_reading_its_content() -> None:
    def forbidden(_path):
        raise AssertionError("private content must not be read during publication")
    errors = _GUARD.validate_paths(["./.codex-tmp/manifest.json"], forbidden, forbidden)
    assert any("not publishable" in error for error in errors)


def test_normalisation_preserves_dot_file_and_directory_names() -> None:
    assert _GUARD._normalise("./.gitignore") == ".gitignore"
    assert _GUARD._normalise(".github\\workflows\\check.yml") == ".github/workflows/check.yml"


def test_manifest_rejects_private_hashes_without_policy() -> None:
    manifest = b'{"module_hashes":{"train.py":"abc"}}'
    errors = _validate("benchmark-results/run.json", manifest)
    assert any("private source fingerprints" in error for error in errors)


def test_private_hashes_are_rejected_even_with_legacy_exclusion_policy() -> None:
    manifest = {
        "module_hashes": {"train.py": "a" * 64},
        "source_code_upload_policy": {
            "train.py": "prohibited_from_repository_sha256_only",
            "source_contents_included": False,
        },
    }
    assert _validate("benchmark-results/run.json", json.dumps(manifest).encode())


def test_nested_private_fingerprints_are_rejected_in_docs_json() -> None:
    manifest = {"runs": [{"modeling_neollm.py": {"sha256": "a" * 64}}]}
    assert _validate("docs/run.json", json.dumps(manifest).encode())


def test_private_fingerprints_are_rejected_in_markdown() -> None:
    content = f'| `configuration_neollm.py` | `{ "a" * 64 }` |\n'.encode()
    assert _validate("docs/results.md", content)


def test_public_kernel_fingerprints_remain_allowed() -> None:
    manifest = {"files": {"leviathan/backward_dot_kernels.py": "a" * 64}}
    assert _validate("benchmark-results/candidate.json", json.dumps(manifest).encode()) == []


def test_trace_policy_missing_is_reported_without_crashing() -> None:
    assert _validate("benchmark-results/run.json", b'{"trace_summary":{}}')


def test_sanitized_kernel_summary_is_allowed() -> None:
    manifest = (
        b'{"source_code_upload_policy":{"source_contents_included":false},'
        b'"profile_publication_policy":{"raw_trace_included":false,'
        b'"published_scope":"investigated_kernel_aggregate_and_full_step_time_only"},'
        b'"trace_summary":{"scope":"_lev_bwd_ddelta_dot_kernel"}}'
    )
    assert _validate("benchmark-results/run.json", manifest) == []
