"""Reject stale or privacy-expanding candidate manifests before GPU work."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "benchmark/leviathan_candidate_provenance.py"
_SPEC = importlib.util.spec_from_file_location("candidate_provenance", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROVENANCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROVENANCE)


def test_candidate_provenance_rejects_stale_kernel(tmp_path: Path) -> None:
    for name in _PROVENANCE.CANDIDATE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# candidate\n", encoding="utf-8")
    manifest = _PROVENANCE.source_manifest(tmp_path)
    _PROVENANCE.verify_candidate(tmp_path, manifest)
    (tmp_path / "leviathan/forward_impl.py").write_text("# stale\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source mismatch: leviathan/forward_impl.py"):
        _PROVENANCE.verify_candidate(tmp_path, manifest)


def test_candidate_manifest_cannot_include_private_modules(tmp_path: Path) -> None:
    for name in _PROVENANCE.CANDIDATE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# public kernel\n", encoding="utf-8")
    manifest = _PROVENANCE.source_manifest(tmp_path)
    manifest["files"]["private_model.py"] = "a" * 64
    with pytest.raises(ValueError, match="exactly the public kernel files"):
        _PROVENANCE.verify_candidate(tmp_path, manifest)


def test_native_jtok_manifest_is_required_and_rejects_stale_source(tmp_path: Path) -> None:
    for name in (*_PROVENANCE.CANDIDATE_FILES, _PROVENANCE.JTOK_SOURCE):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# public kernel\n", encoding="utf-8")
    old = _PROVENANCE.source_manifest(tmp_path)
    with pytest.raises(ValueError, match="requires its public source"):
        _PROVENANCE.verify_candidate(tmp_path, old, require_jtok=True)
    native = _PROVENANCE.source_manifest(tmp_path, include_jtok=True)
    _PROVENANCE.verify_candidate(tmp_path, native, require_jtok=True)
    (tmp_path / _PROVENANCE.JTOK_SOURCE).write_text("# changed kernel\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source mismatch: leviathan/jtok.py"):
        _PROVENANCE.verify_candidate(tmp_path, native, require_jtok=True)


def test_projection_manifest_requires_both_native_sources(tmp_path: Path) -> None:
    for name in (*_PROVENANCE.CANDIDATE_FILES, _PROVENANCE.JTOK_SOURCE,
                 _PROVENANCE.JTOK_PROJECTION_SOURCE):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# public kernel\n", encoding="utf-8")
    old = _PROVENANCE.source_manifest(tmp_path, include_jtok=True)
    with pytest.raises(ValueError, match="projection candidate requires"):
        _PROVENANCE.verify_candidate(tmp_path, old, require_projection=True)
    manifest = _PROVENANCE.source_manifest(tmp_path, include_projection=True)
    _PROVENANCE.verify_candidate(tmp_path, manifest, require_projection=True)
    del manifest["files"][_PROVENANCE.JTOK_SOURCE]
    with pytest.raises(ValueError, match="exactly the public kernel files"):
        _PROVENANCE.verify_candidate(tmp_path, manifest, require_projection=True)


def test_compact_jtok_manifest_requires_local_support_consumer(tmp_path: Path) -> None:
    for name in (*_PROVENANCE.CANDIDATE_FILES, _PROVENANCE.JTOK_SOURCE,
                 _PROVENANCE.JTOK_PROJECTION_SOURCE, _PROVENANCE.JTOK_COMPACT_SOURCE):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# public kernel\n", encoding="utf-8")
    old = _PROVENANCE.source_manifest(tmp_path, include_projection=True)
    with pytest.raises(ValueError, match="compact JTok candidate requires"):
        _PROVENANCE.verify_candidate(tmp_path, old, require_compact_jtok=True)
    new = _PROVENANCE.source_manifest(tmp_path, include_compact_jtok=True)
    _PROVENANCE.verify_candidate(tmp_path, new, require_jtok=True,
                                 require_projection=True, require_compact_jtok=True)
    (tmp_path / _PROVENANCE.JTOK_COMPACT_SOURCE).write_text("# stale\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source mismatch: leviathan/jtok_compact.py"):
        _PROVENANCE.verify_candidate(tmp_path, new, require_compact_jtok=True)
