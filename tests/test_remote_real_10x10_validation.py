"""Contract tests for the remote real-data validation runner."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_RUNNER_PATH = Path(__file__).parents[1] / "benchmark" / "remote_real_10x10_validation.py"
_SPEC = importlib.util.spec_from_file_location("remote_real_10x10_validation", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


class _FakeDataset:
    column_names = ["input_ids", "attention_mask"]

    def __init__(self, size: int) -> None:
        self._rows = list(range(size))

    def __len__(self) -> int:
        return len(self._rows)

    def select(self, indices):
        selected = list(indices)
        result = _FakeDataset(0)
        result._rows = [self._rows[index] for index in selected]
        return result


def test_contract_is_fixed_at_ten_train_and_validation_steps() -> None:
    assert _RUNNER.REAL_TRAIN_STEPS == 10
    assert _RUNNER.REAL_VALIDATION_STEPS == 10
    assert _RUNNER.REQUIRED_DATASET_COLUMNS == {"input_ids", "attention_mask"}


def test_runner_records_canonical_source_and_critical_cce_hashes() -> None:
    assert _RUNNER.CANONICAL_SOURCE_REPOSITORY == "ml-cross-entropy-jtok-pr"
    assert _RUNNER.CANONICAL_SOURCE_COMMIT == "5a4072d"
    assert _RUNNER.CCE_PROVENANCE_FILES == (
        "leviathan/backward_dot_kernels.py",
        "leviathan/backward_kernels.py",
        "leviathan/jtok.py",
    )


def test_strict_leviathan_guard_is_marked_on_model_configs() -> None:
    class Config:
        pass

    class Model:
        config = Config()
        model = type("InnerModel", (), {"config": Config()})()

    result = {}
    returned = _RUNNER._require_strict_leviathan(Model(), result)
    assert returned is not None
    assert Model.config._require_leviathan_triton is True
    assert result["runtime_verification"]["reference_fallback_allowed"] is False


def test_runner_has_no_dataset_download_call() -> None:
    tree = ast.parse(_RUNNER_PATH.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_dataset"
    ]
    assert calls == []


def test_training_module_loader_exposes_sibling_modules(monkeypatch, tmp_path: Path) -> None:
    sibling = tmp_path / "configuration_neollm.py"
    sibling.write_text("VALUE = 17\n", encoding="utf-8")
    train = tmp_path / "train.py"
    train.write_text(
        "from configuration_neollm import VALUE\n"
        "loaded_value = VALUE\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_RUNNER.sys, "path", [entry for entry in _RUNNER.sys.path if entry != str(tmp_path)])
    module = _RUNNER._load_training_module(train)
    assert module.loaded_value == 17


def test_existing_split_loader_selects_exactly_ten_batches(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    datasets = {
        "train": _FakeDataset(64),
        "validation": _FakeDataset(64),
    }

    def loader(path: Path):
        return datasets[path.name]

    train, validation = _RUNNER._prepare_preexisting_splits(tmp_path, 4, loader=loader)
    assert len(train) == 40
    assert len(validation) == 40


def test_loader_accepts_explicit_train_and_validation_directories(tmp_path: Path) -> None:
    train_path = tmp_path / "tokenized_train"
    validation_path = tmp_path / "tokenized_validation"
    train_path.mkdir()
    validation_path.mkdir()
    datasets = {
        train_path: _FakeDataset(64),
        validation_path: _FakeDataset(64),
    }

    train, validation = _RUNNER._prepare_preexisting_splits(
        None,
        4,
        loader=lambda path: datasets[path],
        train_data_dir=train_path,
        validation_data_dir=validation_path,
    )
    assert len(train) == 40
    assert len(validation) == 40


def test_loader_rejects_missing_real_rows(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()

    def loader(path: Path):
        return _FakeDataset(39)

    with pytest.raises(ValueError, match="fewer rows"):
        _RUNNER._prepare_preexisting_splits(tmp_path, 4, loader=loader)


def test_loader_rejects_a_dataset_without_validation_split(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    class FakeDatasetDict:
        def keys(self):
            return ["train"]

    with pytest.raises(ValueError, match="train and validation"):
        _RUNNER._prepare_preexisting_splits(
            tmp_path, 4, loader=lambda _path: FakeDatasetDict()
        )


def test_diagnostics_redact_hosts_paths_and_secrets() -> None:
    raw = (
        "host=192.0.2.4 path=C:\\Users\\private-user\\run "
        "token=do-not-store https://example.invalid/run"
    )
    redacted = _RUNNER._redact_text(raw)
    assert "192.0.2.4" not in redacted
    assert "private-user" not in redacted
    assert "do-not-store" not in redacted
    assert "example.invalid" not in redacted
