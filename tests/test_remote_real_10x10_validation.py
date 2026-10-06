"""Contract tests for the remote real-data validation runner."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmark"))

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


def test_contract_is_fixed_at_one_hundred_train_and_validation_steps() -> None:
    assert _RUNNER.REAL_TRAIN_STEPS == 100
    assert _RUNNER.REAL_VALIDATION_STEPS == 100
    assert _RUNNER.REQUIRED_DATASET_COLUMNS == {"input_ids", "attention_mask"}


def test_runner_records_canonical_source_and_critical_cce_hashes() -> None:
    assert _RUNNER.CANONICAL_SOURCE_REPOSITORY == "ml-cross-entropy-jtok-pr"
    assert _RUNNER.CANONICAL_SOURCE_COMMIT == "5a4072d"
    assert _RUNNER.CCE_PROVENANCE_FILES == (
        *_RUNNER.CANDIDATE_FILES,
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


def test_metrics_configuration_legacy_off_leaves_module_untouched() -> None:
    training = SimpleNamespace(unrelated_setting=17)
    before = vars(training).copy()
    result = _RUNNER._configure_dynamics_metrics(training, False, 10)
    assert vars(training) == before
    assert result == {
        "enabled": False, "supported": False, "logging_steps": 10,
        "sample_tokens": None, "parameter_samples_per_tensor": None,
        "parameter_interval": None, "log_records": [],
        "note": "private observability output; benchmark cadence is explicit, not the production default",
    }


def test_metrics_configuration_legacy_on_rejects_without_mutation() -> None:
    training = SimpleNamespace(unrelated_setting=17)
    before = vars(training).copy()
    with pytest.raises(RuntimeError, match="complete dynamics API.*USE_DYNAMICS_METRICS"):
        _RUNNER._configure_dynamics_metrics(training, True, 10)
    assert vars(training) == before


@pytest.mark.parametrize("missing", [
    "USE_DYNAMICS_METRICS", "DYNAMICS_SAMPLE_TOKENS",
    "DYNAMICS_PARAMETER_SAMPLES", "DYNAMICS_INTERVAL",
])
def test_metrics_configuration_partial_on_rejects_before_mutation(missing: str) -> None:
    training = SimpleNamespace(USE_DYNAMICS_METRICS=False, DYNAMICS_SAMPLE_TOKENS=64,
                               DYNAMICS_PARAMETER_SAMPLES=128, DYNAMICS_INTERVAL=250)
    delattr(training, missing)
    before = vars(training).copy()
    with pytest.raises(RuntimeError, match="complete dynamics API.*" + missing):
        _RUNNER._configure_dynamics_metrics(training, True, 10)
    assert vars(training) == before


def test_metrics_configuration_partial_off_disables_existing_flag_without_new_attributes() -> None:
    training = SimpleNamespace(USE_DYNAMICS_METRICS=True, DYNAMICS_INTERVAL=250)
    result = _RUNNER._configure_dynamics_metrics(training, False, 10)
    assert vars(training) == {"USE_DYNAMICS_METRICS": False, "DYNAMICS_INTERVAL": 250}
    assert result["supported"] is False
    assert result["parameter_interval"] is None


@pytest.mark.parametrize("enabled", [False, True])
def test_metrics_configuration_supported_sets_flag_and_cadence_only(enabled: bool) -> None:
    training = SimpleNamespace(USE_DYNAMICS_METRICS=not enabled, DYNAMICS_SAMPLE_TOKENS=64,
                               DYNAMICS_PARAMETER_SAMPLES=128, DYNAMICS_INTERVAL=250,
                               unrelated_setting=17)
    result = _RUNNER._configure_dynamics_metrics(training, enabled, 10)
    assert vars(training) == {
        "USE_DYNAMICS_METRICS": enabled, "DYNAMICS_SAMPLE_TOKENS": 64,
        "DYNAMICS_PARAMETER_SAMPLES": 128, "DYNAMICS_INTERVAL": 10, "unrelated_setting": 17,
    }
    assert result["enabled"] is enabled
    assert result["supported"] is True
    assert result["sample_tokens"] == 64
    assert result["parameter_samples_per_tensor"] == 128
    assert result["parameter_interval"] == 10


def test_existing_split_loader_selects_exactly_one_hundred_batches(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    datasets = {
        "train": _FakeDataset(640),
        "validation": _FakeDataset(640),
    }

    def loader(path: Path):
        return datasets[path.name]

    train, validation = _RUNNER._prepare_preexisting_splits(tmp_path, 4, loader=loader)
    assert len(train) == 400
    assert len(validation) == 400


def test_loader_accepts_explicit_train_and_validation_directories(tmp_path: Path) -> None:
    train_path = tmp_path / "tokenized_train"
    validation_path = tmp_path / "tokenized_validation"
    train_path.mkdir()
    validation_path.mkdir()
    datasets = {
        train_path: _FakeDataset(640),
        validation_path: _FakeDataset(640),
    }

    train, validation = _RUNNER._prepare_preexisting_splits(
        None,
        4,
        loader=lambda path: datasets[path],
        train_data_dir=train_path,
        validation_data_dir=validation_path,
    )
    assert len(train) == 400
    assert len(validation) == 400


def test_loader_rejects_missing_real_rows(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()

    def loader(path: Path):
        return _FakeDataset(399)

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


def test_loader_rejects_validation_5025_for_one_hundred_batches_of_64(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    datasets = {"train": _FakeDataset(6400), "validation": _FakeDataset(5025)}
    with pytest.raises(ValueError, match=r"validation.*available=5025.*required=6400.*100 batches of 64"):
        _RUNNER._prepare_preexisting_splits(tmp_path, 64, loader=lambda path: datasets[path.name])


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


def test_loader_training_source_validation_is_explicit_disjoint_and_no_repetition(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    datasets = {"train": _FakeDataset(12800), "validation": _FakeDataset(5025)}
    train, validation = _RUNNER._prepare_preexisting_splits(
        tmp_path, 64, loader=lambda path: datasets[path.name], validation_from_train=True)
    assert train._rows == list(range(6400))
    assert validation._rows == list(range(6400, 12800))
    assert set(train._rows).isdisjoint(validation._rows)


def test_loader_training_source_validation_rejects_overlap_when_rows_short(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    datasets = {"train": _FakeDataset(12799), "validation": _FakeDataset(6400)}
    with pytest.raises(ValueError, match=r"validation.*available=12799.*required=12800"):
        _RUNNER._prepare_preexisting_splits(
            tmp_path, 64, loader=lambda path: datasets[path.name], validation_from_train=True)


def test_stable_timing_excludes_cold_warmup_and_profile_active_steps() -> None:
    records = [
        {"optimizer_step": step, "phase": _RUNNER._profile_phase(step),
         "compute_seconds": 10.0 if step <= 4 else 0.2,
         "e2e_seconds": 11.0 if step <= 4 else 0.25}
        for step in range(1, 11)
    ]
    result = _RUNNER._stable_step_statistics(records)
    assert result["stable_optimizer_steps"] == [5, 6, 7, 8, 9, 10]
    assert result["compute"]["steps_per_second"] == 5.0
    assert result["e2e"]["steps_per_second"] == 4.0
    assert result["compute"]["count"] == 6


def test_stable_timing_statistics_and_p95_are_explicit() -> None:
    result = _RUNNER._step_statistics([0.2, 0.1, 0.4, 0.3])
    assert result["median_seconds"] == 0.25
    assert result["steps_per_second"] == 4.0
    assert result["p95_seconds"] == pytest.approx(0.385)
    assert result["population_stddev_seconds"] > 0


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float('nan')], [float('inf')]])
def test_stable_timing_rejects_missing_or_invalid_evidence(values) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _RUNNER._step_statistics(values)


def test_stable_timing_callback_uses_runner_output_not_training_arguments(tmp_path: Path) -> None:
    tree = ast.parse(_RUNNER_PATH.read_text(encoding="utf-8"))
    callback = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef) and node.name == "TimingCallback")
    path = tmp_path / "result.json"
    cuda = SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda: 8,
                           max_memory_reserved=lambda: 16)
    result = {"timing": {}}
    scope = {
        "TrainerCallback": object, "torch": SimpleNamespace(cuda=cuda),
        "time": SimpleNamespace(perf_counter=lambda: 4.0), "math": math,
        "args": SimpleNamespace(output=path), "result": result, "profile_holder": {},
        "result_output_path": path,
        "timing": {"_step_starts": [1.0], "train_step_seconds": [],
                   "_compute_start": 2.0, "_e2e_start": 1.0, "step_records": [],
                   "_loss_snapshot": SimpleNamespace(item=lambda: 2.5), "train_losses": []},
        "_profile_phase": _RUNNER._profile_phase, "_write_result": _RUNNER._write_result,
    }
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(_RUNNER_PATH), "exec"), scope)
    # Deliberately lacks .output, just like real Transformers TrainingArguments.
    training_args = SimpleNamespace(output_dir="unused")
    control = object()
    assert scope["TimingCallback"]().on_step_end(training_args, None, control) is control
    assert path.exists()
    assert result["timing"]["step_records"][0]["compute_seconds"] == 2.0
