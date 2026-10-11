"""Contract tests for the remote real-data validation runner."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_cuda_graph_tree_replay_misses_are_not_reported_as_capture_failures():
    tree = ast.parse(_RUNNER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compiler_snapshot"
    )
    # Execute the actual snapshot body with metadata-only stand-ins; this
    # validates report semantics without creating a GPU manager or model.
    function.body = [node for node in function.body if not isinstance(node, ast.ImportFrom)]
    manager = SimpleNamespace(
        roots={},
        num_rerecord={},
        debug_fail_counter=2,
    )
    scope = {
        "counters": {
            "stats": {"unique_graphs": 1},
            "graph_break": {},
            "recompiles": {},
            "inductor": {},
        },
        "compilation_time_metrics": {},
        "metrics": SimpleNamespace(generated_kernel_count=3),
        "get_manager": lambda **kwargs: manager,
        "sys": SimpleNamespace(modules={}),
        "_external_jit_snapshot": lambda: {"cce_triton_jit_entries": {}, "cute_jit_entries": {}},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(_RUNNER_PATH), "exec"), scope)
    snapshot = scope["compiler_snapshot"]()
    assert snapshot["cuda_graph_new_recording_events"] == 2
    assert snapshot["cuda_graph_rerecord_count"] == 0
    assert "cuda_graph_recording_failures" not in snapshot


@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_native_fallback_guard_rejects_runtime_fallback(phase):
    compiler = SimpleNamespace(leviathan_forward_ref=lambda: None, leviathan_backward=lambda: None)
    result = {}
    _RUNNER._forbid_leviathan_reference_fallbacks(compiler, result)
    method = compiler.leviathan_forward_ref if phase == "forward" else compiler.leviathan_backward
    with pytest.raises(RuntimeError, match=f"reference {phase} fallback"):
        method()
    assert result["leviathan_reference_fallback_attempts"][phase] == 1
    assert sum(result["leviathan_reference_fallback_attempts"].values()) == 1


@pytest.mark.parametrize("name", ["LEV_DDELTA_BM", "LEV_DDELTA_BD", "LEV_DDELTA_BR"])
@pytest.mark.parametrize("value", ["auto", "0", "-1"])
def test_tile_reporting_label_cannot_silently_select_reference(monkeypatch, name, value):
    for key in ("LEV_DDELTA_BM", "LEV_DDELTA_BD", "LEV_DDELTA_BR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="positive integer or unset"):
        _RUNNER._validate_leviathan_tile_overrides()


def test_tile_overrides_accept_unset_automatic_defaults_and_integers(monkeypatch):
    for key in ("LEV_DDELTA_BM", "LEV_DDELTA_BD", "LEV_DDELTA_BR"):
        monkeypatch.delenv(key, raising=False)
    _RUNNER._validate_leviathan_tile_overrides()
    for key, value in zip(("LEV_DDELTA_BM", "LEV_DDELTA_BD", "LEV_DDELTA_BR"), ("128", "1", "64")):
        monkeypatch.setenv(key, value)
    _RUNNER._validate_leviathan_tile_overrides()


@pytest.mark.parametrize("backend", ["off", "reference", "triton", "cute"])
def test_normalization_comparison_cli(monkeypatch, tmp_path, backend):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--mode",
            "off",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "private.json"),
            "--profile",
            str(tmp_path / "private.trace.json"),
            "--embedding-mode",
            "table",
            "--siamesenorm-backend",
            backend,
        ],
    )
    args = _RUNNER._parse_args()
    assert args.embedding_mode == "table"
    assert args.siamesenorm_backend == backend


def test_perplexity_overhead_control_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--mode",
            "off",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "private.json"),
            "--profile",
            str(tmp_path / "private.trace.json"),
            "--no-perplexity-metrics",
        ],
    )
    assert _RUNNER._parse_args().no_perplexity_metrics


@pytest.mark.parametrize("phase", ["prewarm", "initial", "transition", "validation"])
def test_phase_execution_traces_have_distinct_paths(tmp_path, phase):
    path = tmp_path / "execution.trace.json"
    assert _RUNNER._trace_output_path(path, phase) == tmp_path / f"execution.trace-{phase}.json"


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
        "from configuration_neollm import VALUE\nloaded_value = VALUE\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        _RUNNER.sys, "path", [entry for entry in _RUNNER.sys.path if entry != str(tmp_path)]
    )
    module = _RUNNER._load_training_module(train)
    assert module.loaded_value == 17


def test_metrics_configuration_legacy_off_leaves_module_untouched() -> None:
    training = SimpleNamespace(unrelated_setting=17)
    before = vars(training).copy()
    result = _RUNNER._configure_dynamics_metrics(training, False, 10)
    assert vars(training) == before
    assert result == {
        "enabled": False,
        "supported": False,
        "logging_steps": 10,
        "sample_tokens": None,
        "parameter_samples_per_tensor": None,
        "parameter_interval": None,
        "log_records": [],
        "note": "private observability output; benchmark cadence is explicit, not the production default",
    }


def test_metrics_configuration_legacy_on_rejects_without_mutation() -> None:
    training = SimpleNamespace(unrelated_setting=17)
    before = vars(training).copy()
    with pytest.raises(RuntimeError, match="complete dynamics API.*USE_DYNAMICS_METRICS"):
        _RUNNER._configure_dynamics_metrics(training, True, 10)
    assert vars(training) == before


@pytest.mark.parametrize(
    "missing",
    [
        "USE_DYNAMICS_METRICS",
        "DYNAMICS_SAMPLE_TOKENS",
        "DYNAMICS_PARAMETER_SAMPLES",
        "DYNAMICS_INTERVAL",
    ],
)
def test_metrics_configuration_partial_on_rejects_before_mutation(missing: str) -> None:
    training = SimpleNamespace(
        USE_DYNAMICS_METRICS=False,
        DYNAMICS_SAMPLE_TOKENS=64,
        DYNAMICS_PARAMETER_SAMPLES=128,
        DYNAMICS_INTERVAL=250,
    )
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
    training = SimpleNamespace(
        USE_DYNAMICS_METRICS=not enabled,
        DYNAMICS_SAMPLE_TOKENS=64,
        DYNAMICS_PARAMETER_SAMPLES=128,
        DYNAMICS_INTERVAL=250,
        unrelated_setting=17,
    )
    result = _RUNNER._configure_dynamics_metrics(training, enabled, 10)
    assert vars(training) == {
        "USE_DYNAMICS_METRICS": enabled,
        "DYNAMICS_SAMPLE_TOKENS": 64,
        "DYNAMICS_PARAMETER_SAMPLES": 128,
        "DYNAMICS_INTERVAL": 10,
        "unrelated_setting": 17,
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
        _RUNNER._prepare_preexisting_splits(tmp_path, 4, loader=lambda _path: FakeDatasetDict())


def test_loader_rejects_validation_5025_for_one_hundred_batches_of_64(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    datasets = {"train": _FakeDataset(6400), "validation": _FakeDataset(5025)}
    with pytest.raises(
        ValueError, match=r"validation.*available=5025.*required=6400.*100 batches of 64"
    ):
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


def test_loader_training_source_validation_is_explicit_disjoint_and_no_repetition(
    tmp_path: Path,
) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    datasets = {"train": _FakeDataset(12800), "validation": _FakeDataset(5025)}
    train, validation = _RUNNER._prepare_preexisting_splits(
        tmp_path, 64, loader=lambda path: datasets[path.name], validation_from_train=True
    )
    assert train._rows == list(range(6400))
    assert validation._rows == list(range(6400, 12800))
    assert set(train._rows).isdisjoint(validation._rows)


def test_loader_training_source_validation_rejects_overlap_when_rows_short(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    datasets = {"train": _FakeDataset(12799), "validation": _FakeDataset(6400)}
    with pytest.raises(ValueError, match=r"validation.*available=12799.*required=12800"):
        _RUNNER._prepare_preexisting_splits(
            tmp_path, 64, loader=lambda path: datasets[path.name], validation_from_train=True
        )


def test_stable_timing_excludes_cold_warmup_and_profile_active_steps() -> None:
    records = [
        {
            "optimizer_step": step,
            "phase": _RUNNER._profile_phase(step),
            "compute_seconds": 10.0 if step <= 4 else 0.2,
            "e2e_seconds": 11.0 if step <= 4 else 0.25,
        }
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


@pytest.mark.parametrize("patch_steps", [0, 67])
def test_patch_comparison_configuration_keeps_original_sequences_and_continuous_updates(
    patch_steps,
):
    training = SimpleNamespace(
        USE_PATCH_TRAINING=False,
        PATCH_SIZE=2,
        PATCH_TRAINING_STEPS=0,
        AdEMAMixTrainer=SimpleNamespace(prewarm_patch_signatures=lambda: None),
    )
    result = _RUNNER._configure_patch_training(training, 4, patch_steps)
    assert training.USE_PATCH_TRAINING and training.PATCH_SIZE == 4
    assert training.PATCH_TRAINING_STEPS == patch_steps
    assert result["raw_sequence_length"] == 512
    assert (result["patch_core_length"], result["token_core_length"]) == (128, 512)
    assert result["token_steps"] == 100 - patch_steps
    assert result["matched_token_control"] == (patch_steps == 0)
    assert result["optimizer_transition"] == "continuous"
    assert not result["concatenate_examples"]
    assert result["validation"] == "ordinary_unpatched_tokens"


def test_patch_configuration_rejects_unsupported_api_before_mutation():
    training = SimpleNamespace(USE_PATCH_TRAINING=False, PATCH_SIZE=2, PATCH_TRAINING_STEPS=0)
    before = vars(training).copy()
    with pytest.raises(ValueError, match="prewarm Trainer API"):
        _RUNNER._configure_patch_training(training, 4, 67)
    assert vars(training) == before


def test_patch_transition_profiling_is_excluded_from_stable_latency():
    records = [
        {
            "optimizer_step": step,
            "phase": _RUNNER._profile_phase(step, 67),
            "compute_seconds": 0.2,
            "e2e_seconds": 0.25,
        }
        for step in range(1, 101)
    ]
    report = _RUNNER._stable_step_statistics(records)
    assert report["compute"]["count"] == 93
    assert not set((1, 2, 3, 4, 67, 68, 69)).intersection(report["stable_optimizer_steps"])
    assert _RUNNER._profile_phase(67, 67) == "transition_profile_warmup"
    assert _RUNNER._profile_phase(68, 67) == "transition_profile_active"
    assert _RUNNER._profile_phase(70, 67) == "stable_unprofiled"


@pytest.mark.parametrize(
    "key",
    [
        "unique_graphs",
        "graph_breaks",
        "recompiles",
        "generated_inductor_kernels",
        "compilation_event_counts",
        "cuda_graph_tree_nodes",
        "cuda_graph_rerecord_count",
        "cuda_graph_new_recording_events",
        "cce_triton_jit_entries",
        "cute_jit_entries",
    ],
)
def test_patch_signature_guard_detects_compilation_or_recording_growth(key):
    warm = {key: 1, "inductor_counters": {"cache_hit": 0}}
    _RUNNER._assert_prewarmed_signatures_stable(
        warm, {key: 1, "inductor_counters": {"cache_hit": 10}}
    )
    with pytest.raises(RuntimeError, match=key):
        _RUNNER._assert_prewarmed_signatures_stable(warm, {key: 2})


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_stable_timing_rejects_missing_or_invalid_evidence(values) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _RUNNER._step_statistics(values)


def test_stable_timing_callback_uses_runner_output_not_training_arguments(tmp_path: Path) -> None:
    tree = ast.parse(_RUNNER_PATH.read_text(encoding="utf-8"))
    callback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "TimingCallback"
    )
    path = tmp_path / "result.json"
    cuda = SimpleNamespace(
        synchronize=lambda: None, max_memory_allocated=lambda: 8, max_memory_reserved=lambda: 16
    )
    result = {"timing": {}}
    scope = {
        "TrainerCallback": object,
        "torch": SimpleNamespace(cuda=cuda),
        "time": SimpleNamespace(perf_counter=lambda: 4.0),
        "math": math,
        "args": SimpleNamespace(output=path),
        "result": result,
        "profile_holder": {},
        "result_output_path": path,
        "timing": {
            "_step_starts": [1.0],
            "train_step_seconds": [],
            "_compute_start": 2.0,
            "_e2e_start": 1.0,
            "step_records": [],
            "_loss_snapshot": SimpleNamespace(item=lambda: 2.5),
            "train_losses": [],
        },
        "_profile_phase": _RUNNER._profile_phase,
        "_write_result": _RUNNER._write_result,
    }
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(_RUNNER_PATH), "exec"), scope)
    # Deliberately lacks .output, just like real Transformers TrainingArguments.
    training_args = SimpleNamespace(output_dir="unused")
    control = object()
    assert scope["TimingCallback"]().on_step_end(training_args, None, control) is control
    assert path.exists()
    assert result["timing"]["step_records"][0]["compute_seconds"] == 2.0


def _completed_curriculum_comparison_fixture():
    from copy import deepcopy

    control = {
        key: {}
        for key in (
            "module_hashes",
            "runtime",
            "kernel_policy",
            "normalization_comparison",
            "perplexity_metrics",
            "topk_metrics",
            "imports",
            "data_policy",
            "dynamics",
        )
    }
    control.update(
        status="ok", mode="off", batch_size=64, sequence_length=512, seed=42, data_seed=42
    )
    control["patch_curriculum"] = dict(
        enabled=True,
        matched_token_control=True,
        patch_steps=0,
        patch_size=4,
        raw_sequence_length=512,
        validation="normal",
        optimizer_transition="continuous",
        concatenate_examples=False,
    )
    control["timing"] = dict(
        train_steps_completed=100,
        validation_steps_completed=100,
        train_wall_seconds=30.0,
        train_compute_seconds_sum=20.0,
        train_batch_ready_seconds_sum=22.0,
        validation_wall_seconds=8.0,
        validation_prediction_seconds_sum=7.0,
    )
    control["timing"]["step_records"] = [
        dict(
            optimizer_step=step,
            phase=_RUNNER._profile_phase(step),
            curriculum_phase="token",
            compute_seconds=0.2,
        )
        for step in range(1, 101)
    ]
    control["memory"] = {
        key: 100
        for key in (
            "train_peak_allocated_bytes",
            "train_peak_reserved_bytes",
            "validation_peak_allocated_bytes",
            "validation_peak_reserved_bytes",
        )
    }
    candidate = deepcopy(control)
    candidate["patch_curriculum"].update(matched_token_control=False, patch_steps=67)
    candidate["timing"]["train_compute_seconds_sum"] = 10.0
    for row in candidate["timing"]["step_records"]:
        step = row["optimizer_step"]
        row.update(
            phase=_RUNNER._profile_phase(step, 67),
            curriculum_phase="patch" if step <= 67 else "token",
        )
        # The first profiled transition steps must not bias recovery timing.
        row["compute_seconds"] = 10.0 if step in (68, 69) else 0.202
    candidate["memory"]["train_peak_allocated_bytes"] = 120
    return control, candidate


def test_curriculum_comparison_reuses_control_and_matches_stable_token_indices():
    control, candidate = _completed_curriculum_comparison_fixture()
    result = _RUNNER._compare_curriculum_runs(control, candidate)
    assert result["token_recovery"]["optimizer_steps"] == list(range(70, 101))
    assert result["token_recovery"]["latency_change_percent"] == pytest.approx(1.0)
    assert result["timing"]["train_compute_seconds_sum"]["change_percent"] == -50.0
    assert result["memory"]["train_peak_allocated_bytes"]["change_bytes"] == 20


@pytest.mark.parametrize(
    "key",
    [
        "module_hashes",
        "runtime",
        "kernel_policy",
        "perplexity_metrics",
        "topk_metrics",
        "data_policy",
        "imports",
        "seed",
    ],
)
def test_curriculum_comparison_rejects_mismatched_controls(key):
    control, candidate = _completed_curriculum_comparison_fixture()
    candidate[key] = "different"
    with pytest.raises(ValueError, match="mismatched"):
        _RUNNER._compare_curriculum_runs(control, candidate)


def test_curriculum_comparison_rejects_partial_or_wrong_phase_controls():
    control, candidate = _completed_curriculum_comparison_fixture()
    control["timing"]["validation_steps_completed"] = 99
    with pytest.raises(ValueError, match="completed real100"):
        _RUNNER._compare_curriculum_runs(control, candidate)
    control["timing"]["validation_steps_completed"] = 100
    control["patch_curriculum"]["matched_token_control"] = False
    with pytest.raises(ValueError, match="matched all-K1"):
        _RUNNER._compare_curriculum_runs(control, candidate)


@pytest.mark.parametrize("count", [1.0, 3.76e33, float("nan"), -0.1, 0.5])
def test_perplexity_guard_rejects_phantom_patch_phase(count):
    with pytest.raises(RuntimeError, match="target count"):
        _RUNNER._validate_perplexity_record({"train_patch_valid_targets": count}, 1, 10, 64, 0)


@pytest.mark.parametrize(
    "metrics",
    [
        {"train_valid_targets": 327681.0},
        {"train_valid_targets": 1.0, "train_ce": float("nan"), "train_ppl": 2.0},
        {"train_valid_targets": 1.0, "train_ce": 1.0, "train_ppl": 2.0, "train_mile_ppl": 0.5},
        {"train_patch_ce": 1.0},
    ],
)
def test_perplexity_guard_rejects_corrupt_statistics(metrics):
    with pytest.raises(RuntimeError):
        _RUNNER._validate_perplexity_record(metrics, 1, 10, 64, 0)


def test_perplexity_guard_accepts_mixed_transition_window():
    metrics = {
        "train_valid_targets": 3 * 64 * 511.0,
        "train_ce": 2.0,
        "train_ppl": 7.389,
        "train_mile_loss": 2.1,
        "train_mile_ppl": 8.166,
        "train_patch_valid_targets": 7 * 64 * 508.0,
        "train_patch_ce": 3.0,
        "train_patch_exp_ce": 20.085,
        "train_patch_mile_loss": 3.1,
        "train_patch_mile_exp_loss": 22.198,
    }
    _RUNNER._validate_perplexity_record(metrics, 61, 70, 64, 67)
    metrics["train_patch_valid_targets"] = 8 * 64 * 512.0
    with pytest.raises(RuntimeError, match="target count"):
        _RUNNER._validate_perplexity_record(metrics, 61, 70, 64, 67)


def test_curriculum_comparison_rejects_old_corrupt_control():
    control, candidate = _completed_curriculum_comparison_fixture()
    control["timing"]["perplexity_log_records"] = [
        {"optimizer_step": 20, "metrics": {"train_patch_valid_targets": 3.76e33}}
    ]
    with pytest.raises(RuntimeError, match="target count"):
        _RUNNER._compare_curriculum_runs(control, candidate)
