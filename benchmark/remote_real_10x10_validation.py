"""Run the real NeoLLM path for the current fixed 100+100 validation contract.

This runner is deliberately read-only with respect to datasets.  It loads an
already-tokenized Hugging Face dataset with ``load_from_disk`` and never calls
``load_dataset``.  A missing or undersized dataset is an error, not a reason to
download one or silently switch to synthetic data.

The runner imports the supplied training script and patches only the dataset
loader and short-run controls.  Model construction, Leviathan/JToK dispatch,
losses, ``torch.compile(max-autotune)``, the optimizer, and the other training
policy remain in the supplied training script. The run is exactly 100
optimizer steps followed by evaluation over exactly 100 batches. Synthetic
inputs belong in a separate edge-case probe and are not accepted here.

The JSON result intentionally contains no connection details, credentials,
personal paths, command lines, or raw exception tracebacks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from leviathan_candidate_provenance import CANDIDATE_FILES, verify_candidate

REAL_TRAIN_STEPS = 100
REAL_VALIDATION_STEPS = 100
REQUIRED_DATASET_COLUMNS = frozenset(("input_ids", "attention_mask"))
CANONICAL_SOURCE_REPOSITORY = "ml-cross-entropy-jtok-pr"
CANONICAL_SOURCE_COMMIT = "5a4072d"
CCE_PROVENANCE_FILES = (
    *CANDIDATE_FILES,
    "leviathan/jtok.py",
)


def _profile_phase(step: int, patch_steps: int = 0) -> str:
    """One-based phase for the fixed wait=1, warmup=1, active=2 schedule."""
    if step < 1:
        raise ValueError("optimizer step must be one-based")
    if patch_steps > 4 and step in (patch_steps, patch_steps + 1, patch_steps + 2):
        return "transition_profile_warmup" if step == patch_steps else "transition_profile_active"
    return {1: "cold_wait", 2: "profile_warmup", 3: "profile_active", 4: "profile_active"}.get(
        step, "stable_unprofiled"
    )


def _configure_patch_training(
    training: Any, patch_size: int | None, patch_steps: int
) -> dict[str, Any]:
    """Opt into the existing original-sequence producer, not example packing."""
    if patch_size is None:
        if hasattr(training, "USE_PATCH_TRAINING"):
            training.USE_PATCH_TRAINING = False
        return {"enabled": False, "patch_steps": 0}
    required = ("USE_PATCH_TRAINING", "PATCH_SIZE", "PATCH_TRAINING_STEPS")
    if (
        patch_size != 4
        or not 0 <= patch_steps < REAL_TRAIN_STEPS
        or any(not hasattr(training, name) for name in required)
        or not callable(
            getattr(getattr(training, "AdEMAMixTrainer", None), "prewarm_patch_signatures", None)
        )
    ):
        raise ValueError(
            "Patch comparison requires K=4, steps in [0,99] and the prewarm Trainer API."
        )
    training.USE_PATCH_TRAINING = True
    training.PATCH_SIZE = patch_size
    training.PATCH_TRAINING_STEPS = patch_steps
    return {
        "enabled": True,
        "patch_size": patch_size,
        "patch_steps": patch_steps,
        "token_steps": REAL_TRAIN_STEPS - patch_steps,
        "raw_sequence_length": 512,
        "patch_core_length": 512 // patch_size,
        "token_core_length": 512,
        "validation": "ordinary_unpatched_tokens",
        "optimizer_transition": "continuous",
        "matched_token_control": patch_steps == 0,
        "concatenate_examples": False,
    }


def _trace_output_path(path: Path, phase: str) -> Path:
    return path.with_name(f"{path.stem}-{phase}{path.suffix}")


def _assert_prewarmed_signatures_stable(warm: dict[str, Any], current: dict[str, Any]) -> None:
    # General Inductor counters also include cache hits: growth there does not
    # mean compilation. These fields specifically measure new code/capture.
    fields = (
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
    )
    changed = [key for key in fields if current.get(key) != warm.get(key)]
    if changed:
        raise RuntimeError("Prewarmed execution signatures changed: " + ", ".join(changed))


def _external_jit_snapshot() -> dict[str, Any]:
    from triton.runtime.jit import JITFunction

    entries = {}
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("cut_cross_entropy."):
            continue
        for function in tuple(vars(module).values()):
            while (
                not isinstance(function, JITFunction) and getattr(function, "fn", None) is not None
            ):
                function = function.fn
            if isinstance(function, JITFunction) and function.device_caches:
                entries[function.fn.__module__ + "." + function.fn.__name__] = {
                    str(device): len(cache[0]) for device, cache in function.device_caches.items()
                }
    return {
        "cce_triton_jit_entries": entries,
        "cute_jit_entries": {
            name: len(module._CACHE)
            for name, module in tuple(sys.modules.items())
            if name.startswith("cut_cross_entropy.")
            and isinstance(getattr(module, "_CACHE", None), dict)
        },
    }


def _step_statistics(values: list[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("step times must be finite and positive")
    ordered = sorted(values)
    p95_index = 0.95 * (len(ordered) - 1)
    left = math.floor(p95_index)
    right = math.ceil(p95_index)
    p95 = ordered[left] + (ordered[right] - ordered[left]) * (p95_index - left)
    median = statistics.median(values)
    return {
        "count": len(values),
        "median_seconds": median,
        "steps_per_second": 1.0 / median,
        "p95_seconds": p95,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "population_stddev_seconds": statistics.pstdev(values),
    }


def _stable_step_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [record for record in records if record["phase"] == "stable_unprofiled"]
    return {
        "metric": "reciprocal_of_median_batch_ready_compute_seconds",
        "stable_optimizer_steps": [record["optimizer_step"] for record in stable],
        "compute": _step_statistics([record["compute_seconds"] for record in stable]),
        "e2e": _step_statistics([record["e2e_seconds"] for record in stable]),
        "p95_definition": "linear_interpolation_at_0.95_times_n_minus_one",
    }


def _redact_text(value: object) -> str:
    """Return a bounded diagnostic string without secrets or personal paths."""

    text = str(value)[:1000]
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<redacted-host>", text)
    text = re.sub(
        r"(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\s:'\"]+",
        "<redacted-local-user>",
        text,
    )
    text = re.sub(r"(?i)(/home/|/root(?:/|\b))[^\s:'\"]*", "<redacted-user-path>", text)
    text = re.sub(r"(?i)https?://[^\s'\"]+", "<redacted-url>", text)
    text = re.sub(
        r"(?i)(?:password|passwd|token|secret|api[_-]?key|private[_-]?key|authorization)"
        r"\s*[:=]\s*[^\s,;]+",
        "<redacted-secret>",
        text,
    )
    text = re.sub(
        r"(?i)(?:--?(?:password|passwd|token|secret|api[_-]?key))\s+[^\s]+",
        "<redacted-secret-argument>",
        text,
    )
    return text


def _validate_perplexity_record(
    metrics: dict[str, float], first_step: int, last_step: int, batch_size: int, patch_steps: int
) -> None:
    """Reject nonfinite/impossible telemetry before accepting a real-data run."""
    patch_updates = max(0, min(last_step, patch_steps) - min(first_step - 1, patch_steps))
    total_updates = last_step - first_step + 1
    for stem, updates, exp_key in (
        ("train_", total_updates - patch_updates, "ppl"),
        ("train_patch_", patch_updates, "exp_ce"),
    ):
        count = metrics.get(stem + "valid_targets", 0.0)
        if (
            not math.isfinite(count)
            or count < 0
            or count != int(count)
            or count > updates * batch_size * 512
        ):
            raise RuntimeError("Invalid perplexity target count for the logged curriculum window.")
        if count:
            for key in ("ce", exp_key):
                value = metrics.get(stem + key, float("nan"))
                if not math.isfinite(value) or value < 0 or (key == exp_key and value < 1):
                    raise RuntimeError("Invalid perplexity loss or exponential.")
            for key in ("mile_loss", "mile_ppl" if stem == "train_" else "mile_exp_loss"):
                if stem + key not in metrics:
                    continue
                value = metrics[stem + key]
                if not math.isfinite(value) or value < (1 if key != "mile_loss" else 0):
                    raise RuntimeError("Invalid MiLe perplexity metric.")
        elif any(
            metrics.get(stem + key, 0.0) != 0.0
            for key in ("ce", exp_key, "mile_loss", "mile_ppl", "mile_exp_loss")
        ):
            raise RuntimeError("Perplexity metrics exist without supervised targets.")


def _compare_curriculum_runs(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare completed private runs, without repeating an existing control.

    Source/runtime/policy must match; only the patch-step boundary may differ.
    Token recovery compares the same unprofiled optimizer-step indices in both
    runs, not late candidate steps against cold/profiled control updates. This
    is one sequential comparison, not evidence of statistical equivalence.
    """
    for run in (control, candidate):
        timing = run.get("timing", {})
        if (
            run.get("status") != "ok"
            or timing.get("train_steps_completed") != REAL_TRAIN_STEPS
            or timing.get("validation_steps_completed") != REAL_VALIDATION_STEPS
        ):
            raise ValueError("Curriculum comparison requires completed real100+100 runs.")
        previous = 0
        for entry in timing.get("perplexity_log_records", []):
            _validate_perplexity_record(
                entry["metrics"],
                previous + 1,
                entry["optimizer_step"],
                run["batch_size"],
                run["patch_curriculum"]["patch_steps"],
            )
            previous = entry["optimizer_step"]
    for key in (
        "mode",
        "batch_size",
        "sequence_length",
        "seed",
        "data_seed",
        "data_policy",
        "module_hashes",
        "runtime",
        "kernel_policy",
        "normalization_comparison",
        "perplexity_metrics",
        "topk_metrics",
        "imports",
    ):
        if key not in control or key not in candidate or control[key] != candidate[key]:
            raise ValueError(f"Curriculum comparison has mismatched {key}.")
    for key in (
        "enabled",
        "logging_steps",
        "sample_tokens",
        "parameter_samples_per_tensor",
        "parameter_interval",
    ):
        if control.get("dynamics", {}).get(key) != candidate.get("dynamics", {}).get(key):
            raise ValueError("Curriculum comparison has mismatched dynamics policy.")
    baseline = control.get("patch_curriculum", {})
    curriculum = candidate.get("patch_curriculum", {})
    if (
        not baseline.get("matched_token_control")
        or not curriculum.get("enabled")
        or baseline.get("patch_steps") != 0
        or curriculum.get("patch_steps", 0) <= 0
    ):
        raise ValueError("Compare a matched all-K1 control with a positive patch curriculum.")
    for key in (
        "patch_size",
        "raw_sequence_length",
        "validation",
        "optimizer_transition",
        "concatenate_examples",
    ):
        if key not in baseline or baseline[key] != curriculum.get(key):
            raise ValueError(f"Curriculum comparison has mismatched patch {key}.")
    left = {
        row["optimizer_step"]: row
        for row in control["timing"]["step_records"]
        if row["phase"] == "stable_unprofiled"
    }
    right = {
        row["optimizer_step"]: row
        for row in candidate["timing"]["step_records"]
        if row["phase"] == "stable_unprofiled" and row["curriculum_phase"] == "token"
    }
    indices = sorted(left.keys() & right.keys())
    if not indices:
        raise ValueError("No common stable token-phase optimizer steps.")
    reference = _step_statistics([left[index]["compute_seconds"] for index in indices])[
        "median_seconds"
    ]
    restored = _step_statistics([right[index]["compute_seconds"] for index in indices])[
        "median_seconds"
    ]
    result = {
        "token_recovery": {
            "optimizer_steps": indices,
            "control_median_seconds": reference,
            "curriculum_token_median_seconds": restored,
            "latency_change_percent": 100 * (restored / reference - 1),
            "note": "matched K1, not an untouched ordinary-training baseline",
        },
        "timing": {},
        "memory": {},
        "evidence": "one sequential profiled comparison; not a statistical equivalence claim",
    }
    for key in (
        "train_wall_seconds",
        "train_compute_seconds_sum",
        "train_batch_ready_seconds_sum",
        "validation_wall_seconds",
        "validation_prediction_seconds_sum",
    ):
        before, after = control["timing"][key], candidate["timing"][key]
        if not math.isfinite(before) or not math.isfinite(after) or before <= 0 or after <= 0:
            raise ValueError("Comparison timings must be finite and positive.")
        result["timing"][key] = {
            "control": before,
            "curriculum": after,
            "change_percent": 100 * (after / before - 1),
        }
    for key in (
        "train_peak_allocated_bytes",
        "train_peak_reserved_bytes",
        "validation_peak_allocated_bytes",
        "validation_peak_reserved_bytes",
    ):
        before, after = control["memory"][key], candidate["memory"][key]
        result["memory"][key] = {
            "control": before,
            "curriculum": after,
            "change_bytes": after - before,
            "change_percent": 100 * (after / before - 1) if before else None,
        }
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_python_tree(root: Path) -> dict[str, Any]:
    """Hash Python sources in an imported package without recording its path."""

    entries: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        entries.append(f"{relative}:{_sha256_file(path)}")
    tree_digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return {"python_file_count": len(entries), "python_tree_sha256": tree_digest}


def _cce_provenance(root: Path) -> dict[str, Any]:
    """Return the package tree and critical CCE file hashes without paths."""

    tree = _sha256_python_tree(root)
    tree["tracked_files_sha256"] = {
        relative: _sha256_file(root / relative) for relative in CCE_PROVENANCE_FILES
    }
    return tree


def _load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("neollm_remote_train_10x10", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("training module could not be loaded")
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configure_dynamics_metrics(training: Any, enabled: bool, logging_steps: int) -> dict[str, Any]:
    """Configure opt-in metrics without requiring their API on legacy controls."""
    required = (
        "USE_DYNAMICS_METRICS",
        "DYNAMICS_SAMPLE_TOKENS",
        "DYNAMICS_PARAMETER_SAMPLES",
        "DYNAMICS_INTERVAL",
    )
    missing = [name for name in required if not hasattr(training, name)]
    if enabled and missing:
        raise RuntimeError("training script lacks the complete dynamics API: " + ", ".join(missing))
    supported = not missing
    if hasattr(training, "USE_DYNAMICS_METRICS"):
        training.USE_DYNAMICS_METRICS = bool(enabled)
    if supported:
        training.DYNAMICS_INTERVAL = logging_steps
    return {
        "enabled": bool(enabled),
        "supported": supported,
        "logging_steps": logging_steps,
        "sample_tokens": training.DYNAMICS_SAMPLE_TOKENS if supported else None,
        "parameter_samples_per_tensor": training.DYNAMICS_PARAMETER_SAMPLES if supported else None,
        "parameter_interval": training.DYNAMICS_INTERVAL if supported else None,
        "log_records": [],
        "note": "private observability output; benchmark cadence is explicit, not the production default",
    }


def _load_from_disk(path: Path):
    # Import lazily so the local contract tests do not need the datasets
    # package, and keep the only permitted dataset operation explicit.
    from datasets import load_from_disk

    return load_from_disk(str(path))


def _dataset_keys(dataset: object) -> set[str]:
    keys = getattr(dataset, "keys", None)
    if not callable(keys):
        return set()
    return {str(key) for key in keys()}


def _dataset_columns(dataset: object) -> set[str]:
    columns = getattr(dataset, "column_names", None)
    if isinstance(columns, dict):
        result: set[str] = set()
        for split_columns in columns.values():
            result.update(str(column) for column in split_columns)
        return result
    if columns is None:
        return set()
    return {str(column) for column in columns}


def _prepare_preexisting_splits(
    data_dir: Path | None,
    batch_size: int,
    loader: Callable[[Path], object] | None = None,
    train_data_dir: Path | None = None,
    validation_data_dir: Path | None = None,
    validation_from_train: bool = False,
) -> tuple[object, object]:
    """Load and truncate existing train/validation splits without writing.

    The returned datasets contain exactly the number of rows needed for 100
    batches each.  The caller's ``HFTokenDataset`` and collator still perform
    the actual batch construction in the real training path.
    Explicit validation_from_train selects the next, disjoint training-source
    rows for evaluation; it never silently falls back to training data and is
    not a measurement on the original held-out validation split.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    load = loader or _load_from_disk
    if train_data_dir is not None or validation_data_dir is not None:
        if train_data_dir is None or validation_data_dir is None:
            raise ValueError("both explicit train and validation directories are required")
        if not train_data_dir.is_dir() or not validation_data_dir.is_dir():
            raise FileNotFoundError("pre-existing tokenized split directory not found")
        train_dataset = load(train_data_dir)
        validation_dataset = load(validation_data_dir)
    else:
        if data_dir is None or not data_dir.is_dir():
            raise FileNotFoundError("pre-existing tokenized dataset directory not found")
        train_path = data_dir / "train"
        validation_path = data_dir / "validation"
        if not validation_path.exists():
            validation_path = data_dir / "val"

        if train_path.is_dir() and validation_path.is_dir():
            train_dataset = load(train_path)
            validation_dataset = load(validation_path)
        else:
            dataset = load(data_dir)
            keys = _dataset_keys(dataset)
            if not {"train", "validation"}.issubset(keys):
                if not {"train", "val"}.issubset(keys):
                    raise ValueError(
                        "dataset must contain pre-existing train and validation splits"
                    )
                validation_key = "val"
            else:
                validation_key = "validation"
            train_dataset = dataset["train"]
            validation_dataset = dataset[validation_key]

    needed_rows = REAL_TRAIN_STEPS * batch_size
    needed_validation_rows = REAL_VALIDATION_STEPS * batch_size
    validation_offset = needed_rows if validation_from_train else 0
    if validation_from_train:
        validation_dataset = train_dataset
    for name, dataset, needed in (
        ("train", train_dataset, needed_rows),
        ("validation", validation_dataset, validation_offset + needed_validation_rows),
    ):
        missing = REQUIRED_DATASET_COLUMNS - _dataset_columns(dataset)
        if missing:
            raise ValueError(f"{name} split is missing required token columns")
        if len(dataset) < needed:
            steps = REAL_TRAIN_STEPS if name == "train" else REAL_VALIDATION_STEPS
            raise ValueError(
                f"{name} split has fewer rows than required: available={len(dataset)}, "
                f"required={needed} ({steps} batches of {batch_size}"
                f" plus {validation_offset if name == 'validation' else 0} preceding rows)"
            )

    return (
        train_dataset.select(range(needed_rows)),
        validation_dataset.select(
            range(validation_offset, validation_offset + needed_validation_rows)
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "jtok", "jtokm"), required=True)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=Path)
    data_group.add_argument("--train-data-dir", type=Path)
    parser.add_argument("--validation-data-dir", type=Path)
    parser.add_argument(
        "--validation-from-train",
        action="store_true",
        help="Evaluate the next disjoint training-source rows, not the original held-out split.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, default=Path("/train.py"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-mode", choices=("leviathan", "table"), default="leviathan")
    parser.add_argument("--siamesenorm-backend", choices=("off", "reference", "triton", "cute"))
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--dynamics-metrics", action="store_true")
    parser.add_argument(
        "--no-perplexity-metrics",
        action="store_true",
        help="Matched overhead control only; normal runs collect CCE CE/MiLe PPL.",
    )
    parser.add_argument(
        "--no-topk-metrics",
        action="store_true",
        help="Matched overhead control; disable full-token Top-1/3/5.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--patch-size",
        type=int,
        choices=(4,),
        help="Keep each raw512 sequence; use core128 then core512, with full prewarm.",
    )
    parser.add_argument(
        "--patch-steps",
        type=int,
        default=67,
        help="Completed optimizer-update boundary; zero selects the matched K1 control.",
    )
    parser.add_argument(
        "--control-results",
        type=Path,
        help="Reuse a completed private matched-K1 result for timing/memory/recovery comparison.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Chrome trace path; profiling is diagnostic and not the gate.",
    )
    return parser.parse_args()


def _runtime_snapshot(torch: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        snapshot.update(
            {
                "device": str(torch.cuda.get_device_name()),
                "capability": list(torch.cuda.get_device_capability()),
                "device_count": int(torch.cuda.device_count()),
            }
        )
    return snapshot


def _require_strict_leviathan(model: Any, result: dict[str, Any]) -> Any:
    """Require the model's existing no-fallback guard for the whole run."""

    marked_configs = 0
    for target in (model, getattr(model, "model", None)):
        config = getattr(target, "config", None)
        if config is None:
            continue
        setattr(config, "_require_leviathan_triton", True)
        marked_configs += 1
    if marked_configs == 0:
        raise RuntimeError("model exposes no config for the strict Leviathan guard")
    result["runtime_verification"] = {
        "leviathan_triton_required": True,
        "reference_fallback_allowed": False,
        "marked_config_objects": marked_configs,
    }
    return model


def _validate_leviathan_tile_overrides() -> None:
    """Reporting labels such as 'auto' are not integer kernel overrides."""
    for name in ("LEV_DDELTA_BM", "LEV_DDELTA_BD", "LEV_DDELTA_BR"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be a positive integer or unset") from error
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer or unset")


def _forbid_leviathan_reference_fallbacks(compiler: Any, result: dict[str, Any]) -> None:
    """Fail closed at the actual compiler boundary, not just model readiness."""
    attempts = {"forward": 0, "backward": 0}
    result["leviathan_reference_fallback_attempts"] = attempts

    def forbidden(phase: str):
        def reject(*_args: Any, **_kwargs: Any):
            attempts[phase] += 1
            raise RuntimeError(f"real validation forbids Leviathan reference {phase} fallback")

        return reject

    compiler.leviathan_forward_ref = forbidden("forward")
    compiler.leviathan_backward = forbidden("backward")


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.output.exists() or args.profile.exists():
        raise FileExistsError(
            "Choose fresh private result/profile paths; evidence is never overwritten."
        )
    if any(
        _trace_output_path(args.profile, phase).exists()
        for phase in ("prewarm", "initial", "transition", "validation")
    ):
        raise FileExistsError("A phase trace already exists; choose a fresh private run name.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.control_results is not None and (args.patch_size is None or args.patch_steps <= 0):
        raise ValueError("--control-results requires a positive patch curriculum.")
    control_results = (
        json.loads(args.control_results.read_text(encoding="utf-8"))
        if args.control_results is not None
        else None
    )
    if not 1 <= args.logging_steps <= REAL_TRAIN_STEPS:
        raise ValueError("--logging-steps must be between one and the train step count")
    if (args.train_data_dir is None) != (args.validation_data_dir is None):
        raise ValueError("--train-data-dir and --validation-data-dir must be supplied together")

    # Offline mode is defensive; the loader below still refuses to create or
    # download a dataset.  The tokenizer/model cache must already be present.
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NEOLLM_JTOK_MODE"] = args.mode
    os.environ["NEOLLM_JTOK_KERNEL_BACKEND"] = "triton"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    result: dict[str, Any] = {
        "schema": "neollm-remote-real-100x100-v1",
        "status": "started",
        "mode": args.mode,
        "backend": "triton",
        "provenance": {
            "canonical_repository": CANONICAL_SOURCE_REPOSITORY,
            "canonical_commit": CANONICAL_SOURCE_COMMIT,
        },
        "runner_initial_seed": 1729,
        "train_steps_requested": REAL_TRAIN_STEPS,
        "validation_steps_requested": REAL_VALIDATION_STEPS,
        "batch_size": args.batch_size,
        "sequence_length": 512,
        "data_policy": {
            "source": "preexisting_tokenized_remote_data",
            "download_allowed": False,
            "synthetic_allowed_for_gate": False,
            "validation_source": (
                "disjoint_preexisting_train_rows"
                if args.validation_from_train
                else "preexisting_validation_split"
            ),
            "validation_is_original_heldout_split": not args.validation_from_train,
            "training_row_range": [0, REAL_TRAIN_STEPS * args.batch_size],
            "validation_row_range": [
                REAL_TRAIN_STEPS * args.batch_size,
                (REAL_TRAIN_STEPS + REAL_VALIDATION_STEPS) * args.batch_size,
            ]
            if args.validation_from_train
            else [0, REAL_VALIDATION_STEPS * args.batch_size],
        },
        "kernel_policy": {
            "jtok_route_vjp_factor": os.environ.get("JTOK_ROUTE_VJP_FACTOR", "0"),
            "jtok_compact_spline_vjp": os.environ.get("JTOK_COMPACT_SPLINE_VJP", "0"),
            "jtok_projection_split": os.environ.get("JTOK_PROJECTION_SPLIT", "0"),
            "jtok_sparse_coeff_updates": os.environ.get("JTOK_SPARSE_COEFF_UPDATES", "0"),
            "compact_spline": os.environ.get("LEV_COMPACT_SPLINE", "0"),
            "d_delta_splits": os.environ.get("LEV_DDELTA_SPLITS", "1"),
            "d_delta_block_m": os.environ.get("LEV_DDELTA_BM", "auto"),
            "d_delta_block_d": os.environ.get("LEV_DDELTA_BD", "auto"),
            "d_delta_block_r": os.environ.get("LEV_DDELTA_BR", "auto"),
            "fused_chain_d_delta": os.environ.get("LEV_FUSE_CHAIN_DDELTA", "auto"),
        },
        "timing": {"train_step_seconds": []},
    }
    _write_result(args.output, result)

    timing: dict[str, Any] = {
        "train_step_seconds": [],
        "step_records": [],
        "train_losses": [],
        "eval_prediction_steps": 0,
        "validation_step_seconds": [],
        "perplexity_log_records": [],
    }
    profile_holder: dict[str, Any] = {"profiler": None}

    try:
        import torch
        from torch._dynamo.utils import counters
        from transformers import TrainerCallback

        def compiler_snapshot():
            from torch._dynamo.utils import compilation_time_metrics
            from torch._inductor import metrics
            from torch._inductor.cudagraph_trees import get_manager

            snapshot = {
                "unique_graphs": counters["stats"]["unique_graphs"],
                "graph_breaks": sum(counters["graph_break"].values()),
                "recompiles": sum(counters["recompiles"].values()),
                "generated_inductor_kernels": metrics.generated_kernel_count,
                "compilation_event_counts": {
                    str(k): len(v) for k, v in compilation_time_metrics.items()
                },
                "inductor_counters": dict(counters["inductor"]),
            }
            # Read only an existing manager: observing must not create one.
            manager = get_manager(create_if_none_exists=False)
            if manager is not None:
                pending = [n for nodes in manager.roots.values() for n in nodes]
                seen = set()
                while pending:
                    node = pending.pop()
                    if id(node) in seen:
                        continue
                    seen.add(id(node))
                    pending.extend(n for nodes in node.children.values() for n in nodes)
                snapshot["cuda_graph_tree_nodes"] = len(seen)
                snapshot["cuda_graph_rerecord_count"] = sum(
                    sum(v.values()) for v in manager.num_rerecord.values()
                )
                # Despite its name, debug_fail_counter counts tree replay
                # misses that require a new recording, not capture failures.
                snapshot["cuda_graph_new_recording_events"] = manager.debug_fail_counter
            else:
                snapshot["cuda_graph_tree_nodes"] = None
            # External JIT caches are distinct from Dynamo/Inductor graphs.
            cute_module = sys.modules.get("cut_cross_entropy.siamesenorm._cute")
            snapshot["siamesenorm_cute_jit_entries"] = (
                len(cute_module._CACHE) if cute_module is not None else 0
            )
            snapshot.update(_external_jit_snapshot())
            return snapshot

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the remote real validation")
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)

        training = _load_training_module(args.train_script)
        if not hasattr(training, "USE_PERPLEXITY_METRICS"):
            raise RuntimeError("The training script lacks CCE/MiLe perplexity collection.")
        training.USE_PERPLEXITY_METRICS = not args.no_perplexity_metrics
        if not hasattr(training, "USE_TOPK_METRICS"):
            raise RuntimeError("Private Trainer lacks explicit full-token Top-k configuration.")
        training.USE_TOPK_METRICS = not args.no_topk_metrics
        result["topk_metrics"] = {
            "enabled": not args.no_topk_metrics,
            "coverage": "all_valid_supervised_token_targets",
            "scope": "ordinary_token_phase_and_validation_only",
            "weighting": "unweighted_for_both_ce_and_mile",
        }
        result["perplexity_metrics"] = {
            "enabled": not args.no_perplexity_metrics,
            "source": "cce_ce_and_mile_before_auxiliary_losses",
        }
        result["patch_curriculum"] = _configure_patch_training(
            training, args.patch_size, args.patch_steps
        )
        result["dynamics"] = _configure_dynamics_metrics(
            training, args.dynamics_metrics, args.logging_steps
        )
        result["runtime"] = _runtime_snapshot(torch)
        result["runtime_verification"] = {
            "leviathan_triton_required": True,
            "reference_fallback_allowed": False,
        }
        result["module_hashes"] = {
            "train.py": _sha256_file(args.train_script),
            "modeling_neollm.py": _sha256_file(args.train_script.parent / "modeling_neollm.py"),
            "configuration_neollm.py": _sha256_file(
                args.train_script.parent / "configuration_neollm.py"
            ),
        }

        import cut_cross_entropy

        if args.embedding_mode == "leviathan":
            from cut_cross_entropy.leviathan import compiler as leviathan_compiler

            _validate_leviathan_tile_overrides()
            _forbid_leviathan_reference_fallbacks(leviathan_compiler, result)

        cce_origin = Path(cut_cross_entropy.__file__).resolve()
        from cut_cross_entropy.leviathan.runtime_policy import compact_spline_requested

        projection_candidate = os.environ.get("JTOK_PROJECTION_SPLIT", "0") == "1"
        compact_jtok_candidate = os.environ.get("JTOK_COMPACT_SPLINE_VJP", "0") == "1"
        route_vjp_candidate = os.environ.get("JTOK_ROUTE_VJP_FACTOR", "0") == "1"
        native_jtok_candidate = (
            os.environ.get("JTOK_SPARSE_COEFF_UPDATES", "0") == "1"
            or projection_candidate
            or compact_jtok_candidate
            or route_vjp_candidate
        )

        if (
            compact_spline_requested() or native_jtok_candidate
        ) and args.candidate_manifest is None:
            raise ValueError("candidate validation requires --candidate-manifest")
        if native_jtok_candidate and args.mode == "off":
            raise ValueError("native JTok candidate cannot be claimed when JTok is disabled")
        if args.candidate_manifest is not None:
            verify_candidate(
                cce_origin.parent,
                json.loads(args.candidate_manifest.read_text(encoding="utf-8")),
                require_jtok=native_jtok_candidate,
                require_projection=projection_candidate,
                require_compact_jtok=compact_jtok_candidate,
            )
            result["runtime_verification"]["candidate_sources_verified"] = True
        result["imports"] = {
            "training_module": args.train_script.name,
            "cce_module": cce_origin.name,
            "cce_python_tree": _cce_provenance(cce_origin.parent),
        }
        try:
            import transformers

            result["runtime"]["transformers_version"] = str(transformers.__version__)
        except Exception:
            result["runtime"]["transformers_version"] = None

        # Make accidental calls to the original downloader fail loudly even if
        # a future training-script path calls its module-level symbol.
        def forbid_dataset_download(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("dataset download is disabled by the real validation runner")

        training.load_dataset = forbid_dataset_download

        original_setup_model = training.setup_model

        def strict_setup_model(config_params: dict[str, Any], tokenizer: Any = None):
            params = dict(config_params)
            if args.embedding_mode == "table":
                if args.mode != "off":
                    raise ValueError("table baseline requires JToK disabled")
                params["use_token_generator"] = False
            if args.siamesenorm_backend is not None:
                params["use_siamesenorm"] = args.siamesenorm_backend != "off"
            model = original_setup_model(params, tokenizer=tokenizer)
            result["normalization_comparison"] = {
                "embedding_mode": args.embedding_mode,
                "siamesenorm_backend": args.siamesenorm_backend,
            }
            if args.siamesenorm_backend in ("triton", "cute"):
                if not getattr(model, "_supports_siamesenorm_kernel", False):
                    raise RuntimeError("private model lacks the SiameseNorm kernel integration")
                for module in model.modules():
                    module.siamesenorm_kernel_backend = args.siamesenorm_backend
            if args.embedding_mode == "leviathan":
                return _require_strict_leviathan(model, result)
            result["runtime_verification"]["leviathan_triton_required"] = False
            return model

        training.setup_model = strict_setup_model

        def existing_fineweb(tokenizer: Any, block_size: int = 512, **_kwargs: Any):
            del tokenizer
            if block_size != 512:
                raise ValueError("training sequence length differs from fixed 100+100 contract")
            return _prepare_preexisting_splits(
                args.data_dir,
                args.batch_size,
                train_data_dir=args.train_data_dir,
                validation_data_dir=args.validation_data_dir,
                validation_from_train=args.validation_from_train,
            )

        training.load_and_process_fineweb = existing_fineweb

        training.wandb = type(
            "OfflineWandb",
            (),
            {
                "run": None,
                "init": lambda self, *a, **k: None,
                "log": lambda self, *a, **k: None,
                "finish": lambda self, *a, **k: None,
            },
        )()

        original_setup_args = training.setup_training_args

        def validation_training_args(output_dir: str, run_name: str):
            train_args = original_setup_args(output_dir, run_name)
            if train_args.gradient_accumulation_steps != 1:
                raise ValueError("fixed 100+100 timing currently requires accumulation=1")
            result["seed"] = int(train_args.seed)
            result["data_seed"] = int(train_args.data_seed)
            train_args.max_steps = REAL_TRAIN_STEPS
            train_args.num_train_epochs = 1
            train_args.per_device_train_batch_size = args.batch_size
            train_args.per_device_eval_batch_size = args.batch_size
            train_args.eval_strategy = "no"
            train_args.save_strategy = "no"
            train_args.load_best_model_at_end = False
            train_args.push_to_hub = False
            train_args.report_to = []
            train_args.logging_steps = args.logging_steps
            train_args.logging_first_step = False
            train_args.dataloader_num_workers = 0
            train_args.dataloader_persistent_workers = False
            train_args.dataloader_prefetch_factor = None
            train_args.output_dir = str(args.output.parent / "run_artifacts")
            return train_args

        training.setup_training_args = validation_training_args
        training.create_and_push_model_card = lambda *a, **k: None
        training.save_tokenizer_with_contract = lambda *a, **k: None
        training.AdEMAMixTrainer.save_model = lambda self, *a, **k: None
        result_output_path = args.output

        class TimingCallback(TrainerCallback):
            def __init__(self, trainer=None):
                self.trainer = trainer

            def on_train_begin(self, args, state, control, **kwargs):
                if compact_jtok_candidate:
                    from cut_cross_entropy.leviathan.jtok_compact import prepare_compact_jtok_grids

                    count = prepare_compact_jtok_grids(kwargs["model"])
                    if not count:
                        raise RuntimeError("compact JTok candidate found no grids to prevalidate")
                    result["runtime_verification"]["compact_jtok_grids_prevalidated"] = count
                if result["patch_curriculum"]["enabled"]:
                    trainer = self.trainer
                    indices = (
                        [0, trainer.train_dataset.boundary]
                        if result["patch_curriculum"]["patch_steps"]
                        else [0]
                    )
                    batches = [
                        trainer.data_collator(
                            [
                                trainer.train_dataset[index + row]
                                for row in range(args.per_device_train_batch_size)
                            ]
                        )
                        for index in indices
                    ]
                    evaluation = trainer.data_collator(
                        [
                            trainer.eval_dataset[row]
                            for row in range(args.per_device_eval_batch_size)
                        ]
                    )
                    timing["_prewarming"] = True
                    torch.cuda.synchronize()
                    prewarm_start = time.perf_counter()
                    try:
                        result["prewarm"] = trainer.prewarm_patch_signatures(batches, evaluation)
                    finally:
                        timing["_prewarming"] = False
                    torch.cuda.synchronize()
                    result["prewarm"]["seconds"] = time.perf_counter() - prewarm_start
                    result["prewarm"]["peak_allocated_bytes"] = int(
                        torch.cuda.max_memory_allocated()
                    )
                    result["prewarm"]["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                    result["compiler_after_prewarm"] = compiler_snapshot()
                    profiler = profile_holder.get("profiler")
                    if profiler is not None:
                        profiler.step()  # Save warmup separately, before counted update1.
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timing["train_begin"] = time.perf_counter()
                return control

            def on_step_begin(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                curriculum = result.get("patch_curriculum", {})
                if (
                    curriculum.get("patch_steps", 0)
                    and state.global_step == curriculum["patch_steps"]
                ):
                    result["compiler_before_transition"] = compiler_snapshot()
                    _assert_prewarmed_signatures_stable(
                        result["compiler_after_prewarm"], result["compiler_before_transition"]
                    )
                    result.setdefault("memory", {}).setdefault("by_curriculum_phase", {})[
                        "patch"
                    ] = {
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    }
                    torch.cuda.reset_peak_memory_stats()
                if int(state.global_step) == 4:
                    result["compiler_after_warmup"] = compiler_snapshot()
                    result.setdefault("memory", {}).update(
                        {
                            "cold_profile_peak_allocated_bytes": int(
                                torch.cuda.max_memory_allocated()
                            ),
                            "cold_profile_peak_reserved_bytes": int(
                                torch.cuda.max_memory_reserved()
                            ),
                        }
                    )
                    torch.cuda.reset_peak_memory_stats()
                timing.setdefault("_step_starts", []).append(time.perf_counter())
                return control

            def on_step_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                step_end = time.perf_counter()
                starts = timing.get("_step_starts", [])
                if starts:
                    timing["train_step_seconds"].append(step_end - starts[-1])
                compute_start = timing.pop("_compute_start", None)
                e2e_start = timing.pop("_e2e_start", None)
                if compute_start is None or e2e_start is None:
                    raise RuntimeError("batch-ready compute/e2e timing boundary was not observed")
                step = len(timing["train_step_seconds"])
                curriculum = result.get("patch_curriculum", {})
                patch_steps = curriculum.get("patch_steps", 0)
                record = {
                    "optimizer_step": step,
                    "phase": _profile_phase(step, patch_steps),
                    "compute_seconds": step_end - compute_start,
                    "e2e_seconds": step_end - e2e_start,
                }
                if curriculum.get("enabled"):
                    record.update(
                        {
                            "curriculum_phase": "patch" if step <= patch_steps else "token",
                            "patch_size": curriculum["patch_size"] if step <= patch_steps else 1,
                        }
                    )
                    if patch_steps and step == patch_steps + 1:
                        result["compiler_after_transition"] = compiler_snapshot()
                        _assert_prewarmed_signatures_stable(
                            result["compiler_after_prewarm"], result["compiler_after_transition"]
                        )
                timing["step_records"].append(record)
                loss = timing.pop("_loss_snapshot").item()
                if not math.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                timing["train_losses"].append(float(loss))
                result.setdefault("memory", {}).update(
                    {
                        "train_peak_allocated_bytes": max(
                            result.get("memory", {}).get("train_peak_allocated_bytes", 0),
                            int(torch.cuda.max_memory_allocated()),
                        ),
                        "train_peak_reserved_bytes": max(
                            result.get("memory", {}).get("train_peak_reserved_bytes", 0),
                            int(torch.cuda.max_memory_reserved()),
                        ),
                    }
                )
                record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                record["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                result["timing"]["step_records"] = timing["step_records"]
                _write_result(result_output_path, result)
                profiler = profile_holder.get("profiler")
                if profiler is not None:
                    profiler.step()
                return control

            def on_prediction_step(self, args, state, control, **kwargs):
                timing["eval_prediction_steps"] += 1
                profiler = profile_holder.get("profiler")
                if profiler is not None and not timing.get("_prewarming"):
                    profiler.step()
                return control

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                torch.cuda.synchronize()
                timing["eval_metrics"] = {
                    str(key): float(value)
                    for key, value in (metrics or {}).items()
                    if isinstance(value, (int, float))
                }
                if any(not math.isfinite(value) for value in timing["eval_metrics"].values()):
                    raise RuntimeError("nonfinite validation metric")
                return control

            def on_train_end(self, args, state, control, **kwargs):
                result["compiler_after_train"] = compiler_snapshot()
                if "compiler_after_prewarm" in result:
                    _assert_prewarmed_signatures_stable(
                        result["compiler_after_prewarm"], result["compiler_after_train"]
                    )
                torch.cuda.synchronize()
                timing["train_end"] = time.perf_counter()
                result.setdefault("memory", {}).update(
                    {
                        "stable_train_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "stable_train_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    }
                )
                if result.get("patch_curriculum", {}).get("enabled"):
                    result["memory"].setdefault("by_curriculum_phase", {})["token"] = {
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    }
                torch.cuda.reset_peak_memory_stats()
                return control

        original_trainer_init = training.AdEMAMixTrainer.__init__

        def timed_trainer_init(self, *init_args, **init_kwargs):
            original_trainer_init(self, *init_args, **init_kwargs)
            original_batches = self.get_batch_samples
            original_step = self.training_step
            original_loss = self.compute_loss
            original_log = self.log
            original_prediction_step = self.prediction_step
            original_evaluation_loop = self.evaluation_loop

            def measured_log(logs, *log_args, **log_kwargs):
                is_training_step = "loss" in logs
                if is_training_step:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                output = original_log(logs, *log_args, **log_kwargs)
                if is_training_step:
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    record = timing["step_records"][-1]
                    record["logging_seconds"] = seconds
                    record["compute_plus_logging_seconds"] = record["compute_seconds"] + seconds
                    snapshot = {
                        name: float(value)
                        for name, value in logs.items()
                        if name.startswith("dynamics/")
                    }
                    timing["perplexity_log_records"].append(
                        {
                            "optimizer_step": record["optimizer_step"],
                            "metrics": {
                                name: float(value)
                                for name, value in logs.items()
                                if name.startswith("train_")
                            },
                        }
                    )
                    if not args.no_perplexity_metrics:
                        records = timing["perplexity_log_records"]
                        previous = records[-2]["optimizer_step"] if len(records) > 1 else 0
                        _validate_perplexity_record(
                            records[-1]["metrics"],
                            previous + 1,
                            record["optimizer_step"],
                            args.batch_size,
                            result["patch_curriculum"]["patch_steps"],
                        )
                    result["dynamics"]["log_records"].append(
                        {
                            "optimizer_step": record["optimizer_step"],
                            "logging_seconds": seconds,
                            "metrics": snapshot,
                        }
                    )
                return output

            def measured_prediction_step(*prediction_args, **prediction_kwargs):
                if timing.get("_prewarming"):
                    return original_prediction_step(*prediction_args, **prediction_kwargs)
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = original_prediction_step(*prediction_args, **prediction_kwargs)
                torch.cuda.synchronize()
                timing["validation_step_seconds"].append(time.perf_counter() - start)
                return output

            def measured_evaluation_loop(*evaluation_args, **evaluation_kwargs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = original_evaluation_loop(*evaluation_args, **evaluation_kwargs)
                torch.cuda.synchronize()
                timing["validation_wall_seconds"] = time.perf_counter() - start
                return output

            def measured_batches(*batch_args, **batch_kwargs):
                torch.cuda.synchronize()
                timing["_e2e_start"] = time.perf_counter()
                return original_batches(*batch_args, **batch_kwargs)

            def measured_step(*step_args, **step_kwargs):
                timing["_inside_training_step"] = True
                try:
                    return original_step(*step_args, **step_kwargs)
                finally:
                    timing["_inside_training_step"] = False

            def measured_loss(*loss_args, **loss_kwargs):
                if timing.get("_inside_training_step") and not timing.get("_prewarming"):
                    if "_compute_start" in timing:
                        raise RuntimeError(
                            "multiple loss calls in one fixed-contract optimizer step"
                        )
                    # Trainer has placed/prepared inputs before compute_loss.
                    # Synchronize outstanding H2D before the wall-clock compute boundary.
                    torch.cuda.synchronize()
                    timing["_compute_start"] = time.perf_counter()
                output = original_loss(*loss_args, **loss_kwargs)
                if timing.get("_inside_training_step") and not timing.get("_prewarming"):
                    loss = output[0] if isinstance(output, tuple) else output
                    # Own a tiny scalar copy: cudagraph replay may reuse output storage.
                    timing["_loss_snapshot"] = loss.detach().float().clone()
                return output

            self.get_batch_samples = measured_batches
            self.training_step = measured_step
            self.compute_loss = measured_loss
            self.log = measured_log
            self.prediction_step = measured_prediction_step
            self.evaluation_loop = measured_evaluation_loop
            self.add_callback(TimingCallback(self))

        training.AdEMAMixTrainer.__init__ = timed_trainer_init

        profile_context = nullcontext()
        if args.profile is not None:
            args.profile.parent.mkdir(parents=True, exist_ok=True)

            def trace_handler(profiler: Any) -> None:
                index = len(result["profile"]["outputs"])
                labels = ["initial"]
                if result["patch_curriculum"]["enabled"]:
                    labels = ["prewarm", "initial"]
                    if result["patch_curriculum"]["patch_steps"]:
                        labels.append("transition")
                labels.append("validation")
                label = labels[index]
                path = (
                    args.profile
                    if label == "initial" and not result["patch_curriculum"]["enabled"]
                    else _trace_output_path(args.profile, label)
                )
                if path.exists():
                    raise FileExistsError("Refusing to overwrite a private execution trace.")
                profiler.export_chrome_trace(str(path))
                result["profile"]["outputs"].append({"phase": label, "output_name": path.name})

            def profile_schedule(index):
                prewarm = result["patch_curriculum"]["enabled"]
                if prewarm and index == 0:
                    return torch.profiler.ProfilerAction.RECORD_AND_SAVE
                step = index if prewarm else index + 1
                if step > REAL_TRAIN_STEPS:
                    validation_step = step - REAL_TRAIN_STEPS
                    return {
                        2: torch.profiler.ProfilerAction.WARMUP,
                        3: torch.profiler.ProfilerAction.RECORD,
                        4: torch.profiler.ProfilerAction.RECORD_AND_SAVE,
                    }.get(validation_step, torch.profiler.ProfilerAction.NONE)
                phase = _profile_phase(step, result["patch_curriculum"]["patch_steps"])
                if "warmup" in phase:
                    return torch.profiler.ProfilerAction.WARMUP
                if "active" in phase:
                    last = step in (4, result["patch_curriculum"]["patch_steps"] + 2)
                    return (
                        torch.profiler.ProfilerAction.RECORD_AND_SAVE
                        if last
                        else torch.profiler.ProfilerAction.RECORD
                    )
                return torch.profiler.ProfilerAction.NONE

            profile_context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=profile_schedule,
                on_trace_ready=trace_handler,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            )
            result["profile"] = {
                "requested": True,
                "diagnostic_only": True,
                "output_name": args.profile.name,
                "outputs": [],
            }
        else:
            result["profile"] = {"requested": False, "diagnostic_only": True}

        with profile_context as profiler:
            profile_holder["profiler"] = profiler
            training.main()

        result["compiler_after_validation"] = compiler_snapshot()
        if "compiler_after_prewarm" in result:
            _assert_prewarmed_signatures_stable(
                result["compiler_after_prewarm"], result["compiler_after_validation"]
            )

        if len(timing["train_step_seconds"]) != REAL_TRAIN_STEPS:
            raise RuntimeError("real validation completed an unexpected train step count")
        if timing["eval_prediction_steps"] != REAL_VALIDATION_STEPS:
            raise RuntimeError("real validation completed an unexpected eval step count")
        if args.candidate_manifest is not None:
            verify_candidate(
                cce_origin.parent,
                json.loads(args.candidate_manifest.read_text(encoding="utf-8")),
                require_jtok=native_jtok_candidate,
                require_projection=projection_candidate,
                require_compact_jtok=compact_jtok_candidate,
            )

        result["status"] = "ok"
        result.setdefault("memory", {}).update(
            {
                "validation_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "validation_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
        if args.dynamics_metrics:
            snapshots = result["dynamics"]["log_records"]
            if len(snapshots) != REAL_TRAIN_STEPS // args.logging_steps or any(
                not record["metrics"] for record in snapshots
            ):
                raise RuntimeError(
                    "dynamics metrics were not exposed at the requested logging cadence"
                )
            for record in snapshots:
                observed = record["metrics"]
                if any(not math.isfinite(value) for value in observed.values()):
                    raise RuntimeError("nonfinite dynamics observation")
                prefix = "dynamics/leviathan/"
                if sum(name.startswith(prefix) for name in observed) < 3:
                    raise RuntimeError("missing Leviathan dynamics metrics")
                if args.mode != "off" and not any(
                    name.startswith("dynamics/" + args.mode + "/") and name.endswith("usage_index")
                    for name in observed
                ):
                    raise RuntimeError("missing surface usage metric")
        train_step_median = statistics.median(timing["train_step_seconds"])
        train_step_median_after_first = statistics.median(timing["train_step_seconds"][1:])
        result["timing"] = {
            "train_step_seconds": timing["train_step_seconds"],
            "train_step_median_seconds": train_step_median,
            "train_step_median_after_first_seconds": train_step_median_after_first,
            "train_steps_per_second": 1.0 / train_step_median,
            "train_steps_per_second_after_first": 1.0 / train_step_median_after_first,
            "train_steps_completed": len(timing["train_step_seconds"]),
            "train_wall_seconds": timing["train_end"] - timing["train_begin"],
            "train_compute_seconds_sum": sum(
                record["compute_seconds"] for record in timing["step_records"]
            ),
            "train_batch_ready_seconds_sum": sum(
                record["e2e_seconds"] for record in timing["step_records"]
            ),
            "validation_steps_completed": timing["eval_prediction_steps"],
            "validation_wall_seconds": timing["validation_wall_seconds"],
            "validation_step_seconds": timing["validation_step_seconds"],
            "validation_prediction_seconds_sum": sum(timing["validation_step_seconds"]),
            "validation_prediction": _step_statistics(timing["validation_step_seconds"]),
            "perplexity_log_records": timing["perplexity_log_records"],
            "eval_metrics": timing.get("eval_metrics", {}),
            "train_losses": timing["train_losses"],
            "step_records": timing["step_records"],
            "stable": _stable_step_statistics(timing["step_records"]),
            "instrumentation": "host_cuda_sync_at_batch_fetch_compute_start_and_optimizer_end",
        }
        stable_records = [r for r in timing["step_records"] if r["phase"] == "stable_unprofiled"]
        result["timing"]["stable"]["compute_plus_logging"] = _step_statistics(
            [r.get("compute_plus_logging_seconds", r["compute_seconds"]) for r in stable_records]
        )
        result["timing"]["stable"]["logging_events"] = _step_statistics(
            [r["logging_seconds"] for r in stable_records if "logging_seconds" in r]
        )
        if result["patch_curriculum"]["enabled"]:
            for phase in ("patch", "token"):
                rows = [
                    record
                    for record in timing["step_records"]
                    if record["curriculum_phase"] == phase
                ]
                if rows:
                    result["memory"].setdefault("by_curriculum_phase", {})[phase] = {
                        key: max(row[key] for row in rows)
                        for key in ("peak_allocated_bytes", "peak_reserved_bytes")
                    }
            result["timing"]["by_curriculum_phase"] = {
                phase: _stable_step_statistics(
                    [
                        record
                        for record in timing["step_records"]
                        if record["curriculum_phase"] == phase
                    ]
                )
                for phase in ("patch", "token")
                if any(
                    record["curriculum_phase"] == phase and record["phase"] == "stable_unprofiled"
                    for record in timing["step_records"]
                )
            }
        if control_results is not None:
            result["matched_control_comparison"] = _compare_curriculum_runs(control_results, result)
    except BaseException as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error_message"] = _redact_text(error)
        result["timing"] = {
            "train_step_seconds": timing["train_step_seconds"],
            "train_steps_completed": len(timing["train_step_seconds"]),
            "validation_steps_completed": timing["eval_prediction_steps"],
            "step_records": timing["step_records"],
            "train_losses": timing["train_losses"],
        }
        _write_result(args.output, result)
        raise
    finally:
        result.setdefault("timing", {})
        result["timing"].setdefault("train_steps_completed", len(timing["train_step_seconds"]))
        result["timing"].setdefault("validation_steps_completed", timing["eval_prediction_steps"])
        _write_result(args.output, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
