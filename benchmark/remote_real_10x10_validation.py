"""Run the real NeoLLM path for the fixed 10+10 validation contract.

This runner is deliberately read-only with respect to datasets.  It loads an
already-tokenized Hugging Face dataset with ``load_from_disk`` and never calls
``load_dataset``.  A missing or undersized dataset is an error, not a reason to
download one or silently switch to synthetic data.

The runner imports the supplied training script and patches only the dataset
loader and short-run controls.  Model construction, Leviathan/JToK dispatch,
losses, ``torch.compile(max-autotune)``, the optimizer, and the other training
policy remain in the supplied training script.  The short run is exactly ten
optimizer steps followed by evaluation over exactly ten batches.  Synthetic
inputs belong in a separate edge-case probe and are not accepted here.

The JSON result intentionally contains no connection details, credentials,
personal paths, command lines, or raw exception tracebacks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

REAL_TRAIN_STEPS = 10
REAL_VALIDATION_STEPS = 10
REQUIRED_DATASET_COLUMNS = frozenset(("input_ids", "attention_mask"))
CANONICAL_SOURCE_REPOSITORY = "ml-cross-entropy-jtok-pr"
CANONICAL_SOURCE_COMMIT = "5a4072d"
CCE_PROVENANCE_FILES = (
    "leviathan/backward_dot_kernels.py",
    "leviathan/backward_kernels.py",
    "leviathan/jtok.py",
)


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
) -> tuple[object, object]:
    """Load and truncate existing train/validation splits without writing.

    The returned datasets contain exactly the number of rows needed for ten
    batches each.  The caller's ``HFTokenDataset`` and collator still perform
    the actual batch construction in the real training path.
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
    for name, dataset, needed in (
        ("train", train_dataset, needed_rows),
        ("validation", validation_dataset, needed_validation_rows),
    ):
        missing = REQUIRED_DATASET_COLUMNS - _dataset_columns(dataset)
        if missing:
            raise ValueError(f"{name} split is missing required token columns")
        if len(dataset) < needed:
            raise ValueError(f"{name} split has fewer rows than the fixed 10-batch run")

    return (
        train_dataset.select(range(needed_rows)),
        validation_dataset.select(range(needed_validation_rows)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "jtok", "jtokm"), required=True)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=Path)
    data_group.add_argument("--train-data-dir", type=Path)
    parser.add_argument("--validation-data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, default=Path("/train.py"))
    parser.add_argument("--batch-size", type=int, default=64)
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


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
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
        "schema": "neollm-remote-real-10x10-v1",
        "status": "started",
        "mode": args.mode,
        "backend": "triton",
        "provenance": {
            "canonical_repository": CANONICAL_SOURCE_REPOSITORY,
            "canonical_commit": CANONICAL_SOURCE_COMMIT,
        },
        "seed": 1729,
        "train_steps_requested": REAL_TRAIN_STEPS,
        "validation_steps_requested": REAL_VALIDATION_STEPS,
        "batch_size": args.batch_size,
        "sequence_length": 512,
        "data_policy": {
            "source": "preexisting_tokenized_remote_data",
            "download_allowed": False,
            "synthetic_allowed_for_gate": False,
        },
        "kernel_policy": {
            "d_delta_splits": os.environ.get("LEV_DDELTA_SPLITS", "1"),
            "d_delta_block_m": os.environ.get("LEV_DDELTA_BM", "auto"),
            "d_delta_block_d": os.environ.get("LEV_DDELTA_BD", "auto"),
            "d_delta_block_r": os.environ.get("LEV_DDELTA_BR", "auto"),
            "fused_chain_d_delta": os.environ.get(
                "LEV_FUSE_CHAIN_DDELTA", "auto"
            ),
        },
        "timing": {"train_step_seconds": []},
    }
    _write_result(args.output, result)

    timing: dict[str, Any] = {
        "train_step_seconds": [],
        "eval_prediction_steps": 0,
    }
    profile_holder: dict[str, Any] = {"profiler": None}

    try:
        import torch
        from transformers import TrainerCallback

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the remote real validation")
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)

        training = _load_training_module(args.train_script)
        result["runtime"] = _runtime_snapshot(torch)
        result["runtime_verification"] = {
            "leviathan_triton_required": True,
            "reference_fallback_allowed": False,
        }
        result["module_hashes"] = {
            "train.py": _sha256_file(args.train_script),
            "modeling_neollm.py": _sha256_file(
                args.train_script.parent / "modeling_neollm.py"
            ),
            "configuration_neollm.py": _sha256_file(
                args.train_script.parent / "configuration_neollm.py"
            ),
        }

        import cut_cross_entropy

        cce_origin = Path(cut_cross_entropy.__file__).resolve()
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
            raise RuntimeError("dataset download is disabled by the 10x10 runner")

        training.load_dataset = forbid_dataset_download

        original_setup_model = training.setup_model

        def strict_setup_model(config_params: dict[str, Any], tokenizer: Any = None):
            model = original_setup_model(config_params, tokenizer=tokenizer)
            return _require_strict_leviathan(model, result)

        training.setup_model = strict_setup_model

        def existing_fineweb(tokenizer: Any, block_size: int = 512, **_kwargs: Any):
            del tokenizer, block_size
            return _prepare_preexisting_splits(
                args.data_dir,
                args.batch_size,
                train_data_dir=args.train_data_dir,
                validation_data_dir=args.validation_data_dir,
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
            train_args.max_steps = REAL_TRAIN_STEPS
            train_args.num_train_epochs = 1
            train_args.per_device_train_batch_size = args.batch_size
            train_args.per_device_eval_batch_size = args.batch_size
            train_args.eval_strategy = "no"
            train_args.save_strategy = "no"
            train_args.load_best_model_at_end = False
            train_args.push_to_hub = False
            train_args.report_to = []
            train_args.logging_steps = 1
            train_args.logging_first_step = True
            train_args.dataloader_num_workers = 0
            train_args.dataloader_persistent_workers = False
            train_args.dataloader_prefetch_factor = None
            train_args.output_dir = str(args.output.parent / "run_artifacts")
            return train_args

        training.setup_training_args = validation_training_args
        training.create_and_push_model_card = lambda *a, **k: None
        training.save_tokenizer_with_contract = lambda *a, **k: None
        training.AdEMAMixTrainer.save_model = lambda self, *a, **k: None

        class TimingCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timing["train_begin"] = time.perf_counter()
                return control

            def on_step_begin(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                timing.setdefault("_step_starts", []).append(time.perf_counter())
                return control

            def on_step_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                starts = timing.get("_step_starts", [])
                if starts:
                    timing["train_step_seconds"].append(
                        time.perf_counter() - starts[-1]
                    )
                profiler = profile_holder.get("profiler")
                if profiler is not None:
                    profiler.step()
                return control

            def on_prediction_step(self, args, state, control, **kwargs):
                timing["eval_prediction_steps"] += 1
                return control

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                torch.cuda.synchronize()
                timing["eval_metrics"] = {
                    str(key): float(value)
                    for key, value in (metrics or {}).items()
                    if isinstance(value, (int, float))
                }
                return control

            def on_train_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                timing["train_end"] = time.perf_counter()
                return control

        original_trainer_init = training.AdEMAMixTrainer.__init__

        def timed_trainer_init(self, *init_args, **init_kwargs):
            original_trainer_init(self, *init_args, **init_kwargs)
            self.add_callback(TimingCallback())

        training.AdEMAMixTrainer.__init__ = timed_trainer_init

        profile_context = nullcontext()
        if args.profile is not None:
            args.profile.parent.mkdir(parents=True, exist_ok=True)

            def trace_handler(profiler: Any) -> None:
                profiler.export_chrome_trace(str(args.profile))

            profile_context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                on_trace_ready=trace_handler,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            )
            result["profile"] = {
                "requested": True,
                "diagnostic_only": True,
                "output_name": args.profile.name,
            }
        else:
            result["profile"] = {"requested": False, "diagnostic_only": True}

        with profile_context as profiler:
            profile_holder["profiler"] = profiler
            training.main()

        if len(timing["train_step_seconds"]) != REAL_TRAIN_STEPS:
            raise RuntimeError("real validation completed an unexpected train step count")
        if timing["eval_prediction_steps"] != REAL_VALIDATION_STEPS:
            raise RuntimeError("real validation completed an unexpected eval step count")

        result["status"] = "ok"
        train_step_median = statistics.median(timing["train_step_seconds"])
        train_step_median_after_first = statistics.median(
            timing["train_step_seconds"][1:]
        )
        result["timing"] = {
            "train_step_seconds": timing["train_step_seconds"],
            "train_step_median_seconds": train_step_median,
            "train_step_median_after_first_seconds": train_step_median_after_first,
            "train_steps_per_second": 1.0 / train_step_median,
            "train_steps_per_second_after_first": 1.0 / train_step_median_after_first,
            "train_steps_completed": len(timing["train_step_seconds"]),
            "validation_steps_completed": timing["eval_prediction_steps"],
            "eval_metrics": timing.get("eval_metrics", {}),
        }
    except BaseException as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error_message"] = _redact_text(error)
        result["timing"] = {
            "train_step_seconds": timing["train_step_seconds"],
            "train_steps_completed": len(timing["train_step_seconds"]),
            "validation_steps_completed": timing["eval_prediction_steps"],
        }
        _write_result(args.output, result)
        raise
    finally:
        result.setdefault("timing", {})
        result["timing"].setdefault(
            "train_steps_completed", len(timing["train_step_seconds"])
        )
        result["timing"].setdefault(
            "validation_steps_completed", timing["eval_prediction_steps"]
        )
        _write_result(args.output, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
