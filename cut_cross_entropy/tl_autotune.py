# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
import heapq
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import triton
from triton import Config, cdiv
from triton.runtime import autotuner, driver
from triton.testing import (
    get_dram_gbps,
    get_max_simd_tflops,
    get_max_tensorcore_tflops,
)

_AUTOTUNE: bool = os.getenv("CCE_AUTOTUNE", "0") != "0"

# Smallest B/V tiles in the autotune search space. Locks use this granularity
# so different candidate tiles never serialize on an unrelated lock.
CCE_LOCK_BLOCK_B = 16
CCE_LOCK_BLOCK_V = 32


@dataclass
class NoneSupportRestorer:
    reset_idx_or_name: list[int | str]
    restore_idx_or_name: list[int | str]
    _restore_copies: dict[str | int, torch.Tensor | None] = field(default_factory=dict, init=False)

    def pre_hook(
        self,
        args: list[torch.Tensor | None | Any] | dict[str, torch.Tensor | None | Any],
        reset_only: bool = False,
    ) -> None:
        for i in self.reset_idx_or_name:
            if isinstance(i, str):
                assert isinstance(args, dict)
                v = args[i]
            else:
                assert isinstance(args, list)
                v = args[i]

            if v is not None:
                assert isinstance(v, torch.Tensor)
                v.zero_()

        if not reset_only:
            for i in self.restore_idx_or_name:
                if isinstance(i, str):
                    assert isinstance(args, dict)
                    v = args[i]
                else:
                    assert isinstance(args, list)
                    v = args[i]

                if v is not None:
                    assert isinstance(v, torch.Tensor)
                    self._restore_copies[i] = v.clone()
                else:
                    self._restore_copies[i] = None

    def post_hook(
        self,
        args: list[torch.Tensor | None | Any] | dict[str, torch.Tensor | None | Any],
        exception=None,
    ) -> None:
        for i, old_v in self._restore_copies.items():
            if isinstance(i, str):
                assert isinstance(args, dict)
                v = args[i]
            else:
                assert isinstance(args, list)
                v = args[i]

            if v is not None:
                assert isinstance(v, torch.Tensor)
                assert old_v is not None

                v.copy_(old_v)

        self._restore_copies = {}


@functools.wraps(triton.autotune)
def _cce_autotune(*args, **kwargs) -> Callable[..., autotuner.Autotuner]:
    def decorator(fn):
        reset_idx_or_name = kwargs.pop("reset_to_zero", [])
        restore_idx_or_name = kwargs.pop("restore_value", [])

        restorer = NoneSupportRestorer(reset_idx_or_name, restore_idx_or_name)
        if len(reset_idx_or_name) > 0:
            kwargs["pre_hook"] = restorer.pre_hook

        if len(restore_idx_or_name) > 0:
            kwargs["post_hook"] = restorer.post_hook

        return triton.autotune(*args, **kwargs)(fn)

    return decorator


@functools.lru_cache()
def get_clock_rate_in_khz(device: int) -> int:
    # Triton's driver properties already refer to the active logical CUDA
    # device, including CUDA_VISIBLE_DEVICES remapping. Avoid nvidia-smi/NVML
    # physical index 0 and keep the cache isolated per device.
    return driver.active.utils.get_device_properties(device)["sm_clock_rate"]


def get_tensorcore_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = (
        driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    )  # on recent GPUs
    tflops = (
        min(num_subcores, total_warps)
        / num_subcores
        * get_max_tensorcore_tflops(dtype, get_clock_rate_in_khz(device), device)
    )
    return tflops


def get_simd_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = (
        driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    )  # on recent GPUs
    tflops = (
        min(num_subcores, total_warps)
        / num_subcores
        * get_max_simd_tflops(dtype, get_clock_rate_in_khz(device), device)
    )
    return tflops


def get_tflops(device, num_ctas, num_warps, dtype):
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8 and dtype == torch.float32:
        return get_simd_tflops(device, num_ctas, num_warps, dtype)
    return get_tensorcore_tflops(device, num_ctas, num_warps, dtype)


def early_config_prune(
    configs,
    named_args,
    *,
    shared_memory_factor: float = 1.0,
    max_num_warps: int | None = None,
    **kwargs,
):
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability()
    # BLOCK_B, BLOCK_V, BLOCK_D, SPLIT_K, num_warps, num_stages
    dtsize = named_args["E"].element_size()

    if max_num_warps is not None:
        configs = [config for config in configs if config.num_warps <= max_num_warps]

    # 1. make sure we have enough smem
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_B, BLOCK_V, BLOCK_D, num_stages = (
            kw["BLOCK_B"],
            kw["BLOCK_V"],
            kw["BLOCK_D"],
            config.num_stages,
        )

        max_shared_memory = driver.active.utils.get_device_properties(device)["max_shared_mem"]
        required_shared_memory = (
            shared_memory_factor * (BLOCK_B + BLOCK_V) * BLOCK_D * num_stages * dtsize
        )
        if required_shared_memory > max_shared_memory:
            continue

        pruned_configs.append(config)

    configs = pruned_configs

    # group configs by (BLOCK_B,_N,_K, num_warps)
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_B, BLOCK_V, BLOCK_D, num_warps, num_stages = (
            kw["BLOCK_B"],
            kw["BLOCK_V"],
            kw["BLOCK_D"],
            config.num_warps,
            config.num_stages,
        )

        key = (BLOCK_B, BLOCK_V, BLOCK_D, num_warps)
        if key in configs_map:
            configs_map[key].append((config, num_stages))
        else:
            configs_map[key] = [(config, num_stages)]

    pruned_configs = []
    for k, v in configs_map.items():
        BLOCK_B, BLOCK_V, BLOCK_D, num_warps = k
        if capability[0] >= 8:
            # Approximate the recent NVIDIA tensor-core pipeline. Final
            # selection is still measured; this only prunes stage duplicates.
            mmas = BLOCK_B * BLOCK_V * BLOCK_D / (16 * 8 * 16)
            mma_cycles = mmas / min(4, num_warps) * 8

            ldgsts_latency = 300  # Does this matter?
            optimal_num_stages = ldgsts_latency / mma_cycles

            # nearest stages, prefer large #stages
            nearest = heapq.nsmallest(
                2,
                v,
                key=lambda x: (
                    10 + abs(x[1] - optimal_num_stages)
                    if (x[1] - optimal_num_stages) < 0
                    else x[1] - optimal_num_stages
                ),
            )

            for n in nearest:
                pruned_configs.append(n[0])
        else:  # Volta & Turing only supports num_stages <= 2
            random_config = v[0][0]
            random_config.num_stages = 2
            pruned_configs.append(random_config)
    return pruned_configs


def _total_ops_fn(B, V, D) -> float:
    return 2 * B * V * D + 10 * B * V


def _total_store_fn(B, V, D, dtsize, num_cta_b, num_cta_v):
    return B * dtsize


def estimate_matmul_time(
    # backend, device,
    num_warps,
    num_stages,  #
    E,
    B,
    V,
    D,  #
    BLOCK_B,
    BLOCK_V,
    BLOCK_D,
    debug=False,
    total_ops_fn=_total_ops_fn,
    total_store_fn=_total_store_fn,
    **kwargs,  #
):
    """return estimated running time in ms
    = max(compute, loading) + store"""
    device = torch.cuda.current_device()
    dtype = E.dtype
    dtsize = E.element_size()

    num_cta_b = cdiv(B, BLOCK_B)
    num_cta_v = cdiv(V, BLOCK_V)
    num_ctas = num_cta_b * num_cta_v

    # If the input is smaller than the block size
    B, V = max(B, BLOCK_B), max(V, BLOCK_V)

    # time to compute
    total_ops = total_ops_fn(B, V, D)
    total_ops = total_ops / (1024 * 1024 * 1024)  # GOPS
    tput = get_tflops(device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput

    # time to load data
    num_sm = driver.active.utils.get_device_properties(device)["multiprocessor_count"]
    active_cta_ratio = min(1, num_ctas / num_sm)
    bw_saturation_ctas = max(1, num_sm // 3)
    active_cta_ratio_bw1 = min(1, num_ctas / bw_saturation_ctas)
    active_cta_ratio_bw2 = max(
        min(1, (num_ctas - bw_saturation_ctas) / max(1, num_sm - bw_saturation_ctas)), 0
    )
    dram_bw = get_dram_gbps(device) * (
        active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05
    )  # in GB/s
    l2_bw = dram_bw * 4  # rough estimation (should be 4.7 for A100?)
    # assume 80% of (following) loads are in L2 cache
    load_a_dram = B * D * dtsize * (1 + 0.2 * (num_cta_v - 1))
    load_a_l2 = B * D * dtsize * 0.8 * (num_cta_v - 1)
    load_b_dram = V * D * dtsize * (1 + 0.2 * (num_cta_b - 1))
    load_b_l2 = V * D * dtsize * 0.8 * (num_cta_b - 1)
    # total
    total_dram = (load_a_dram + load_b_dram) / (1024 * 1024)  # MB
    total_l2 = (load_a_l2 + load_b_l2) / (1024 * 1024)
    # loading time in ms
    load_ms = total_dram / dram_bw + total_l2 / l2_bw

    # estimate storing time
    store_bw = dram_bw * 0.4  # :o
    store_dram = total_store_fn(B, V, D, dtsize, num_cta_b, num_cta_v) / (1024 * 1024)
    store_ms = store_dram / store_bw

    total_time_ms = max(compute_ms, load_ms) + store_ms
    if debug:
        print(
            f"{BLOCK_B=}, {BLOCK_V=}, {BLOCK_D=}, {num_warps=}, {num_stages=}, "
            f"Total time: {total_time_ms}ms, compute time: {compute_ms}ms, "
            f"loading time: {load_ms}ms, store time: {store_ms}ms, "
            f"Activate CTAs: {active_cta_ratio * 100}%"
        )
    return total_time_ms


def get_autotune_config() -> list[Config]:
    """Curated tensor-core tile families for Triton 3.4+ GPUs.

    The previous Cartesian product contained 103 candidates, including many
    redundant I/O-bound variants. The performance model only promoted a small
    subset, so keep those shape families and vary scheduling where it matters.
    """
    return [
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 128},
            num_stages=2,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 256, "BLOCK_D": 32},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_B": 256, "BLOCK_V": 128, "BLOCK_D": 32},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_B": 256, "BLOCK_V": 64, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 256, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32},
            num_stages=3,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32},
            num_stages=4,
            num_warps=8,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 64, "BLOCK_D": 32},
            num_stages=3,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 128, "BLOCK_D": 32},
            num_stages=3,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 32, "BLOCK_D": 32},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 64, "BLOCK_D": 64},
            num_stages=3,
            num_warps=4,
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 128, "BLOCK_D": 64},
            num_stages=3,
            num_warps=4,
        ),
        Config({"BLOCK_B": 64, "BLOCK_V": 32, "BLOCK_D": 32}, num_stages=3, num_warps=2),
        Config({"BLOCK_B": 32, "BLOCK_V": 128, "BLOCK_D": 32}, num_stages=3, num_warps=4),
        Config({"BLOCK_B": 16, "BLOCK_V": 128, "BLOCK_D": 32}, num_stages=3, num_warps=4),
    ]


def _heuristics_from_config(
    config: Config,
    fp32_config: Config | None = None,
    arg_name: str | None = None,
    *,
    adaptive_block_b: bool = False,
) -> Callable[..., autotuner.Heuristics]:
    def block_b(args, default: int, fp32_default: int | None) -> int:
        if fp32_default is not None and arg_name is not None:
            if args[arg_name].dtype == torch.float32:
                return fp32_default
        if not adaptive_block_b:
            return default
        b = args["B"]
        if b <= 16:
            return 16
        if b <= 32:
            return 32
        if b <= 64:
            return 64
        return default

    if fp32_config is None:
        kwargs = config.all_kwargs()
        return triton.heuristics(
            {
                k: (
                    (lambda args, _v=v: block_b(args, _v, None))
                    if k == "BLOCK_B"
                    else (lambda args, _v=v: _v)
                )
                for k, v in kwargs.items()
            }
        )
    else:
        assert arg_name is not None

        kwargs = config.all_kwargs()
        fp32_kwargs = fp32_config.all_kwargs()
        assert kwargs.keys() == fp32_kwargs.keys()

        keys_opts = list(kwargs.items())
        fp32_opts = [fp32_kwargs[k] for k, _ in keys_opts]
        return triton.heuristics(
            {
                k: (
                    (
                        lambda args, _v=v, _fp32_v=fp32_v: block_b(
                            args, _v, _fp32_v
                        )
                    )
                    if k == "BLOCK_B"
                    else (
                        lambda args, _v=v, _fp32_v=fp32_v: (
                            _fp32_v if args[arg_name].dtype == torch.float32 else _v
                        )
                    )
                )
                for (k, v), fp32_v in zip(keys_opts, fp32_opts, strict=True)
            }
        )


## NOTE
# Forward and backward keep the same tile dimensions and dot precision so they
# reconstruct logits consistently. Scheduling (warps/stages) may differ.
def _cce_best_config() -> Config:
    return Config(dict(BLOCK_B=128, BLOCK_V=128, BLOCK_D=32), num_warps=4, num_stages=3)


def _cce_best_config_fp32() -> Config:
    return Config(dict(BLOCK_B=32, BLOCK_V=128, BLOCK_D=32), num_warps=4, num_stages=3)


def _cce_backward_best_config() -> Config:
    return Config(dict(BLOCK_B=128, BLOCK_V=128, BLOCK_D=32), num_warps=8, num_stages=3)


def _cce_backward_best_config_fp32() -> Config:
    # The backward epilogue needs more shared memory than forward. Three
    # stages exceeds the 99 KiB limit on some consumer GPUs by ~1 KiB.
    return Config(dict(BLOCK_B=32, BLOCK_V=128, BLOCK_D=32), num_warps=4, num_stages=2)


def _cce_backward_low_smem_config() -> Config:
    """Large tile for devices reporting a lower shared-memory ceiling.

    The eight-warp schedule needs 106,496 bytes, but reducing scheduling to
    four warps keeps the same 128x128 tensor-core tile within the lower budget.
    This avoids the severe launch-count increase of the old 32x128 fallback.
    """
    return Config(dict(BLOCK_B=128, BLOCK_V=128, BLOCK_D=16), num_warps=4, num_stages=2)


def _cce_backward_heuristic_config() -> Config:
    """Choose a BF16-safe tile from the active device's SMEM capability.

    The BF16/FP16 fast tile is valid on larger parts, but Triton reports
    106,496 bytes for its eight-warp backward epilogue. Devices reporting a
    lower per-block limit keep the same tensor-core tile with four warps. This
    changes scheduling, not arithmetic, and avoids multiplying the number of
    backward programs as batch, vocabulary, context, or hidden size grows.
    """
    config = _cce_backward_best_config()
    if not torch.cuda.is_available():
        return config
    try:
        properties = driver.active.utils.get_device_properties(torch.cuda.current_device())
        if properties["max_shared_mem"] < 106496:
            return _cce_backward_low_smem_config()
    except (KeyError, RuntimeError):
        # Keep import/fallback behavior unchanged if a CUDA driver is not
        # initialized yet; Triton will select the normal config at launch.
        pass
    return config


def cce_forward_autotune() -> Callable[..., autotuner.Autotuner | autotuner.Heuristics]:
    if _AUTOTUNE:
        return _cce_autotune(
            configs=get_autotune_config(),
            key=["V", "D", "B_BIN", "MODE"],
            prune_configs_by={
                "early_config_prune": early_config_prune,
                "perf_model": estimate_matmul_time,
                "top_k": 6,
            },
            restore_value=["LSE", "MeanLogit"],
            reset_to_zero=["LA"],
            cache_results=True,
        )
    else:
        return _heuristics_from_config(
            _cce_best_config(),
            _cce_best_config_fp32(),
            "E",
            adaptive_block_b=True,
        )


def _bw_total_ops_fn(B, V, D) -> float:
    return 2 * B * V * D + 6 * B * V + 0.2 * (2 * B * V * D + 2 * B * V * D)


def _bw_total_store_fn(B, V, D, dtsize, num_cta_b, num_cta_v):
    return 0.2 * (num_cta_v * B * D * dtsize + num_cta_b * D * V * dtsize)


def cce_backward_autotune() -> Callable[..., autotuner.Autotuner | autotuner.Heuristics]:
    if _AUTOTUNE:
        return _cce_autotune(
            configs=get_autotune_config(),
            key=["V", "D", "B_BIN", "MODE"],
            prune_configs_by={
                "early_config_prune": functools.partial(
                    early_config_prune, shared_memory_factor=2.0
                ),
                "perf_model": functools.partial(
                    estimate_matmul_time,
                    total_ops_fn=_bw_total_ops_fn,
                    total_store_fn=_bw_total_store_fn,
                ),
                "top_k": 4,
            },
            reset_to_zero=["dE", "dC", "dBias"],
            cache_results=True,
        )
    else:
        return _heuristics_from_config(
            _cce_backward_heuristic_config(),
            _cce_backward_best_config_fp32(),
            "E",
            adaptive_block_b=True,
        )


def _indexed_dot_best_config() -> Config:
    return Config(dict(BLOCK_B=128, BLOCK_D=256), num_warps=16, num_stages=4)


def _indexed_dot_all_configs() -> list[Config]:
    return [
        Config(
            dict(
                BLOCK_B=128,
                BLOCK_D=128,
            ),
            num_warps=4,
            num_stages=4,
        ),
        Config(
            dict(
                BLOCK_B=128,
                BLOCK_D=128,
            ),
            num_warps=8,
            num_stages=4,
        ),
        Config(
            dict(
                BLOCK_B=256,
                BLOCK_D=256,
            ),
            num_warps=16,
            num_stages=4,
        ),
        Config(
            dict(
                BLOCK_B=256,
                BLOCK_D=128,
            ),
            num_warps=16,
            num_stages=4,
        ),
        Config(
            dict(
                BLOCK_B=128,
                BLOCK_D=256,
            ),
            num_warps=16,
            num_stages=4,
        ),
    ]


def indexed_dot_autotune() -> Callable[..., autotuner.Autotuner | autotuner.Heuristics]:
    if _AUTOTUNE:
        return _cce_autotune(
            configs=_indexed_dot_all_configs(),
            key=["D", "B_BIN"],
            reset_to_zero=["Out"],
        )
    else:
        return _heuristics_from_config(_indexed_dot_best_config())
