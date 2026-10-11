from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cut_cross_entropy import tl_autotune


def _config_signature(config):
    return config.kwargs, config.num_warps, config.num_stages


@pytest.mark.parametrize(
    ("enabled", "block_b", "configured", "expected"),
    [
        (False, 128, 4, 4),
        (True, 32, 4, 4),
        (True, 64, 4, 8),
        (True, 128, 4, 8),
        (True, 128, 8, 8),
        (True, 128, 16, 16),
    ],
)
def test_topk_forward_warp_guard(enabled, block_b, configured, expected):
    args = dict(TopK=object() if enabled else None, BLOCK_B=block_b, num_warps=configured)
    assert tl_autotune._topk_forward_num_warps(args) == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_topk_forward_warp_heuristics_runs_after_tile_selection(monkeypatch, enabled):
    monkeypatch.setattr(tl_autotune, "_AUTOTUNE", False)

    class Capture:
        arg_names = ["E", "B", "D", "V", "TopK"]

        def run(self, *args, **kwargs):
            return kwargs

    kernel = tl_autotune.cce_forward_autotune()(Capture())
    actual = kernel.run(
        E=SimpleNamespace(dtype=torch.bfloat16),
        B=512,
        D=512,
        V=4096,
        TopK=object() if enabled else None,
    )
    assert actual["num_warps"] == (8 if enabled else 4)
    assert [actual[key] for key in ("BLOCK_B", "BLOCK_V", "BLOCK_D")] == [128, 128, 32]


@pytest.mark.parametrize(
    ("max_shared_mem", "expected_warps"),
    [(106_495, 4), (106_496, 8), (163_840, 8)],
)
def test_backward_heuristic_uses_reported_shared_memory(
    monkeypatch, max_shared_mem: int, expected_warps: int
):
    observed_devices = []
    fake_driver = SimpleNamespace(
        active=SimpleNamespace(
            utils=SimpleNamespace(
                get_device_properties=lambda device: (
                    observed_devices.append(device) or {"max_shared_mem": max_shared_mem}
                )
            )
        )
    )
    monkeypatch.setattr(tl_autotune, "driver", fake_driver)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    config = tl_autotune._cce_backward_heuristic_config()

    assert observed_devices == [3]
    assert config.kwargs == {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32}
    assert config.num_warps == expected_warps
    assert config.num_stages == 3


@pytest.mark.parametrize("failure", [KeyError("missing"), RuntimeError("no driver")])
def test_backward_heuristic_keeps_default_when_properties_are_unavailable(
    monkeypatch, failure: Exception
):
    def fail(_device):
        raise failure

    fake_driver = SimpleNamespace(
        active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=fail))
    )
    monkeypatch.setattr(tl_autotune, "driver", fake_driver)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    actual = tl_autotune._cce_backward_heuristic_config()
    expected = tl_autotune._cce_backward_best_config()

    assert _config_signature(actual) == _config_signature(expected)


def test_backward_heuristic_keeps_default_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    actual = tl_autotune._cce_backward_heuristic_config()
    expected = tl_autotune._cce_backward_best_config()

    assert _config_signature(actual) == _config_signature(expected)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, 8),
        ({"PATCH_SIZE": 1}, 8),
        ({"PATCH_SIZE": 0}, 4),
        ({"PATCH_SIZE": 5}, 4),
        ({"dtype": torch.float32}, 4),
        ({"dtype": torch.float16}, 4),
        ({"D": 64}, 4),
        ({"BLOCK_B": 64}, 4),
        ({"BLOCK_V": 64}, 4),
        ({"BLOCK_D": 32}, 4),
        ({"num_stages": 3}, 4),
        ({"num_warps": 8}, 8),
    ],
)
def test_patch_backward_warp_guard_preserves_other_geometries(overrides, expected):
    args = dict(
        PATCH_SIZE=4,
        D=128,
        BLOCK_B=128,
        BLOCK_V=128,
        BLOCK_D=16,
        num_stages=2,
        num_warps=4,
    )
    overrides = dict(overrides)
    args["E"] = SimpleNamespace(dtype=overrides.pop("dtype", torch.bfloat16))
    args.update(overrides)
    assert tl_autotune._patch_backward_num_warps(args) == expected


def test_patch_backward_warp_heuristics_runs_after_tile_selection(monkeypatch):
    monkeypatch.setattr(tl_autotune, "_AUTOTUNE", False)
    monkeypatch.setattr(
        tl_autotune,
        "_cce_backward_heuristic_config",
        tl_autotune._cce_backward_low_smem_config,
    )

    class Capture:
        arg_names = ["E", "B", "D", "V", "PATCH_SIZE"]

        def run(self, *args, **kwargs):
            return kwargs

    kernel = tl_autotune.cce_backward_autotune()(Capture())
    actual = kernel.run(
        E=SimpleNamespace(dtype=torch.bfloat16),
        B=129,
        D=128,
        V=257,
        PATCH_SIZE=4,
    )
    assert actual["num_warps"] == 8
    assert actual["num_stages"] == 2
    assert [actual[key] for key in ("BLOCK_B", "BLOCK_V", "BLOCK_D")] == [128, 128, 16]
