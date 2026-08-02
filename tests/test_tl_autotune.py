from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cut_cross_entropy import tl_autotune


def _config_signature(config):
    return config.kwargs, config.num_warps, config.num_stages


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
        active=SimpleNamespace(
            utils=SimpleNamespace(get_device_properties=fail)
        )
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
