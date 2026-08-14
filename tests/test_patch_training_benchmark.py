from __future__ import annotations

import sys

import pytest

from benchmark.patch_training_e2e import _parse_args


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--dtype", "fp32"], "invalid choice"),
        (["--sequence", "1"], "--sequence must be at least 2"),
    ],
)
def test_rejects_unsupported_compiled_geometries(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["patch_training_e2e", *arguments])

    with pytest.raises(SystemExit, match="2"):
        _parse_args()

    assert message in capsys.readouterr().err
