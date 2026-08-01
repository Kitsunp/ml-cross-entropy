# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_cross_entropy.cce_lse_forward import (
    _split_v_env_enabled,
    cce_lse_forward_kernel,
)
from cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def test_split_v_is_explicitly_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("CCE_SPLIT_V", raising=False)
    assert not _split_v_env_enabled()

    monkeypatch.setenv("CCE_SPLIT_V", "1")
    assert _split_v_env_enabled()

    monkeypatch.setenv("CCE_SPLIT_V", "off")
    assert not _split_v_env_enabled()


def _lse(
    e: torch.Tensor, c: torch.Tensor, bias: torch.Tensor | None, softcap: float | None
) -> torch.Tensor:
    logits = e @ c.T
    if bias is not None:
        logits += bias

    if softcap is not None:
        logits = softcapping(logits, softcap)
    return torch.logsumexp(logits.float(), dim=-1)


@skip_no_cuda
@pytest.mark.parametrize(
    "dtype,error_tol",
    [
        (torch.float32, 1e-5),
        (torch.float16, 1e-3),
        (torch.bfloat16, 1e-2),
    ],
)
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("shape", [(256, 512, 128), (255, 507, 128), (255, 507, 123)])
def test_lse(
    dtype: torch.dtype,
    error_tol: float,
    softcap: float | None,
    has_bias: bool,
    shape: tuple[int, int, int],
):
    torch.set_float32_matmul_precision("highest")
    torch.cuda.manual_seed(0)

    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D = shape
    e = torch.randn((N, D), device="cuda", dtype=dtype) / (D**0.5)
    c = torch.randn((V, D), device="cuda", dtype=dtype)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    if has_bias:
        bias = torch.randn(V, device="cuda", dtype=dtype) * 0.02
    else:
        bias = None

    gt = _lse(e.float(), c.float(), bias.float() if bias is not None else None, softcap)

    torch.set_float32_matmul_precision("highest" if dtype == torch.float32 else "high")
    ref = _lse(e, c, bias, softcap)

    cce_lse = cce_lse_forward_kernel(e, c, bias, softcap=softcap).lse

    expected_error = (gt - ref).abs()
    cce_error = (gt - cce_lse).abs()

    assert (
        cce_error <= (expected_error + error_tol)
    ).all(), f"{(cce_error - expected_error).relu().max()=}"


@skip_no_cuda
def test_logit_avg_excludes_padded_tile_rows_with_bias() -> None:
    torch.cuda.manual_seed(0)
    n, v, d = 17, 63, 32
    e = torch.randn((n, d), device="cuda", dtype=torch.bfloat16) / (d**0.5)
    c = torch.randn((v, d), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(v, device="cuda", dtype=torch.bfloat16)

    result = cce_lse_forward_kernel(e, c, bias, return_logit_avg=True).logit_avg
    expected = (e @ c.T + bias).float().mean(dim=0)

    assert result is not None
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@skip_no_cuda
def test_split_reduction_matches_lock_with_all_optional_outputs(monkeypatch) -> None:
    torch.cuda.manual_seed(1)
    n, v, d = 37, 257, 64
    e = torch.randn((n, d), device="cuda", dtype=torch.bfloat16) / (d**0.5)
    c = torch.randn((v, d), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(v, device="cuda", dtype=torch.bfloat16) * 0.02
    targets = torch.randint(0, v, (n,), device="cuda")
    valids = torch.arange(0, n - 1, 2, device="cuda", dtype=torch.int64)

    outputs = {}
    for reduction in ("lock", "split"):
        monkeypatch.setenv("CCE_FORWARD_REDUCTION", reduction)
        outputs[reduction] = cce_lse_forward_kernel(
            e,
            c,
            bias,
            valids,
            softcap=20.0,
            targets=targets,
            shift=1,
            return_logit_avg=True,
            return_mean_logit=True,
        )

    lock = outputs["lock"]
    split = outputs["split"]
    torch.testing.assert_close(split.lse, lock.lse, atol=3e-4, rtol=3e-4)
    assert lock.logit_avg is not None and split.logit_avg is not None
    torch.testing.assert_close(split.logit_avg, lock.logit_avg, atol=3e-4, rtol=3e-4)
    assert lock.neg_correct_logit is not None and split.neg_correct_logit is not None
    torch.testing.assert_close(split.neg_correct_logit, lock.neg_correct_logit)
    assert lock.mean_logit is not None and split.mean_logit is not None
    torch.testing.assert_close(split.mean_logit, lock.mean_logit, atol=3e-4, rtol=3e-4)


@skip_no_cuda
def test_auto_split_selector_is_memory_bounded(monkeypatch) -> None:
    from cut_cross_entropy.cce_lse_forward_split import (
        select_split_v_config,
        split_v_workspace_bytes,
        use_split_reduction,
    )

    # Exercise the policy itself rather than depending on the test machine's
    # capability. Automatic Split-V is intentionally restricted to CC12.x.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    c = torch.empty((2048, 32), device="cuda", dtype=torch.bfloat16)
    small_e = torch.empty((512, 32), device="cuda", dtype=torch.bfloat16)
    large_e = torch.empty((513, 32), device="cuda", dtype=torch.bfloat16)

    assert use_split_reduction(small_e, c, 512, return_mean_logit=False)
    assert not use_split_reduction(large_e, c, 513, return_mean_logit=False)
    assert not use_split_reduction(small_e, c, 512, return_mean_logit=True)
    assert not use_split_reduction(small_e.float(), c.float(), 512, return_mean_logit=False)

    config = select_split_v_config(
        small_e,
        c,
        512,
        return_mean_logit=False,
        return_logit_avg=True,
        has_targets=True,
    )
    assert config.splits >= 1
    assert config.split_memory_bytes <= 2 * config.base_memory_bytes
    assert config.split_memory_bytes == (
        config.base_memory_bytes
        - 4 * ((512 + 15) // 16)
        + split_v_workspace_bytes(512, config.splits)
    )
    assert (config.block_b, config.block_v, config.block_d) in {
        (32, 128, 32),
        (64, 128, 32),
        (128, 128, 32),
        (128, 64, 32),
    }


@skip_no_cuda
def test_explicit_split_sentinel_falls_back_to_lock(monkeypatch) -> None:
    import importlib

    split_module = importlib.import_module("cut_cross_entropy.cce_lse_forward_split")
    split_module.clear_split_v_config_cache()
    monkeypatch.setenv("CCE_FORWARD_REDUCTION", "split")
    monkeypatch.delenv("CCE_SPLIT_V_ALLOW_UNVALIDATED", raising=False)
    monkeypatch.setattr(split_module, "_split_v_profile", lambda *_args, **_kwargs: None)

    def fail_if_launched(*_args, **_kwargs):
        pytest.fail("unsupported split-V sentinel must fall back to the lock path")

    monkeypatch.setattr(split_module, "cce_lse_forward_split", fail_if_launched)
    e = torch.randn((3, 16), device="cuda", dtype=torch.bfloat16)
    c = torch.randn((7, 16), device="cuda", dtype=torch.bfloat16)

    result = cce_lse_forward_kernel(e, c)
    assert result.lse.shape == (3,)

