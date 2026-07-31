import pytest
import torch
import torch.nn.functional as F

from cut_cross_entropy import linear_cross_entropy, meap_mask_inputs
from cut_cross_entropy.cce_backward import _auto_fp16_accumulation_dtypes
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.utils import TensorInfo

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


@skip_no_cuda
@pytest.mark.parametrize("mile", [False, True])
@pytest.mark.parametrize("mu", [False, True])
def test_auto_fp16_extensions_with_meap_padding(
    monkeypatch: pytest.MonkeyPatch, mile: bool, mu: bool
) -> None:
    """Exercise the eligible SM12 route with the fused μ dC finalization."""
    for name, value in {
        "CCE_DE_ACCUM_DTYPE": "auto",
        "CCE_DC_ACCUM_DTYPE": "auto",
        "CCE_FP16_ACCUM_SCALE": "auto",
        "CCE_BACKWARD_REDUCTION": "lock",
        "CCE_MU_FUSED_CAST": "1",
    }.items():
        monkeypatch.setenv(name, value)

    batch, sequence_length, vocab, dim = 128, 64, 4096, 512
    generator = torch.Generator(device="cuda").manual_seed(6100 + int(mile) + 2 * int(mu))
    embedding_weight = torch.randn(vocab, dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    classifier = torch.randn(vocab, dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    input_ids = torch.randint(
        vocab - 1,
        (batch, sequence_length),
        device="cuda",
        dtype=torch.long,
        generator=generator,
    )
    lengths = torch.full((batch,), sequence_length, device="cuda", dtype=torch.long)
    lengths[::4] -= 16
    padding_mask = torch.arange(sequence_length, device="cuda").unsqueeze(0) >= lengths[:, None]
    targets = input_ids.masked_fill(padding_mask, IGNORE_INDEX)
    input_ids, selected = meap_mask_inputs(
        input_ids,
        vocab - 1,
        padding_mask=padding_mask,
        mask_ratio=0.15,
        seed=6107,
        return_mask=True,
    )
    assert selected.any()

    hidden = F.embedding(input_ids, embedding_weight).requires_grad_(True)
    classifier = classifier.detach().requires_grad_(True)
    loss, metrics = linear_cross_entropy(
        hidden,
        classifier,
        targets,
        shift=1,
        impl="cce_kahan_full_c",
        mile_enabled=mile,
        mile_gamma=1.0,
        mu_loss_enabled=mu,
        mu_loss_lambda=1e-4,
        return_loss_metrics=True,
    )
    reconstructed = (
        metrics["ntp_ce_unweighted"]
        + metrics["mile_reweighting_delta"]
        + metrics["mu_loss"]
    )
    torch.testing.assert_close(reconstructed, loss.detach(), rtol=2e-5, atol=2e-5)
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert classifier.grad is not None and torch.isfinite(classifier.grad).all()

    valid_tokens = int((targets[:, 1:] != IGNORE_INDEX).sum())
    e_flat = hidden.detach().flatten(0, -2)
    de_fp16, dc_fp16 = _auto_fp16_accumulation_dtypes(
        e_flat,
        TensorInfo(e_flat.dtype, True),
        classifier.detach(),
        TensorInfo(classifier.dtype, True),
        valid_tokens,
        accum_e_fp32=True,
        accum_c_fp32=True,
        dlse=None,
        mile_weight=torch.ones(valid_tokens, device="cuda") if mile else None,
        mu=torch.ones(dim, device="cuda") if mu else None,
        mile_gamma=1.0 if mile else None,
        reduce_e_grad=False,
        pg=None,
    )
    assert de_fp16 is True
    assert dc_fp16 is True

    if mu:
        monkeypatch.setenv("CCE_MU_FUSED_CAST", "0")
        _, dc_fp32 = _auto_fp16_accumulation_dtypes(
            e_flat,
            TensorInfo(e_flat.dtype, True),
            classifier.detach(),
            TensorInfo(classifier.dtype, True),
            valid_tokens,
            accum_e_fp32=True,
            accum_c_fp32=True,
            dlse=None,
            mile_weight=torch.ones(valid_tokens, device="cuda") if mile else None,
            mu=torch.ones(dim, device="cuda"),
            mile_gamma=1.0 if mile else None,
            reduce_e_grad=False,
            pg=None,
        )
        assert dc_fp32 is False
