from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from cut_cross_entropy import torch_2_14
from cut_cross_entropy.utils import is_package_greater_or_equal


def test_torch_2_14_boolean_matches_installed_version() -> None:
    assert torch_2_14.TORCH_2_14_OR_NEWER == is_package_greater_or_equal(
        "torch", "2.14"
    )
    assert (
        not torch_2_14.TORCH_2_14_CUDA_INTEGRATION
        or torch_2_14.TORCH_2_14_OR_NEWER
    )
    assert (
        not torch_2_14.TORCH_2_14_GRAPH_ANNOTATIONS
        or torch_2_14.TORCH_2_14_CUDA_INTEGRATION
    )


def test_kernel_region_is_noop_when_annotations_are_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch_2_14, "_mark_kernels", None)
    monkeypatch.setattr(torch_2_14, "_MemPool", None)

    with torch_2_14.cuda_kernel_region("test.region", torch.device("cuda")):
        value = 7
    assert value == 7


def test_kernel_region_uses_backward_false(monkeypatch) -> None:
    calls: list[tuple[object, bool]] = []

    @contextmanager
    def fake_mark(annotation, *, backward):
        calls.append((annotation, backward))
        yield

    monkeypatch.setattr(torch_2_14, "_mark_kernels", fake_mark)
    monkeypatch.setattr(torch_2_14, "_MemPool", None)
    with torch_2_14.cuda_kernel_region("test.region", torch.device("cuda")):
        pass

    assert calls == [
        (
            {
                "name": "test.region",
                "library": "cut_cross_entropy",
                "component": "test",
                "phase": "region",
                "operator": "cut_cross_entropy::test_region",
            },
            False,
        )
    ]


def test_warmup_hook_runs_once_per_geometry(monkeypatch) -> None:
    calls: list[None] = []
    monkeypatch.setattr(
        torch_2_14,
        "_mark_warmup_incomplete",
        lambda: calls.append(None),
    )
    torch_2_14._WARMED_GEOMETRIES.clear()

    torch_2_14.mark_warmup_incomplete_once("kernel", (1024, "bf16"))
    torch_2_14.mark_warmup_incomplete_once("kernel", (1024, "bf16"))
    torch_2_14.mark_warmup_incomplete_once("kernel", (2048, "bf16"))

    assert len(calls) == 2


def test_memory_annotations_are_noop_by_default(monkeypatch) -> None:
    monkeypatch.setattr(torch_2_14, "_annotate_tensor", None)
    torch_2_14.annotate_tensors("test", tensor=None)


def test_memory_annotations_register_component_and_role(monkeypatch) -> None:
    calls: list[tuple[object, str]] = []

    class FakeCudaTensor:
        is_cuda = True

    monkeypatch.setattr(
        torch_2_14,
        "_annotate_tensor",
        lambda tensor, metadata: calls.append((tensor, metadata)),
    )
    tensor = FakeCudaTensor()
    torch_2_14.annotate_tensors("cce.forward", lse=tensor)  # type: ignore[arg-type]

    assert calls == [(tensor, "cut_cross_entropy::cce.forward.lse")]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch_2_14.TORCH_2_14_CUDA_MEMORY_POOL,
    reason="PyTorch 2.14 CUDA memory-pool integration is required",
)
def test_cuda_kernel_regions_share_one_pool_per_device() -> None:
    device = torch.device("cuda")
    first = torch_2_14._cuda_memory_pool(device)
    second = torch_2_14._cuda_memory_pool(device)
    assert first is second

    with torch_2_14.cuda_kernel_region("test.pool", device):
        allocation = torch.empty(4096, device=device)
    torch.cuda.synchronize()

    assert allocation.is_cuda
    assert first is not None
    assert first.snapshot()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch_2_14.TORCH_2_14_OR_NEWER,
    reason="PyTorch 2.14 CUDA graph annotations are required",
)
def test_cuda_graph_region_registers_kernel_metadata(monkeypatch) -> None:
    graph_annotations = pytest.importorskip("torch.cuda.graph_annotations")
    if not graph_annotations.is_available():
        pytest.skip("CUDA graph annotation driver support is unavailable")
    monkeypatch.setattr(torch_2_14, "_mark_kernels", graph_annotations.mark_kernels)

    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(1024, device=device)
    torch.sin(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, enable_annotations=True):
        with torch_2_14.cuda_kernel_region("test.graph", device):
            output = torch.sin(x)
    graph.replay()
    torch.cuda.synchronize()

    annotations = graph_annotations.get_kernel_annotations()
    flat = [item for values in annotations.values() for item in values]
    assert torch.isfinite(output).all()
    assert any(
        item.get("name") == "test.graph"
        and item.get("component") == "test"
        and item.get("phase") == "graph"
        and item.get("operator") == "cut_cross_entropy::test_graph"
        for item in flat
    )
