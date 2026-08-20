from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from cut_cross_entropy import linear_cross_entropy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile global and MEAP-conditioned MiLe normalization."
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=513)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=64_402)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--vram-limit-gib", type=float, default=10.0)
    return parser.parse_args()


def _measure(
    functions: dict[str, Callable[[], torch.Tensor]],
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    steps: int,
) -> tuple[dict[str, list[float]], dict[str, tuple[int, int]]]:
    for _ in range(warmup):
        for function in functions.values():
            function()
    torch.cuda.synchronize()

    timings = {name: [] for name in functions}
    names = tuple(functions)
    for index in range(steps):
        order = names if index % 2 == 0 else tuple(reversed(names))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end))

    peaks = {}
    for name, function in functions.items():
        embedding, classifier = tensors[name]
        embedding.grad = None
        classifier.grad = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        function()
        torch.cuda.synchronize()
        peaks[name] = (baseline, torch.cuda.max_memory_allocated())
    return timings, peaks


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.warmup < 0:
        raise ValueError("steps must be positive and warmup must be non-negative")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    base_embedding = torch.randn(
        args.batch,
        args.sequence,
        args.hidden,
        device=device,
        dtype=dtype,
    )
    base_classifier = torch.randn(
        args.vocab,
        args.hidden,
        device=device,
        dtype=dtype,
    )
    targets = torch.randint(
        args.vocab,
        (args.batch, args.sequence),
        device=device,
    )
    groups = torch.zeros_like(targets, dtype=torch.bool)
    groups[:, 1::7] = True

    tensors = {
        name: (
            base_embedding.detach().clone().requires_grad_(True),
            base_classifier.detach().clone().requires_grad_(True),
        )
        for name in ("global", "grouped")
    }
    del base_embedding, base_classifier

    def make_step(name: str, *, grouped: bool) -> Callable[[], torch.Tensor]:
        embedding, classifier = tensors[name]

        def loss_function() -> torch.Tensor:
            return linear_cross_entropy(
                embedding,
                classifier,
                targets,
                shift=1,
                impl="cce_kahan_full_c",
                mile_enabled=True,
                mile_gamma=1.0,
                mile_group_mask=groups if grouped else None,
                filter_eps=None,
            )

        # max-autotune is intentionally local to this benchmark. The library
        # neither enables nor changes a caller's torch.compile configuration.
        compiled = torch.compile(loss_function, fullgraph=True, mode="max-autotune")

        def step() -> torch.Tensor:
            embedding.grad = None
            classifier.grad = None
            loss = compiled()
            loss.backward()
            return loss

        return step

    functions = {
        "global": make_step("global", grouped=False),
        "grouped": make_step("grouped", grouped=True),
    }
    timings, peaks = _measure(
        functions,
        tensors,
        warmup=args.warmup,
        steps=args.steps,
    )

    print(
        f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"rows={args.batch * (args.sequence - 1)} vocab={args.vocab} "
        f"hidden={args.hidden}"
    )
    for name in ("global", "grouped"):
        values = timings[name]
        baseline, peak = peaks[name]
        print(
            f"{name:7s} mean_ms={statistics.mean(values):.4f} "
            f"median_ms={statistics.median(values):.4f} "
            f"min_ms={min(values):.4f} max_ms={max(values):.4f} "
            f"baseline_mib={baseline / 2**20:.2f} peak_mib={peak / 2**20:.2f}"
        )

    max_peak = max(peak for _baseline, peak in peaks.values())
    if max_peak > args.vram_limit_gib * 1024**3:
        raise RuntimeError(f"benchmark exceeded {args.vram_limit_gib:g} GiB allocated VRAM")


if __name__ == "__main__":
    main()
