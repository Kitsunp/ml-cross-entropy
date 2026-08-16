"""Compare a separate MEAP override with Leviathan's fused output epilogue.

The runner changes no production compile settings. ``max-autotune`` is local to
this benchmark, the default measurement is five steps, and the allocated-memory
ceiling defaults to 10 GiB.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from cut_cross_entropy.leviathan import (
    LeviathanConfig,
    LeviathanGenerator,
    leviathan_embedding_compiler_safe,
)


def _paired_events(
    separate,
    fused,
    *,
    warmup: int,
    steps: int,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    for _ in range(warmup):
        separate()
        fused()
    torch.cuda.synchronize()
    timings = {"separate": [], "fused": []}

    def record(name: str, function) -> None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        timings[name].append(start.elapsed_time(end))

    for index in range(steps):
        order = (("separate", separate), ("fused", fused))
        if index % 2:
            order = tuple(reversed(order))
        for name, function in order:
            record(name, function)

    def summary(values: list[float]) -> tuple[float, float, float, float]:
        return (
            statistics.mean(values),
            statistics.median(values),
            min(values),
            max(values),
        )

    return summary(timings["separate"]), summary(timings["fused"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--vram-limit-gib", type=float, default=10.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.tokens < 1 or args.steps < 1 or args.warmup < 0:
        raise ValueError("tokens/steps must be positive and warmup non-negative")

    cfg = LeviathanConfig(
        vocab_size=64_402,
        hidden_size=512,
        generator_d_seed=128,
        generator_num_modes=8,
        generator_num_knots=16,
        generator_k=3,
        generator_krank=64,
        dtype=torch.bfloat16,
    )
    generator = LeviathanGenerator(cfg).cuda()
    names = (
        "codebooks",
        "head_proj_weight",
        "head_norm_weight",
        "head_norm_bias",
        "head_spline_delta",
        "head_out_weight",
    )
    params = {name: getattr(generator, name) for name in names}
    mask_token_id = 7
    ids = torch.randint(cfg.vocab_size, (args.tokens,), device="cuda")
    ids[::7] = mask_token_id
    mask_embedding = torch.nn.Parameter(
        torch.randn(cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    )

    def separate_output() -> torch.Tensor:
        output = leviathan_embedding_compiler_safe(
            ids, params, cfg, generator.knot_grid
        )
        return torch.where(
            ids.eq(mask_token_id).unsqueeze(-1),
            mask_embedding,
            output,
        )

    def fused_output() -> torch.Tensor:
        return leviathan_embedding_compiler_safe(
            ids,
            params,
            cfg,
            generator.knot_grid,
            mask_embedding=mask_embedding,
            mask_token_id=mask_token_id,
        )

    def separate_loss() -> torch.Tensor:
        return separate_output().float().square().mean()

    def fused_loss() -> torch.Tensor:
        return fused_output().float().square().mean()

    @torch.no_grad()
    def separate_inference_fn() -> torch.Tensor:
        return separate_output()

    @torch.no_grad()
    def fused_inference_fn() -> torch.Tensor:
        return fused_output()

    separate_inference = torch.compile(
        separate_inference_fn,
        fullgraph=True,
        mode="max-autotune",
    )
    fused_inference = torch.compile(
        fused_inference_fn,
        fullgraph=True,
        mode="max-autotune",
    )
    separate_training = torch.compile(
        separate_loss, fullgraph=True, mode="max-autotune"
    )
    fused_training = torch.compile(
        fused_loss, fullgraph=True, mode="max-autotune"
    )

    with torch.no_grad():
        torch.testing.assert_close(separate_output(), fused_output(), rtol=0, atol=0)

    trainable = [*params.values(), mask_embedding]

    def clear_grads() -> None:
        for parameter in trainable:
            parameter.grad = None

    def separate_step() -> None:
        clear_grads()
        separate_training().backward()

    def fused_step() -> None:
        clear_grads()
        fused_training().backward()

    forward_separate, forward_fused = _paired_events(
        separate_inference,
        fused_inference,
        warmup=args.warmup,
        steps=args.steps,
    )
    training_separate, training_fused = _paired_events(
        separate_step,
        fused_step,
        warmup=args.warmup,
        steps=args.steps,
    )

    clear_grads()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fused_step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    if peak > int(args.vram_limit_gib * 1024**3):
        raise RuntimeError("allocated-memory peak exceeded the validation ceiling")

    def row(name: str, values: tuple[float, float, float, float]) -> None:
        print(
            f"{name:24s} {values[0]:9.4f} {values[1]:9.4f} "
            f"{values[2]:9.4f} {values[3]:9.4f}"
        )

    print(
        f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"tokens={args.tokens} steps={args.steps} dtype=bfloat16"
    )
    print("path                       mean_ms median_ms    min_ms    max_ms")
    row("forward separate", forward_separate)
    row("forward fused", forward_fused)
    row("train separate", training_separate)
    row("train fused", training_fused)
    print(
        f"fused_baseline_mib={baseline / 2**20:.2f} "
        f"fused_peak_mib={peak / 2**20:.2f}"
    )


if __name__ == "__main__":
    main()
