"""Profile the final MEAP embedding override under torch.compile.

This benchmark changes no production compile settings.  It deliberately uses
``max-autotune`` only inside this isolated runner and measures five steps by
default.  The default shape stays far below the 10 GiB validation ceiling.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from cut_cross_entropy import apply_meap_embedding_override

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _loss(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    output = apply_meap_embedding_override(
        input_ids,
        embeddings,
        mask_embedding,
        mask_token_id,
    )
    return output.float().square().mean()


def _measure(
    function,
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    mask_embedding: torch.Tensor,
    mask_token_id: int,
    *,
    warmup: int,
    steps: int,
) -> tuple[float, float, int, int, int]:
    def step() -> None:
        embeddings.grad = None
        mask_embedding.grad = None
        function(input_ids, embeddings, mask_embedding, mask_token_id).backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    embeddings.grad = None
    mask_embedding.grad = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    timings = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    total_peak = torch.cuda.max_memory_allocated()
    incremental_peak = total_peak - baseline
    return statistics.mean(timings), min(timings), baseline, total_peak, incremental_peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--vram-limit-gib", type=float, default=10.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.steps < 1 or args.warmup < 0:
        raise ValueError("steps must be positive and warmup non-negative.")

    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    mask_token_id = 16
    input_ids = torch.randint(
        17,
        65_536,
        (args.batch, args.sequence),
        device=device,
        dtype=torch.long,
    )
    input_ids[:, ::7] = mask_token_id
    embeddings = torch.randn(
        args.batch,
        args.sequence,
        args.hidden,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    mask_embedding = torch.randn(
        args.hidden,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    compiled = torch.compile(_loss, fullgraph=True, mode="max-autotune")

    eager_embeddings = embeddings.detach().clone().requires_grad_(True)
    eager_mask = mask_embedding.detach().clone().requires_grad_(True)
    eager_value = _loss(
        input_ids, eager_embeddings, eager_mask, mask_token_id
    )
    eager_value.backward()
    compiled_embeddings = embeddings.detach().clone().requires_grad_(True)
    compiled_mask = mask_embedding.detach().clone().requires_grad_(True)
    compiled_value = compiled(
        input_ids, compiled_embeddings, compiled_mask, mask_token_id
    )
    compiled_value.backward()
    tolerances = {
        torch.float16: (2e-3, 2e-4),
        torch.bfloat16: (1e-2, 2e-3),
        torch.float32: (1e-5, 1e-6),
    }
    rtol, atol = tolerances[dtype]
    torch.testing.assert_close(compiled_value, eager_value, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        compiled_embeddings.grad,
        eager_embeddings.grad,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        compiled_mask.grad,
        eager_mask.grad,
        rtol=rtol,
        atol=atol,
    )
    mask_grad_max_abs_error = (
        compiled_mask.grad.float() - eager_mask.grad.float()
    ).abs().max()

    eager_mean, eager_min, eager_baseline, eager_peak, eager_incremental = _measure(
        _loss,
        input_ids,
        embeddings,
        mask_embedding,
        mask_token_id,
        warmup=args.warmup,
        steps=args.steps,
    )
    (
        compiled_mean,
        compiled_min,
        compiled_baseline,
        compiled_peak,
        compiled_incremental,
    ) = _measure(
        compiled,
        input_ids,
        embeddings,
        mask_embedding,
        mask_token_id,
        warmup=args.warmup,
        steps=args.steps,
    )
    limit = int(args.vram_limit_gib * 1024**3)
    if max(eager_peak, compiled_peak) > limit:
        raise RuntimeError(
            f"Measured total allocated peak exceeds {args.vram_limit_gib:.2f} GiB."
        )

    print(
        f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"shape=({args.batch},{args.sequence},{args.hidden}) dtype={args.dtype} "
        f"steps={args.steps} mask_grad_max_abs_error="
        f"{mask_grad_max_abs_error.item():.8g}"
    )
    print("path             mean_ms    min_ms    baseline_mib    peak_mib    increment_mib")
    print(
        f"eager          {eager_mean:9.4f} {eager_min:9.4f} "
        f"{eager_baseline / 2**20:15.2f} {eager_peak / 2**20:11.2f} "
        f"{eager_incremental / 2**20:16.2f}"
    )
    print(
        f"max-autotune   {compiled_mean:9.4f} {compiled_min:9.4f} "
        f"{compiled_baseline / 2**20:15.2f} {compiled_peak / 2**20:11.2f} "
        f"{compiled_incremental / 2**20:16.2f}"
    )


if __name__ == "__main__":
    main()
