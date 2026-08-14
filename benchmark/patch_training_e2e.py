"""End-to-end patch-training benchmark with a graph-stable phase transition.

The benchmark is intentionally separate from the library runtime.  It can cap
its own CUDA process, compile a small causal Transformer with max-autotune, and
compare patch aggregation against processing the same raw tokens individually.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
import triton
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from cut_cross_entropy import PatchTrainingSchedule, linear_cross_entropy

Case = Literal["transition", "patch", "token_baseline"]


class TinyCausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attention_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_up = nn.Linear(dim, mlp_ratio * dim, bias=False)
        self.mlp_down = nn.Linear(mlp_ratio * dim, dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, sequence, dim = inputs.shape
        normalized = self.norm1(inputs)
        qkv = self.qkv(normalized).view(batch, sequence, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention = attention.transpose(1, 2).reshape(batch, sequence, dim)
        hidden = inputs + self.attention_out(attention)
        mlp = self.mlp_down(F.silu(self.mlp_up(self.norm2(hidden))))
        return hidden + mlp


class TinyPatchLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim, dtype=dtype)
        self.blocks = nn.ModuleList(
            TinyCausalBlock(dim, heads, mlp_ratio) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab, dtype=dtype))
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("transition", "patch", "token_baseline"), default="transition")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--patch-steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=16, help="Transformer rows in patch mode")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=1021)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-test-vram-gib", type=float, default=10.0)
    parser.add_argument("--mile", action="store_true")
    parser.add_argument("--mu-loss", action="store_true")
    parser.add_argument("--reset-optimizer-at-transition", action="store_true")
    args = parser.parse_args()

    positive = (
        "steps",
        "pool_size",
        "batch",
        "sequence",
        "dim",
        "layers",
        "heads",
        "mlp_ratio",
        "vocab",
        "patch_size",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.sequence < 2:
        parser.error("--sequence must be at least 2")
    if args.dim % args.heads != 0:
        parser.error("--dim must be divisible by --heads")
    if not 0 <= args.patch_steps <= args.steps:
        parser.error("--patch-steps must be between zero and --steps")
    if args.max_test_vram_gib <= 0:
        parser.error("--max-test-vram-gib must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def _model_forward_flops(
    model: TinyPatchLM,
    batch: int,
    sequence: int,
    dim: int,
    dtype: torch.dtype,
) -> int:
    sample = torch.randn(batch, sequence, dim, device="cuda", dtype=dtype)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(sample)
    return int(counter.get_total_flops())


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.set_float32_matmul_precision("high")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_test_vram_gib / total_gib))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    schedule = PatchTrainingSchedule(args.patch_steps, args.patch_size)
    torch.manual_seed(20260814)
    torch.cuda.manual_seed_all(20260814)
    generator = torch.Generator(device="cuda").manual_seed(20260814)

    model = TinyPatchLM(
        args.vocab,
        args.dim,
        args.layers,
        args.heads,
        args.mlp_ratio,
        dtype,
    ).to(device="cuda", dtype=dtype)
    model.train()
    raw_patch_length = args.sequence * args.patch_size
    patch_pool = torch.randint(
        args.vocab,
        (args.pool_size, args.batch, raw_patch_length),
        device="cuda",
        generator=generator,
    )
    token_pool = torch.randint(
        args.vocab,
        (args.pool_size, args.batch, args.sequence),
        device="cuda",
        generator=generator,
    )

    patch_model_flops = _model_forward_flops(
        model, args.batch, args.sequence, args.dim, dtype
    )
    token_model_flops = _model_forward_flops(
        model, args.batch, raw_patch_length, args.dim, dtype
    )
    patch_cce_flops = 2 * args.batch * (args.sequence - 1) * args.vocab * args.dim
    token_cce_flops = 2 * args.batch * (raw_patch_length - 1) * args.vocab * args.dim

    patch_enabled = args.case != "token_baseline"

    def loss_function(core_inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        hidden = model(core_inputs)
        shift = 0
        if args.case == "token_baseline":
            # Use CCE's canonical causal contract. The compiler-safe ordinary
            # route is intentionally built around shift > 0; manually slicing
            # both tensors with shift=0 would expose data-dependent ``valids``
            # construction to Dynamo instead of the opaque boundary.
            shift = 1
        else:
            hidden = hidden[..., :-1, :]
        return linear_cross_entropy(
            hidden,
            model.embedding.weight,
            targets,
            bias=model.output_bias,
            impl="cce_exact",
            filter_eps=None,
            shift=shift,
            mile_enabled=args.mile,
            mu_loss_enabled=args.mu_loss,
            patch_training_enabled=patch_enabled,
        )

    def prepare_patch(pool_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = patch_pool[pool_index]
        embeddings = model.embedding(token_ids)
        core_inputs = embeddings.unflatten(-2, (args.sequence, args.patch_size)).mean(-2)
        targets = token_ids[..., args.patch_size :].unflatten(
            -1, (args.sequence - 1, args.patch_size)
        )
        return core_inputs, schedule.prepare_patch_targets(targets)

    def prepare_token_phase(pool_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = token_pool[pool_index]
        core_inputs = model.embedding(token_ids)
        targets = schedule.prepare_token_targets(token_ids[..., 1:])
        return core_inputs, targets

    def prepare_token_baseline(pool_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = patch_pool[pool_index]
        return model.embedding(token_ids), token_ids

    if args.case == "patch":
        prepare: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]] = (
            lambda _step, pool_index: prepare_patch(pool_index)
        )
    elif args.case == "token_baseline":
        prepare = lambda _step, pool_index: prepare_token_baseline(pool_index)
    else:
        prepare = lambda step, pool_index: (
            prepare_patch(pool_index)
            if schedule.is_patch_step(step)
            else prepare_token_phase(pool_index)
        )

    def new_optimizer() -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled_loss = torch.compile(
        loss_function,
        fullgraph=True,
        dynamic=False,
        mode=args.compile_mode,
    )
    initial_model_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    optimizer = new_optimizer()

    def train_step(step: int, pool_index: int) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        core_inputs, targets = prepare(step, pool_index)
        loss = compiled_loss(core_inputs, targets)
        loss.backward()
        optimizer.step()
        return loss.detach()

    def profiled_train_step(step: int, pool_index: int) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        core_inputs, targets = prepare(step, pool_index)
        torch.cuda.nvtx.range_push(f"patch_training_e2e_{args.case}_forward")
        loss = compiled_loss(core_inputs, targets)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"patch_training_e2e_{args.case}_backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"patch_training_e2e_{args.case}_optimizer")
        optimizer.step()
        torch.cuda.nvtx.range_pop()
        return loss.detach()

    warmup_steps = []
    if args.case == "transition":
        warmup_steps = [0, args.patch_steps]
    elif args.case == "patch":
        warmup_steps = [0]
    else:
        warmup_steps = [args.steps]
    for warmup_index in range(args.warmup):
        train_step(warmup_steps[warmup_index % len(warmup_steps)], warmup_index % args.pool_size)
    optimizer.zero_grad(set_to_none=True)
    model.load_state_dict(initial_model_state)
    del initial_model_state
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                value.zero_()
    under_ncu = "NV_COMPUTE_PROFILER_PERFWORKS_DIR" in os.environ
    measured_train_step = profiled_train_step if under_ncu else train_step
    torch.cuda.synchronize()
    gc.collect()

    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    total_start = _event()
    total_end = _event()
    patch_start = _event()
    patch_end = _event()
    token_start = _event()
    token_end = _event()
    boundary_steps = (
        {
            step
            for step in range(args.patch_steps - 2, args.patch_steps + 3)
            if 0 <= step < args.steps
        }
        if args.case == "transition"
        else set()
    )
    boundary_events = {step: (_event(), _event()) for step in boundary_steps}
    transition_host_ms = 0.0
    first_loss_tensor: torch.Tensor | None = None
    final_loss_tensor: torch.Tensor | None = None

    if under_ncu:
        torch.cuda.nvtx.range_push(f"patch_training_e2e_{args.case}")
    total_start.record()
    if args.case == "transition" and args.patch_steps > 0:
        patch_start.record()
    elif args.case == "transition":
        token_start.record()
    for step in range(args.steps):
        if args.case == "transition" and step == args.patch_steps and step > 0:
            patch_end.record()
            if args.reset_optimizer_at_transition:
                torch.cuda.synchronize()
                reset_start = time.perf_counter()
                optimizer = new_optimizer()
                transition_host_ms = (time.perf_counter() - reset_start) * 1000
            token_start.record()
        events = boundary_events.get(step)
        if events is not None:
            events[0].record()
        loss = measured_train_step(step, step % args.pool_size)
        if events is not None:
            events[1].record()
        if step == 0:
            first_loss_tensor = loss
        final_loss_tensor = loss
    total_end.record()
    if args.case == "transition":
        if args.patch_steps == args.steps:
            patch_end.record()
        else:
            token_end.record()
    if under_ncu:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if first_loss_tensor is None or final_loss_tensor is None:
        raise RuntimeError("The measured loop did not execute any steps.")
    first_loss = float(first_loss_tensor)
    final_loss = float(final_loss_tensor)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    total_ms = total_start.elapsed_time(total_end)
    phase_timings: dict[str, dict[str, float | int]] = {}
    if args.case == "transition":
        if args.patch_steps:
            phase_timings["patch"] = {
                "steps": args.patch_steps,
                "ms_per_step": patch_start.elapsed_time(patch_end) / args.patch_steps,
                "predicted_tokens_per_step": args.batch
                * (args.sequence - 1)
                * args.patch_size,
                "raw_tokens_per_step": args.batch * raw_patch_length,
            }
        token_steps = args.steps - args.patch_steps
        if token_steps:
            phase_timings["token"] = {
                "steps": token_steps,
                "ms_per_step": token_start.elapsed_time(token_end) / token_steps,
                "predicted_tokens_per_step": args.batch * (args.sequence - 1),
                "raw_tokens_per_step": args.batch * args.sequence,
            }
    else:
        predictions = (
            args.batch * (args.sequence - 1) * args.patch_size
            if args.case == "patch"
            else args.batch * (raw_patch_length - 1)
        )
        phase_timings[args.case] = {
            "steps": args.steps,
            "ms_per_step": total_ms / args.steps,
            "predicted_tokens_per_step": predictions,
            "raw_tokens_per_step": args.batch * raw_patch_length,
        }
    for timing in phase_timings.values():
        timing["predicted_tokens_per_second"] = (
            float(timing["predicted_tokens_per_step"]) * 1000 / float(timing["ms_per_step"])
        )
        timing["raw_tokens_per_second"] = (
            float(timing["raw_tokens_per_step"]) * 1000 / float(timing["ms_per_step"])
        )

    stats = torch._dynamo.utils.counters["stats"]
    graph_breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "case": args.case,
        "compile_mode": args.compile_mode,
        "fullgraph": True,
        "shape": {
            "batch": args.batch,
            "patch_core_sequence": args.sequence,
            "raw_patch_tokens": raw_patch_length,
            "dim": args.dim,
            "layers": args.layers,
            "heads": args.heads,
            "vocab": args.vocab,
            "patch_size": args.patch_size,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "steps": args.steps,
        "patch_steps": args.patch_steps if args.case == "transition" else None,
        "warmup": args.warmup,
        "mile": args.mile,
        "mu_loss": args.mu_loss,
        "reset_optimizer_at_transition": args.reset_optimizer_at_transition,
        "transition_optimizer_reset_host_ms": transition_host_ms,
        "loss": {"first": first_loss, "final": final_loss},
        "timing": {
            "total_ms": total_ms,
            "overall_ms_per_step": total_ms / args.steps,
            "phases": phase_timings,
            "boundary_ms": {
                str(step): start.elapsed_time(end)
                for step, (start, end) in boundary_events.items()
            },
        },
        "memory": {
            "process_limit_gib": args.max_test_vram_gib,
            "baseline_allocated_bytes": baseline_allocated,
            "peak_allocated_bytes": peak_allocated,
            "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_reserved_bytes": peak_reserved,
        },
        "graphs": {
            "unique_graphs": int(stats.get("unique_graphs", 0)),
            "calls_captured": int(stats.get("calls_captured", 0)),
            "graph_breaks": int(graph_breaks),
        },
        "forward_flops": {
            "patch_model_counted": patch_model_flops,
            "token_baseline_model_counted": token_model_flops,
            "patch_cce_dense_dot_estimate": patch_cce_flops,
            "token_baseline_cce_dense_dot_estimate": token_cce_flops,
            "patch_total_proxy": patch_model_flops + patch_cce_flops,
            "token_baseline_total_proxy": token_model_flops + token_cce_flops,
            "patch_savings_fraction": 1
            - (patch_model_flops + patch_cce_flops) / (token_model_flops + token_cce_flops),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
