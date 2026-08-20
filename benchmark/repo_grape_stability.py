"""Compare long-run REPO-GRAPE training dynamics with the compiled reference."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from cut_cross_entropy.repo_grape import repo_grape  # noqa: E402

TEST_PATH = ROOT / "tests" / "test_repo_grape.py"
SPEC = importlib.util.spec_from_file_location("repo_grape_test_reference", TEST_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {TEST_PATH}")
reference_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference_module
SPEC.loader.exec_module(reference_module)


class StabilityModel(nn.Module):
    def __init__(self, *, use_kernel: bool, pseudo_factor: int) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        self.pseudo_factor = pseudo_factor
        self.batch = 2
        self.sequence = 64
        self.hidden_dim = 96
        self.q_heads = 8
        self.k_heads = 4
        self.head_dim = 64
        self.rot_half = 16
        self.q_proj = nn.Linear(self.hidden_dim, self.q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.k_heads * self.head_dim, bias=False)
        self.z_proj = nn.Linear(self.hidden_dim, self.q_heads, bias=False)
        self.alpha = nn.Parameter(torch.ones(self.q_heads, dtype=torch.float32))
        self.log_scale = nn.Parameter(
            torch.zeros(self.q_heads, self.head_dim // 2, dtype=torch.float32)
        )
        self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float32))
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float32))
        inv_freq = torch.exp(
            -math.log(10_000.0) * torch.arange(self.rot_half, dtype=torch.float32) / self.rot_half
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer(
            "position_ids",
            torch.arange(self.sequence, dtype=torch.int64).expand(self.batch, -1),
            persistent=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        q_target: torch.Tensor,
        k_target: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = (
                self.q_proj(hidden)
                .view(self.batch, self.sequence, self.q_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                self.k_proj(hidden)
                .view(self.batch, self.sequence, self.k_heads, self.head_dim)
                .transpose(1, 2)
            )
            z = self.z_proj(hidden).transpose(1, 2)
        if self.pseudo_factor == 2:
            q = (
                q.unsqueeze(3)
                .expand(-1, -1, -1, 2, -1)
                .reshape(self.batch, self.q_heads, self.sequence * 2, self.head_dim)
            )
            k = (
                k.unsqueeze(3)
                .expand(-1, -1, -1, 2, -1)
                .reshape(self.batch, self.k_heads, self.sequence * 2, self.head_dim)
            )
        if self.use_kernel:
            q_out, k_out = repo_grape(
                q,
                k,
                z,
                self.position_ids,
                self.inv_freq,
                self.alpha,
                self.log_scale,
                1.0,
                sequence_pseudo_factor=self.pseudo_factor,
                momentum_gamma=0.1,
                output_dtype=torch.bfloat16,
                q_norm_weight=self.q_norm_weight,
                k_norm_weight=self.k_norm_weight,
                rms_norm_eps=1.0e-6,
            )
        else:
            q_out, k_out = reference_module._reference(
                q,
                k,
                z,
                self.position_ids,
                self.inv_freq,
                self.alpha,
                self.log_scale,
                self.q_norm_weight,
                self.k_norm_weight,
                attention_scaling=1.0,
                momentum_gamma=0.1,
                rms_norm_eps=1.0e-6,
                sequence_pseudo_factor=self.pseudo_factor,
                output_dtype=torch.bfloat16,
            )
        return F.mse_loss(q_out.float(), q_target.float()) + F.mse_loss(
            k_out.float(), k_target.float()
        )


def _grad_norm(parameters: list[nn.Parameter]) -> float:
    squared = torch.zeros((), device="cuda", dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.double().square().sum()
    return squared.sqrt().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--pseudo-factor", type=int, choices=(1, 2), default=2)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--memory-limit-gib", type=float, default=10.0)
    args = parser.parse_args()
    if args.steps < 1 or args.check_every < 1:
        raise ValueError("steps and check-every must be positive")

    properties = torch.cuda.get_device_properties(0)
    if args.memory_limit_gib is not None:
        fraction = min(args.memory_limit_gib * 1024**3 / properties.total_memory, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.manual_seed(20260820)
    reference_model = StabilityModel(use_kernel=False, pseudo_factor=args.pseudo_factor).cuda()
    kernel_model = copy.deepcopy(reference_model)
    kernel_model.use_kernel = True
    reference_parameters = list(reference_model.parameters())
    kernel_parameters = list(kernel_model.parameters())
    reference_optimizer = torch.optim.AdamW(reference_parameters, lr=5.0e-5, weight_decay=0.01)
    kernel_optimizer = torch.optim.AdamW(kernel_parameters, lr=5.0e-5, weight_decay=0.01)
    compiled_reference = torch.compile(reference_model, mode="max-autotune", fullgraph=True)
    compiled_kernel = torch.compile(kernel_model, mode="max-autotune", fullgraph=True)

    effective_sequence = reference_model.sequence * args.pseudo_factor
    hidden_bank = torch.randn(
        4,
        reference_model.batch,
        reference_model.sequence,
        reference_model.hidden_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_target_bank = torch.randn(
        4,
        reference_model.batch,
        reference_model.q_heads,
        effective_sequence,
        reference_model.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_target_bank = torch.randn(
        4,
        reference_model.batch,
        reference_model.k_heads,
        effective_sequence,
        reference_model.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    initial_losses: dict[str, float] = {}
    final_losses: dict[str, float] = {}
    maximum_grad_norm = {"reference": 0.0, "kernel": 0.0}
    nonfinite = {"reference": 0, "kernel": 0}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(args.steps):
        bank = step % 4
        inputs = (hidden_bank[bank], q_target_bank[bank], k_target_bank[bank])
        variants = (
            (
                "reference",
                compiled_reference,
                reference_optimizer,
                reference_parameters,
            ),
            ("kernel", compiled_kernel, kernel_optimizer, kernel_parameters),
        )
        for name, model, optimizer, parameters in variants:
            optimizer.zero_grad(set_to_none=True)
            loss = model(*inputs)
            loss.backward()
            if step % args.check_every == 0 or step + 1 == args.steps:
                loss_value = loss.detach().item()
                norm = _grad_norm(parameters)
                maximum_grad_norm[name] = max(maximum_grad_norm[name], norm)
                if not math.isfinite(loss_value) or not math.isfinite(norm):
                    nonfinite[name] += 1
                if step == 0:
                    initial_losses[name] = loss_value
                if step + 1 == args.steps:
                    final_losses[name] = loss_value
            optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    difference_sq = 0.0
    reference_sq = 0.0
    max_abs = 0.0
    all_parameters_finite = True
    for reference_parameter, kernel_parameter in zip(reference_parameters, kernel_parameters):
        reference_value = reference_parameter.detach().float()
        kernel_value = kernel_parameter.detach().float()
        difference = kernel_value - reference_value
        difference_sq += difference.double().square().sum().item()
        reference_sq += reference_value.double().square().sum().item()
        max_abs = max(max_abs, difference.abs().max().item())
        all_parameters_finite &= bool(torch.isfinite(reference_value).all())
        all_parameters_finite &= bool(torch.isfinite(kernel_value).all())

    print(
        json.dumps(
            {
                "device": properties.name,
                "torch": torch.__version__,
                "steps": args.steps,
                "pseudo_factor": args.pseudo_factor,
                "elapsed_seconds": elapsed,
                "paired_steps_per_second": args.steps / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "initial_losses": initial_losses,
                "final_losses": final_losses,
                "maximum_grad_norm": maximum_grad_norm,
                "nonfinite_checks": nonfinite,
                "all_parameters_finite": all_parameters_finite,
                "parameter_relative_l2": math.sqrt(difference_sq / max(reference_sq, 1.0e-30)),
                "parameter_max_abs": max_abs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
