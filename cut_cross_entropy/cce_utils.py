# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import enum
from enum import auto
from typing import TypedDict


class LinearCrossEntropyImpl(enum.IntEnum):
    CCE = auto()
    TORCH_COMPILE = auto()
    CCE_KAHAN_FULL_C = auto()
    CCE_KAHAN_FULL_E = auto()

    CCE_EXACT = auto()
    CCE_KAHAN_FULL_C_FULL_E = auto()
    CCE_KAHAN_FULL = auto()


class CCEPreset(TypedDict):
    filter_eps: float | str | None
    accum_e_fp32: bool
    accum_c_fp32: bool
    filter_e_grad: bool
    filter_c_grad: bool


class CCEPresets:
    names: set[str] = set(
        v.name.lower() for v in LinearCrossEntropyImpl if v.name.lower() != "torch_compile"
    )

    @classmethod
    def build_for_impl(cls, impl: str, opts: CCEPreset) -> CCEPreset:
        if impl not in cls.names:
            raise ValueError(f"{impl!r} not in {cls.names}")

        if impl == "cce":
            return opts

        opts = opts.copy()
        if impl in ("cce_exact", "cce_kahan_full", "cce_kahan_full_c_full_e"):
            opts["filter_eps"] = None
            opts["accum_e_fp32"] = True
            opts["accum_c_fp32"] = True

            return opts

        if impl == "cce_kahan_full_c":
            opts["filter_eps"] = "auto"

            opts["accum_c_fp32"] = True
            opts["filter_c_grad"] = False

            opts["accum_e_fp32"] = True
            opts["filter_e_grad"] = True

            return opts

        if impl == "cce_kahan_full_e":
            opts["filter_eps"] = "auto"

            opts["accum_c_fp32"] = True
            opts["filter_c_grad"] = True

            opts["accum_e_fp32"] = True
            opts["filter_e_grad"] = False

            return opts

        raise NotImplementedError(f"{impl=}")


# Supported active ranks; False disables diagnostics and True retains legacy 5.
SUPPORTED_TOPK = (3, 5)


def normalize_return_topk(value: bool | int) -> int:
    """Normalize legacy bool requests or an explicit maximum diagnostic rank."""
    if type(value) not in (bool, int) or value not in (0, 1, 3, 5):
        raise ValueError("return_topk must be False, True, 3 or 5")
    return 5 if value == 1 else int(value)
