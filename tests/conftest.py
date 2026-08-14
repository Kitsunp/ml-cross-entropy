from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def restore_float32_matmul_precision():
    """Prevent one test's process-global matmul policy from leaking into the next."""
    original = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(original)
        # torch.compile decorators keep process-global specializations. The
        # exhaustive suite deliberately mixes many shapes and dtypes, so each
        # test must release its compile state instead of inheriting another
        # test's cache/recompile budget.
        torch._dynamo.reset()
