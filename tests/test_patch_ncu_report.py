from benchmark.patch_ncu_report import (
    DRAM_BYTES,
    DURATION_NS,
    FADD,
    FFMA,
    FMUL,
    HADD,
    HFMA,
    HMUL,
    TENSOR_FLOPS,
    summarize_rows,
)


def test_summarize_rows_skips_units_and_counts_packed_half_operations() -> None:
    units = {"ID": "", TENSOR_FLOPS: "flop"}
    kernel = {
        "ID": "0",
        TENSOR_FLOPS: "100",
        FADD: "2",
        FFMA: "3",
        FMUL: "5",
        HADD: "7",
        HFMA: "11",
        HMUL: "13",
        DRAM_BYTES: "17",
        DURATION_NS: "19",
    }

    assert summarize_rows([units, kernel]) == {
        "kernel_count": 1,
        "tensor_bf16_fp32_flops": 100,
        "cuda_core_fp32_flops": 13,
        "cuda_core_fp16_flops": 84,
        "measured_floating_point_ops": 197,
        "dram_bytes": 17,
        "summed_kernel_duration_ns": 19,
    }
