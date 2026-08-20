from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = ["repo_grape", "repo_grape_supported"]


@triton.jit
def _repo_grape_fwd_row_kernel(
    q_ptr,
    k_ptr,
    z_ptr,
    position_ids_ptr,
    inv_freq_ptr,
    alpha_ptr,
    log_scale_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    q_out_ptr,
    k_out_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    z_stride_b: tl.constexpr,
    z_stride_h: tl.constexpr,
    z_stride_s: tl.constexpr,
    pid_stride_b: tl.constexpr,
    pid_stride_s: tl.constexpr,
    log_scale_stride_h: tl.constexpr,
    q_out_stride_b: tl.constexpr,
    q_out_stride_h: tl.constexpr,
    q_out_stride_s: tl.constexpr,
    k_out_stride_b: tl.constexpr,
    k_out_stride_h: tl.constexpr,
    k_out_stride_s: tl.constexpr,
    h_k: tl.constexpr,
    s_eff: tl.constexpr,
    head_dim: tl.constexpr,
    rot_half: tl.constexpr,
    attention_scaling,
    momentum_gamma,
    rms_norm_eps,
    HEAD_P: tl.constexpr,
    SEQ_P: tl.constexpr,
    Q_PER_K: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    APPLY_MOMENTUM: tl.constexpr,
    HAS_RMS_NORM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    s_index = row % s_eff
    hk_index = (row // s_eff) % h_k
    b_index = row // (s_eff * h_k)

    hk_base = hk_index // HEAD_P
    pseudo_head = hk_index % HEAD_P
    s_base = s_index // SEQ_P
    pseudo_sequence = s_index % SEQ_P

    if HAS_POSITION_IDS:
        position = tl.load(position_ids_ptr + b_index * pid_stride_b + s_base * pid_stride_s).to(
            tl.float32
        )
    else:
        position = s_base.to(tl.float32)

    r_offsets = tl.arange(0, BLOCK_R)
    r_mask = r_offsets < rot_half
    inv_freq = tl.load(inv_freq_ptr + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    sum_cos = tl.zeros((BLOCK_R,), dtype=tl.float32)
    sum_sin = tl.zeros((BLOCK_R,), dtype=tl.float32)
    sum_cos_previous = tl.zeros((BLOCK_R,), dtype=tl.float32)
    sum_sin_previous = tl.zeros((BLOCK_R,), dtype=tl.float32)
    scale = attention_scaling.to(tl.float32)
    gamma = momentum_gamma.to(tl.float32)
    previous_valid = s_index > 0
    previous_s_index = s_index - 1
    previous_s_base = previous_s_index // SEQ_P
    previous_pseudo_sequence = previous_s_index % SEQ_P

    for query_in_group in tl.static_range(0, Q_PER_K):
        h_repo = hk_base * Q_PER_K + query_in_group
        hq_index = h_repo * HEAD_P + pseudo_head

        z_value = tl.load(
            z_ptr + b_index * z_stride_b + h_repo * z_stride_h + s_base * z_stride_s
        ).to(tl.float32)
        alpha = tl.load(alpha_ptr + h_repo).to(tl.float32)
        base_coordinate = position + alpha * (z_value - position)
        coordinate = base_coordinate * SEQ_P + pseudo_sequence.to(tl.float32)

        log_scale = tl.load(
            log_scale_ptr + h_repo * log_scale_stride_h + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        theta = inv_freq * libdevice.exp(log_scale)
        phase = coordinate * theta
        cosine = tl.cos(phase)
        sine = tl.sin(phase)
        sum_cos += cosine
        sum_sin += sine

        if APPLY_MOMENTUM:
            previous_z = tl.load(
                z_ptr + b_index * z_stride_b + h_repo * z_stride_h + previous_s_base * z_stride_s,
                mask=previous_valid,
                other=0.0,
            ).to(tl.float32)
            if HAS_POSITION_IDS:
                previous_position = tl.load(
                    position_ids_ptr + b_index * pid_stride_b + previous_s_base * pid_stride_s,
                    mask=previous_valid,
                    other=0,
                ).to(tl.float32)
            else:
                previous_position = previous_s_base.to(tl.float32)
            previous_base_coordinate = previous_position + alpha * (previous_z - previous_position)
            previous_coordinate = previous_base_coordinate * SEQ_P + previous_pseudo_sequence.to(
                tl.float32
            )
            previous_phase = previous_coordinate * theta
            previous_cosine = tl.cos(previous_phase)
            previous_sine = tl.sin(previous_phase)
            sum_cos_previous += previous_cosine
            sum_sin_previous += previous_sine

        q_base = b_index * q_stride_b + hq_index * q_stride_h + s_index * q_stride_s
        q_out_base = b_index * q_out_stride_b + hq_index * q_out_stride_h + s_index * q_out_stride_s
        q_first = tl.load(q_ptr + q_base + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
        q_second = tl.load(q_ptr + q_base + rot_half + r_offsets, mask=r_mask, other=0.0).to(
            tl.float32
        )
        if HAS_RMS_NORM:
            d_offsets = tl.arange(0, BLOCK_D)
            d_mask = d_offsets < head_dim
            q_all = tl.load(q_ptr + q_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
            q_inv_rms = libdevice.rsqrt(
                tl.sum(q_all * q_all, axis=0) / head_dim + rms_norm_eps.to(tl.float32)
            )
            q_first *= q_inv_rms * tl.load(
                q_norm_weight_ptr + r_offsets, mask=r_mask, other=0.0
            ).to(tl.float32)
            q_second *= q_inv_rms * tl.load(
                q_norm_weight_ptr + rot_half + r_offsets,
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
        scaled_cosine = cosine * scale
        scaled_sine = sine * scale
        q_rotated_first = q_first * scaled_cosine - q_second * scaled_sine
        q_rotated_second = q_second * scaled_cosine + q_first * scaled_sine
        if APPLY_MOMENTUM:
            previous_q_base = (
                b_index * q_stride_b + hq_index * q_stride_h + previous_s_index * q_stride_s
            )
            previous_q_first = tl.load(
                q_ptr + previous_q_base + r_offsets,
                mask=r_mask & previous_valid,
                other=0.0,
            ).to(tl.float32)
            previous_q_second = tl.load(
                q_ptr + previous_q_base + rot_half + r_offsets,
                mask=r_mask & previous_valid,
                other=0.0,
            ).to(tl.float32)
            if HAS_RMS_NORM:
                previous_q_all = tl.load(
                    q_ptr + previous_q_base + d_offsets,
                    mask=d_mask & previous_valid,
                    other=0.0,
                ).to(tl.float32)
                previous_q_inv_rms = libdevice.rsqrt(
                    tl.sum(previous_q_all * previous_q_all, axis=0) / head_dim
                    + rms_norm_eps.to(tl.float32)
                )
                previous_q_first *= previous_q_inv_rms * tl.load(
                    q_norm_weight_ptr + r_offsets,
                    mask=r_mask,
                    other=0.0,
                ).to(tl.float32)
                previous_q_second *= previous_q_inv_rms * tl.load(
                    q_norm_weight_ptr + rot_half + r_offsets,
                    mask=r_mask,
                    other=0.0,
                ).to(tl.float32)
            previous_q_rotated_first = scale * (
                previous_q_first * previous_cosine - previous_q_second * previous_sine
            )
            previous_q_rotated_second = scale * (
                previous_q_second * previous_cosine + previous_q_first * previous_sine
            )
            q_rotated_first = q_rotated_first + gamma * (q_rotated_first - previous_q_rotated_first)
            q_rotated_second = q_rotated_second + gamma * (
                q_rotated_second - previous_q_rotated_second
            )
        tl.store(q_out_ptr + q_out_base + r_offsets, q_rotated_first, mask=r_mask)
        tl.store(
            q_out_ptr + q_out_base + rot_half + r_offsets,
            q_rotated_second,
            mask=r_mask,
        )

        tail_offsets = tl.arange(0, BLOCK_TAIL)
        tail_dim = 2 * rot_half + tail_offsets
        tail_mask = tail_dim < head_dim
        q_tail = tl.load(q_ptr + q_base + tail_dim, mask=tail_mask, other=0.0)
        if HAS_RMS_NORM:
            q_tail = (
                q_tail.to(tl.float32)
                * q_inv_rms
                * tl.load(q_norm_weight_ptr + tail_dim, mask=tail_mask, other=0.0).to(tl.float32)
            )
        if APPLY_MOMENTUM:
            previous_q_tail = tl.load(
                q_ptr + previous_q_base + tail_dim,
                mask=tail_mask & previous_valid,
                other=0.0,
            )
            if HAS_RMS_NORM:
                previous_q_tail = (
                    previous_q_tail.to(tl.float32)
                    * previous_q_inv_rms
                    * tl.load(
                        q_norm_weight_ptr + tail_dim,
                        mask=tail_mask,
                        other=0.0,
                    ).to(tl.float32)
                )
            q_tail = q_tail + gamma * (q_tail - previous_q_tail)
        tl.store(q_out_ptr + q_out_base + tail_dim, q_tail, mask=tail_mask)

    mean_cos = sum_cos / Q_PER_K
    mean_sin = sum_sin / Q_PER_K
    resultant_sq = mean_cos * mean_cos + mean_sin * mean_sin
    well_defined = resultant_sq > 1.1920928955078125e-7
    safe_resultant_sq = tl.where(well_defined, resultant_sq, 1.0)
    inv_resultant = libdevice.rsqrt(safe_resultant_sq)
    key_cosine = tl.where(well_defined, mean_cos * inv_resultant, 1.0) * scale
    key_sine = tl.where(well_defined, mean_sin * inv_resultant, 0.0) * scale

    if APPLY_MOMENTUM:
        previous_mean_cos = sum_cos_previous / Q_PER_K
        previous_mean_sin = sum_sin_previous / Q_PER_K
        previous_resultant_sq = (
            previous_mean_cos * previous_mean_cos + previous_mean_sin * previous_mean_sin
        )
        previous_well_defined = (previous_resultant_sq > 1.1920928955078125e-7) & previous_valid
        previous_inv_resultant = libdevice.rsqrt(
            tl.where(previous_well_defined, previous_resultant_sq, 1.0)
        )
        previous_key_cosine = (
            tl.where(
                previous_well_defined,
                previous_mean_cos * previous_inv_resultant,
                1.0,
            )
            * scale
        )
        previous_key_sine = (
            tl.where(
                previous_well_defined,
                previous_mean_sin * previous_inv_resultant,
                0.0,
            )
            * scale
        )

    k_base = b_index * k_stride_b + hk_index * k_stride_h + s_index * k_stride_s
    k_out_base = b_index * k_out_stride_b + hk_index * k_out_stride_h + s_index * k_out_stride_s
    k_first = tl.load(k_ptr + k_base + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    k_second = tl.load(k_ptr + k_base + rot_half + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    if HAS_RMS_NORM:
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim
        k_all = tl.load(k_ptr + k_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        k_inv_rms = libdevice.rsqrt(
            tl.sum(k_all * k_all, axis=0) / head_dim + rms_norm_eps.to(tl.float32)
        )
        k_first *= k_inv_rms * tl.load(k_norm_weight_ptr + r_offsets, mask=r_mask, other=0.0).to(
            tl.float32
        )
        k_second *= k_inv_rms * tl.load(
            k_norm_weight_ptr + rot_half + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
    k_rotated_first = k_first * key_cosine - k_second * key_sine
    k_rotated_second = k_second * key_cosine + k_first * key_sine
    if APPLY_MOMENTUM:
        previous_k_base = (
            b_index * k_stride_b + hk_index * k_stride_h + previous_s_index * k_stride_s
        )
        previous_k_first = tl.load(
            k_ptr + previous_k_base + r_offsets,
            mask=r_mask & previous_valid,
            other=0.0,
        ).to(tl.float32)
        previous_k_second = tl.load(
            k_ptr + previous_k_base + rot_half + r_offsets,
            mask=r_mask & previous_valid,
            other=0.0,
        ).to(tl.float32)
        if HAS_RMS_NORM:
            previous_k_all = tl.load(
                k_ptr + previous_k_base + d_offsets,
                mask=d_mask & previous_valid,
                other=0.0,
            ).to(tl.float32)
            previous_k_inv_rms = libdevice.rsqrt(
                tl.sum(previous_k_all * previous_k_all, axis=0) / head_dim
                + rms_norm_eps.to(tl.float32)
            )
            previous_k_first *= previous_k_inv_rms * tl.load(
                k_norm_weight_ptr + r_offsets, mask=r_mask, other=0.0
            ).to(tl.float32)
            previous_k_second *= previous_k_inv_rms * tl.load(
                k_norm_weight_ptr + rot_half + r_offsets,
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
        previous_k_rotated_first = (
            previous_k_first * previous_key_cosine - previous_k_second * previous_key_sine
        )
        previous_k_rotated_second = (
            previous_k_second * previous_key_cosine + previous_k_first * previous_key_sine
        )
        k_rotated_first = k_rotated_first + gamma * (k_rotated_first - previous_k_rotated_first)
        k_rotated_second = k_rotated_second + gamma * (k_rotated_second - previous_k_rotated_second)
    tl.store(k_out_ptr + k_out_base + r_offsets, k_rotated_first, mask=r_mask)
    tl.store(
        k_out_ptr + k_out_base + rot_half + r_offsets,
        k_rotated_second,
        mask=r_mask,
    )

    tail_offsets = tl.arange(0, BLOCK_TAIL)
    tail_dim = 2 * rot_half + tail_offsets
    tail_mask = tail_dim < head_dim
    k_tail = tl.load(k_ptr + k_base + tail_dim, mask=tail_mask, other=0.0)
    if HAS_RMS_NORM:
        k_tail = (
            k_tail.to(tl.float32)
            * k_inv_rms
            * tl.load(k_norm_weight_ptr + tail_dim, mask=tail_mask, other=0.0).to(tl.float32)
        )
    if APPLY_MOMENTUM:
        previous_k_tail = tl.load(
            k_ptr + previous_k_base + tail_dim,
            mask=tail_mask & previous_valid,
            other=0.0,
        )
        if HAS_RMS_NORM:
            previous_k_tail = (
                previous_k_tail.to(tl.float32)
                * previous_k_inv_rms
                * tl.load(
                    k_norm_weight_ptr + tail_dim,
                    mask=tail_mask,
                    other=0.0,
                ).to(tl.float32)
            )
        k_tail = k_tail + gamma * (k_tail - previous_k_tail)
    tl.store(k_out_ptr + k_out_base + tail_dim, k_tail, mask=tail_mask)


@triton.jit
def _repo_grape_stream_rotated_row(
    x_ptr,
    norm_weight_ptr,
    b_index,
    h_index,
    s_index,
    valid,
    cosine,
    sine,
    scale,
    rms_norm_eps,
    x_stride_b: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_s: tl.constexpr,
    head_dim: tl.constexpr,
    rot_half: tl.constexpr,
    INPUT_BF16: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    r_offsets = tl.arange(0, BLOCK_R)
    r_mask = r_offsets < rot_half
    tail_offsets = tl.arange(0, BLOCK_TAIL)
    tail_dim = 2 * rot_half + tail_offsets
    tail_mask = tail_dim < head_dim
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    base = b_index * x_stride_b + h_index * x_stride_h + s_index * x_stride_s
    x_all = tl.load(
        x_ptr + base + d_offsets,
        mask=valid & d_mask,
        other=0.0,
    ).to(tl.float32)
    inv_rms = libdevice.rsqrt(
        tl.sum(x_all * x_all, axis=0) / head_dim + rms_norm_eps.to(tl.float32)
    )
    first = tl.load(x_ptr + base + r_offsets, mask=valid & r_mask, other=0.0).to(tl.float32)
    second = tl.load(
        x_ptr + base + rot_half + r_offsets,
        mask=valid & r_mask,
        other=0.0,
    ).to(tl.float32)
    first *= inv_rms * tl.load(norm_weight_ptr + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    second *= inv_rms * tl.load(norm_weight_ptr + rot_half + r_offsets, mask=r_mask, other=0.0).to(
        tl.float32
    )
    if INPUT_BF16:
        first = first.to(tl.bfloat16).to(tl.float32)
        second = second.to(tl.bfloat16).to(tl.float32)
    scaled_cosine = cosine * scale
    scaled_sine = sine * scale
    rotated_first = first * scaled_cosine - second * scaled_sine
    rotated_second = second * scaled_cosine + first * scaled_sine
    tail = tl.load(x_ptr + base + tail_dim, mask=valid & tail_mask, other=0.0).to(tl.float32)
    tail *= inv_rms * tl.load(norm_weight_ptr + tail_dim, mask=tail_mask, other=0.0).to(tl.float32)
    if INPUT_BF16:
        tail = tail.to(tl.bfloat16).to(tl.float32)
    return rotated_first, rotated_second, tail


@triton.jit
def _repo_grape_fwd_stream_qpk124_kernel(
    q_ptr,
    k_ptr,
    z_ptr,
    position_ids_ptr,
    inv_freq_ptr,
    alpha_ptr,
    log_scale_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    q_out_ptr,
    k_out_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    z_stride_b: tl.constexpr,
    z_stride_h: tl.constexpr,
    z_stride_s: tl.constexpr,
    pid_stride_b: tl.constexpr,
    pid_stride_s: tl.constexpr,
    log_scale_stride_h: tl.constexpr,
    q_out_stride_b: tl.constexpr,
    q_out_stride_h: tl.constexpr,
    q_out_stride_s: tl.constexpr,
    k_out_stride_b: tl.constexpr,
    k_out_stride_h: tl.constexpr,
    k_out_stride_s: tl.constexpr,
    h_k: tl.constexpr,
    s_eff: tl.constexpr,
    head_dim: tl.constexpr,
    rot_half: tl.constexpr,
    attention_scaling,
    momentum_gamma,
    rms_norm_eps,
    HEAD_P: tl.constexpr,
    SEQ_P: tl.constexpr,
    Q_PER_K: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    APPLY_MOMENTUM: tl.constexpr,
    HAS_RMS_NORM: tl.constexpr,
    INPUT_BF16: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(HEAD_P == 1)
    tl.static_assert((SEQ_P == 1) | (SEQ_P == 2) | (SEQ_P == 4))
    tl.static_assert((Q_PER_K == 1) | (Q_PER_K == 2) | (Q_PER_K == 4))
    tl.static_assert(HAS_RMS_NORM)
    tl.static_assert(APPLY_MOMENTUM)
    sequence_blocks = tl.cdiv(s_eff, BLOCK_S)
    program = tl.program_id(0)
    sequence_block = program % sequence_blocks
    hk_index = (program // sequence_blocks) % h_k
    b_index = program // (sequence_blocks * h_k)
    sequence_start = sequence_block * BLOCK_S
    hq0 = hk_index * Q_PER_K
    hq1 = hq0 + 1
    hq2 = hq0 + 2
    hq3 = hq0 + 3

    r_offsets = tl.arange(0, BLOCK_R)
    r_mask = r_offsets < rot_half
    tail_offsets = tl.arange(0, BLOCK_TAIL)
    tail_dim = 2 * rot_half + tail_offsets
    tail_mask = tail_dim < head_dim
    inv_freq = tl.load(inv_freq_ptr + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    alpha0 = tl.load(alpha_ptr + hq0).to(tl.float32)
    log_scale0 = tl.load(
        log_scale_ptr + hq0 * log_scale_stride_h + r_offsets,
        mask=r_mask,
        other=0.0,
    ).to(tl.float32)
    theta0 = inv_freq * libdevice.exp(log_scale0)
    if Q_PER_K >= 2:
        alpha1 = tl.load(alpha_ptr + hq1).to(tl.float32)
        log_scale1 = tl.load(
            log_scale_ptr + hq1 * log_scale_stride_h + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        theta1 = inv_freq * libdevice.exp(log_scale1)
    if Q_PER_K == 4:
        alpha2 = tl.load(alpha_ptr + hq2).to(tl.float32)
        alpha3 = tl.load(alpha_ptr + hq3).to(tl.float32)
        log_scale2 = tl.load(
            log_scale_ptr + hq2 * log_scale_stride_h + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        log_scale3 = tl.load(
            log_scale_ptr + hq3 * log_scale_stride_h + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        theta2 = inv_freq * libdevice.exp(log_scale2)
        theta3 = inv_freq * libdevice.exp(log_scale3)
    scale = attention_scaling.to(tl.float32)
    gamma = momentum_gamma.to(tl.float32)

    previous_q0_first = tl.zeros((BLOCK_R,), dtype=tl.float32)
    previous_q0_second = tl.zeros((BLOCK_R,), dtype=tl.float32)
    previous_q0_tail = tl.zeros((BLOCK_TAIL,), dtype=tl.float32)
    previous_q1_first = tl.zeros((BLOCK_R,), dtype=tl.float32)
    previous_q1_second = tl.zeros((BLOCK_R,), dtype=tl.float32)
    previous_q1_tail = tl.zeros((BLOCK_TAIL,), dtype=tl.float32)
    previous_q2_first = tl.zeros((BLOCK_R,), dtype=tl.float32)
    previous_q2_second = tl.zeros((BLOCK_R,), dtype=tl.float32)
    previous_q2_tail = tl.zeros((BLOCK_TAIL,), dtype=tl.float32)
    previous_q3_first = tl.zeros((BLOCK_R,), dtype=tl.float32)
    previous_q3_second = tl.zeros((BLOCK_R,), dtype=tl.float32)
    previous_q3_tail = tl.zeros((BLOCK_TAIL,), dtype=tl.float32)
    previous_k_first = tl.zeros((BLOCK_R,), dtype=tl.float32)
    previous_k_second = tl.zeros((BLOCK_R,), dtype=tl.float32)
    previous_k_tail = tl.zeros((BLOCK_TAIL,), dtype=tl.float32)

    for stream_step in tl.static_range(0, BLOCK_S + 1):
        s_index = sequence_start + stream_step - 1
        valid = (s_index >= 0) & (s_index < s_eff)
        base_s_index = s_index // SEQ_P
        pseudo_sequence = s_index % SEQ_P
        if HAS_POSITION_IDS:
            position = tl.load(
                position_ids_ptr + b_index * pid_stride_b + base_s_index * pid_stride_s,
                mask=valid,
                other=0,
            ).to(tl.float32)
        else:
            position = base_s_index.to(tl.float32)
        z0 = tl.load(
            z_ptr + b_index * z_stride_b + hq0 * z_stride_h + base_s_index * z_stride_s,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        coordinate0 = (position + alpha0 * (z0 - position)) * SEQ_P + pseudo_sequence
        phase0 = coordinate0 * theta0
        cosine0 = tl.cos(phase0)
        sine0 = tl.sin(phase0)
        if Q_PER_K >= 2:
            z1 = tl.load(
                z_ptr + b_index * z_stride_b + hq1 * z_stride_h + base_s_index * z_stride_s,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            coordinate1 = (position + alpha1 * (z1 - position)) * SEQ_P + pseudo_sequence
            phase1 = coordinate1 * theta1
            cosine1 = tl.cos(phase1)
            sine1 = tl.sin(phase1)
        if Q_PER_K == 4:
            z2 = tl.load(
                z_ptr + b_index * z_stride_b + hq2 * z_stride_h + base_s_index * z_stride_s,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            z3 = tl.load(
                z_ptr + b_index * z_stride_b + hq3 * z_stride_h + base_s_index * z_stride_s,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            coordinate2 = (position + alpha2 * (z2 - position)) * SEQ_P + pseudo_sequence
            coordinate3 = (position + alpha3 * (z3 - position)) * SEQ_P + pseudo_sequence
            phase2 = coordinate2 * theta2
            phase3 = coordinate3 * theta3
            cosine2 = tl.cos(phase2)
            sine2 = tl.sin(phase2)
            cosine3 = tl.cos(phase3)
            sine3 = tl.sin(phase3)

        q0_first, q0_second, q0_tail = _repo_grape_stream_rotated_row(
            q_ptr,
            q_norm_weight_ptr,
            b_index,
            hq0,
            s_index,
            valid,
            cosine0,
            sine0,
            scale,
            rms_norm_eps,
            q_stride_b,
            q_stride_h,
            q_stride_s,
            head_dim,
            rot_half,
            INPUT_BF16,
            BLOCK_R,
            BLOCK_TAIL,
            BLOCK_D,
        )
        if Q_PER_K >= 2:
            q1_first, q1_second, q1_tail = _repo_grape_stream_rotated_row(
                q_ptr,
                q_norm_weight_ptr,
                b_index,
                hq1,
                s_index,
                valid,
                cosine1,
                sine1,
                scale,
                rms_norm_eps,
                q_stride_b,
                q_stride_h,
                q_stride_s,
                head_dim,
                rot_half,
                INPUT_BF16,
                BLOCK_R,
                BLOCK_TAIL,
                BLOCK_D,
            )
        if Q_PER_K == 4:
            q2_first, q2_second, q2_tail = _repo_grape_stream_rotated_row(
                q_ptr,
                q_norm_weight_ptr,
                b_index,
                hq2,
                s_index,
                valid,
                cosine2,
                sine2,
                scale,
                rms_norm_eps,
                q_stride_b,
                q_stride_h,
                q_stride_s,
                head_dim,
                rot_half,
                INPUT_BF16,
                BLOCK_R,
                BLOCK_TAIL,
                BLOCK_D,
            )
            q3_first, q3_second, q3_tail = _repo_grape_stream_rotated_row(
                q_ptr,
                q_norm_weight_ptr,
                b_index,
                hq3,
                s_index,
                valid,
                cosine3,
                sine3,
                scale,
                rms_norm_eps,
                q_stride_b,
                q_stride_h,
                q_stride_s,
                head_dim,
                rot_half,
                INPUT_BF16,
                BLOCK_R,
                BLOCK_TAIL,
                BLOCK_D,
            )
            mean_cos = (cosine0 + cosine1 + cosine2 + cosine3) * 0.25
            mean_sin = (sine0 + sine1 + sine2 + sine3) * 0.25
        elif Q_PER_K == 2:
            mean_cos = (cosine0 + cosine1) * 0.5
            mean_sin = (sine0 + sine1) * 0.5
        else:
            mean_cos = cosine0
            mean_sin = sine0
        resultant_sq = mean_cos * mean_cos + mean_sin * mean_sin
        well_defined = resultant_sq > 1.1920928955078125e-7
        inv_resultant = libdevice.rsqrt(tl.where(well_defined, resultant_sq, 1.0))
        key_cosine = tl.where(well_defined, mean_cos * inv_resultant, 1.0)
        key_sine = tl.where(well_defined, mean_sin * inv_resultant, 0.0)
        k_first, k_second, k_tail = _repo_grape_stream_rotated_row(
            k_ptr,
            k_norm_weight_ptr,
            b_index,
            hk_index,
            s_index,
            valid,
            key_cosine,
            key_sine,
            scale,
            rms_norm_eps,
            k_stride_b,
            k_stride_h,
            k_stride_s,
            head_dim,
            rot_half,
            INPUT_BF16,
            BLOCK_R,
            BLOCK_TAIL,
            BLOCK_D,
        )

        if stream_step > 0:
            q0_out_base = b_index * q_out_stride_b + hq0 * q_out_stride_h + s_index * q_out_stride_s
            k_out_base = (
                b_index * k_out_stride_b + hk_index * k_out_stride_h + s_index * k_out_stride_s
            )
            tl.store(
                q_out_ptr + q0_out_base + r_offsets,
                q0_first + gamma * (q0_first - previous_q0_first),
                mask=valid & r_mask,
            )
            tl.store(
                q_out_ptr + q0_out_base + rot_half + r_offsets,
                q0_second + gamma * (q0_second - previous_q0_second),
                mask=valid & r_mask,
            )
            tl.store(
                q_out_ptr + q0_out_base + tail_dim,
                q0_tail + gamma * (q0_tail - previous_q0_tail),
                mask=valid & tail_mask,
            )
            if Q_PER_K >= 2:
                q1_out_base = (
                    b_index * q_out_stride_b + hq1 * q_out_stride_h + s_index * q_out_stride_s
                )
                tl.store(
                    q_out_ptr + q1_out_base + r_offsets,
                    q1_first + gamma * (q1_first - previous_q1_first),
                    mask=valid & r_mask,
                )
                tl.store(
                    q_out_ptr + q1_out_base + rot_half + r_offsets,
                    q1_second + gamma * (q1_second - previous_q1_second),
                    mask=valid & r_mask,
                )
                tl.store(
                    q_out_ptr + q1_out_base + tail_dim,
                    q1_tail + gamma * (q1_tail - previous_q1_tail),
                    mask=valid & tail_mask,
                )
            if Q_PER_K == 4:
                q2_out_base = (
                    b_index * q_out_stride_b + hq2 * q_out_stride_h + s_index * q_out_stride_s
                )
                q3_out_base = (
                    b_index * q_out_stride_b + hq3 * q_out_stride_h + s_index * q_out_stride_s
                )
                tl.store(
                    q_out_ptr + q2_out_base + r_offsets,
                    q2_first + gamma * (q2_first - previous_q2_first),
                    mask=valid & r_mask,
                )
                tl.store(
                    q_out_ptr + q2_out_base + rot_half + r_offsets,
                    q2_second + gamma * (q2_second - previous_q2_second),
                    mask=valid & r_mask,
                )
                tl.store(
                    q_out_ptr + q2_out_base + tail_dim,
                    q2_tail + gamma * (q2_tail - previous_q2_tail),
                    mask=valid & tail_mask,
                )
                tl.store(
                    q_out_ptr + q3_out_base + r_offsets,
                    q3_first + gamma * (q3_first - previous_q3_first),
                    mask=valid & r_mask,
                )
                tl.store(
                    q_out_ptr + q3_out_base + rot_half + r_offsets,
                    q3_second + gamma * (q3_second - previous_q3_second),
                    mask=valid & r_mask,
                )
                tl.store(
                    q_out_ptr + q3_out_base + tail_dim,
                    q3_tail + gamma * (q3_tail - previous_q3_tail),
                    mask=valid & tail_mask,
                )
            tl.store(
                k_out_ptr + k_out_base + r_offsets,
                k_first + gamma * (k_first - previous_k_first),
                mask=valid & r_mask,
            )
            tl.store(
                k_out_ptr + k_out_base + rot_half + r_offsets,
                k_second + gamma * (k_second - previous_k_second),
                mask=valid & r_mask,
            )
            tl.store(
                k_out_ptr + k_out_base + tail_dim,
                k_tail + gamma * (k_tail - previous_k_tail),
                mask=valid & tail_mask,
            )

        previous_q0_first = q0_first
        previous_q0_second = q0_second
        previous_q0_tail = q0_tail
        if Q_PER_K >= 2:
            previous_q1_first = q1_first
            previous_q1_second = q1_second
            previous_q1_tail = q1_tail
        if Q_PER_K == 4:
            previous_q2_first = q2_first
            previous_q2_second = q2_second
            previous_q2_tail = q2_tail
            previous_q3_first = q3_first
            previous_q3_second = q3_second
            previous_q3_tail = q3_tail
        previous_k_first = k_first
        previous_k_second = k_second
        previous_k_tail = k_tail


@triton.jit
def _repo_grape_fwd_tile_kernel(
    q_ptr,
    k_ptr,
    z_ptr,
    position_ids_ptr,
    inv_freq_ptr,
    alpha_ptr,
    log_scale_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    q_out_ptr,
    k_out_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    z_stride_b: tl.constexpr,
    z_stride_h: tl.constexpr,
    z_stride_s: tl.constexpr,
    pid_stride_b: tl.constexpr,
    pid_stride_s: tl.constexpr,
    log_scale_stride_h: tl.constexpr,
    q_out_stride_b: tl.constexpr,
    q_out_stride_h: tl.constexpr,
    q_out_stride_s: tl.constexpr,
    k_out_stride_b: tl.constexpr,
    k_out_stride_h: tl.constexpr,
    k_out_stride_s: tl.constexpr,
    h_k: tl.constexpr,
    s_eff: tl.constexpr,
    head_dim: tl.constexpr,
    rot_half: tl.constexpr,
    attention_scaling,
    momentum_gamma,
    rms_norm_eps,
    HEAD_P: tl.constexpr,
    SEQ_P: tl.constexpr,
    Q_PER_K: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    APPLY_MOMENTUM: tl.constexpr,
    HAS_RMS_NORM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    sequence_blocks = tl.cdiv(s_eff, BLOCK_S)
    program = tl.program_id(0)
    sequence_block = program % sequence_blocks
    hk_index = (program // sequence_blocks) % h_k
    b_index = program // (sequence_blocks * h_k)

    s_offsets = sequence_block * BLOCK_S + tl.arange(0, BLOCK_S)
    s_valid = s_offsets < s_eff
    previous_s_offsets = s_offsets - 1
    previous_valid = s_valid & (s_offsets > 0)
    previous_load_valid = previous_valid
    hk_base = hk_index // HEAD_P
    pseudo_head = hk_index % HEAD_P
    s_base = s_offsets // SEQ_P
    pseudo_sequence = s_offsets % SEQ_P
    previous_s_base = previous_s_offsets // SEQ_P
    previous_pseudo_sequence = previous_s_offsets % SEQ_P

    if HAS_POSITION_IDS:
        position = tl.load(
            position_ids_ptr + b_index * pid_stride_b + s_base * pid_stride_s,
            mask=s_valid,
            other=0,
        ).to(tl.float32)
        previous_position = tl.load(
            position_ids_ptr + b_index * pid_stride_b + previous_s_base * pid_stride_s,
            mask=previous_load_valid,
            other=0,
        ).to(tl.float32)
    else:
        position = s_base.to(tl.float32)
        previous_position = previous_s_base.to(tl.float32)

    r_offsets = tl.arange(0, BLOCK_R)
    r_mask = r_offsets < rot_half
    inv_freq = tl.load(inv_freq_ptr + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    sum_cos = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
    sum_sin = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
    sum_cos_previous = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
    sum_sin_previous = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
    scale = attention_scaling.to(tl.float32)
    gamma = momentum_gamma.to(tl.float32)

    tail_offsets = tl.arange(0, BLOCK_TAIL)
    tail_dim = 2 * rot_half + tail_offsets
    tail_mask = tail_dim < head_dim

    for query_in_group in tl.static_range(0, Q_PER_K):
        h_repo = hk_base * Q_PER_K + query_in_group
        hq_index = h_repo * HEAD_P + pseudo_head
        alpha = tl.load(alpha_ptr + h_repo).to(tl.float32)
        log_scale = tl.load(
            log_scale_ptr + h_repo * log_scale_stride_h + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        theta = inv_freq * libdevice.exp(log_scale)

        z_value = tl.load(
            z_ptr + b_index * z_stride_b + h_repo * z_stride_h + s_base * z_stride_s,
            mask=s_valid,
            other=0.0,
        ).to(tl.float32)
        coordinate = (position + alpha * (z_value - position)) * SEQ_P + pseudo_sequence.to(
            tl.float32
        )
        phase = coordinate[:, None] * theta[None, :]
        cosine = tl.cos(phase)
        sine = tl.sin(phase)
        sum_cos += cosine
        sum_sin += sine

        if APPLY_MOMENTUM:
            previous_z = tl.load(
                z_ptr + b_index * z_stride_b + h_repo * z_stride_h + previous_s_base * z_stride_s,
                mask=previous_load_valid,
                other=0.0,
            ).to(tl.float32)
            previous_coordinate = (
                previous_position + alpha * (previous_z - previous_position)
            ) * SEQ_P + previous_pseudo_sequence.to(tl.float32)
            previous_phase = previous_coordinate[:, None] * theta[None, :]
            previous_cosine = tl.cos(previous_phase)
            previous_sine = tl.sin(previous_phase)
            sum_cos_previous += previous_cosine
            sum_sin_previous += previous_sine

        q_base = b_index * q_stride_b + hq_index * q_stride_h + s_offsets[:, None] * q_stride_s
        q_out_base = (
            b_index * q_out_stride_b
            + hq_index * q_out_stride_h
            + s_offsets[:, None] * q_out_stride_s
        )
        q_mask = s_valid[:, None] & r_mask[None, :]
        q_first = tl.load(q_ptr + q_base + r_offsets[None, :], mask=q_mask, other=0.0).to(
            tl.float32
        )
        q_second = tl.load(
            q_ptr + q_base + rot_half + r_offsets[None, :],
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        q_tail_mask = s_valid[:, None] & tail_mask[None, :]
        q_tail = tl.load(
            q_ptr + q_base + tail_dim[None, :],
            mask=q_tail_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_RMS_NORM:
            q_inv_rms = libdevice.rsqrt(
                (
                    tl.sum(q_first * q_first, axis=1)
                    + tl.sum(q_second * q_second, axis=1)
                    + tl.sum(q_tail * q_tail, axis=1)
                )
                / head_dim
                + rms_norm_eps.to(tl.float32)
            )
            q_first *= q_inv_rms[:, None] * tl.load(
                q_norm_weight_ptr + r_offsets,
                mask=r_mask,
                other=0.0,
            )[None, :].to(tl.float32)
            q_second *= q_inv_rms[:, None] * tl.load(
                q_norm_weight_ptr + rot_half + r_offsets,
                mask=r_mask,
                other=0.0,
            )[None, :].to(tl.float32)
            q_tail *= q_inv_rms[:, None] * tl.load(
                q_norm_weight_ptr + tail_dim,
                mask=tail_mask,
                other=0.0,
            )[None, :].to(tl.float32)
        q_rotated_first = scale * (q_first * cosine - q_second * sine)
        q_rotated_second = scale * (q_second * cosine + q_first * sine)
        if APPLY_MOMENTUM:
            previous_q_base = (
                b_index * q_stride_b
                + hq_index * q_stride_h
                + previous_s_offsets[:, None] * q_stride_s
            )
            previous_q_mask = previous_load_valid[:, None] & r_mask[None, :]
            previous_q_first = tl.load(
                q_ptr + previous_q_base + r_offsets[None, :],
                mask=previous_q_mask,
                other=0.0,
            ).to(tl.float32)
            previous_q_second = tl.load(
                q_ptr + previous_q_base + rot_half + r_offsets[None, :],
                mask=previous_q_mask,
                other=0.0,
            ).to(tl.float32)
            previous_q_tail = tl.load(
                q_ptr + previous_q_base + tail_dim[None, :],
                mask=previous_load_valid[:, None] & tail_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if HAS_RMS_NORM:
                previous_q_inv_rms = libdevice.rsqrt(
                    (
                        tl.sum(previous_q_first * previous_q_first, axis=1)
                        + tl.sum(previous_q_second * previous_q_second, axis=1)
                        + tl.sum(previous_q_tail * previous_q_tail, axis=1)
                    )
                    / head_dim
                    + rms_norm_eps.to(tl.float32)
                )
                previous_q_first *= previous_q_inv_rms[:, None] * tl.load(
                    q_norm_weight_ptr + r_offsets,
                    mask=r_mask,
                    other=0.0,
                )[None, :].to(tl.float32)
                previous_q_second *= previous_q_inv_rms[:, None] * tl.load(
                    q_norm_weight_ptr + rot_half + r_offsets,
                    mask=r_mask,
                    other=0.0,
                )[None, :].to(tl.float32)
                previous_q_tail *= previous_q_inv_rms[:, None] * tl.load(
                    q_norm_weight_ptr + tail_dim,
                    mask=tail_mask,
                    other=0.0,
                )[None, :].to(tl.float32)
            previous_q_rotated_first = scale * (
                previous_q_first * previous_cosine - previous_q_second * previous_sine
            )
            previous_q_rotated_second = scale * (
                previous_q_second * previous_cosine + previous_q_first * previous_sine
            )
            q_rotated_first += gamma * (q_rotated_first - previous_q_rotated_first)
            q_rotated_second += gamma * (q_rotated_second - previous_q_rotated_second)
        tl.store(
            q_out_ptr + q_out_base + r_offsets[None, :],
            q_rotated_first,
            mask=q_mask,
        )
        tl.store(
            q_out_ptr + q_out_base + rot_half + r_offsets[None, :],
            q_rotated_second,
            mask=q_mask,
        )

        q_tail_out_base = (
            b_index * q_out_stride_b
            + hq_index * q_out_stride_h
            + s_offsets[:, None] * q_out_stride_s
        )
        if APPLY_MOMENTUM:
            q_tail += gamma * (q_tail - previous_q_tail)
        tl.store(
            q_out_ptr + q_tail_out_base + tail_dim[None, :],
            q_tail,
            mask=q_tail_mask,
        )

    mean_cos = sum_cos / Q_PER_K
    mean_sin = sum_sin / Q_PER_K
    resultant_sq = mean_cos * mean_cos + mean_sin * mean_sin
    well_defined = resultant_sq > 1.1920928955078125e-7
    inv_resultant = libdevice.rsqrt(tl.where(well_defined, resultant_sq, 1.0))
    key_cosine = tl.where(well_defined, mean_cos * inv_resultant, 1.0) * scale
    key_sine = tl.where(well_defined, mean_sin * inv_resultant, 0.0) * scale

    if APPLY_MOMENTUM:
        previous_mean_cos = sum_cos_previous / Q_PER_K
        previous_mean_sin = sum_sin_previous / Q_PER_K
        previous_resultant_sq = (
            previous_mean_cos * previous_mean_cos + previous_mean_sin * previous_mean_sin
        )
        previous_well_defined = (previous_resultant_sq > 1.1920928955078125e-7) & previous_valid[
            :, None
        ]
        previous_inv_resultant = libdevice.rsqrt(
            tl.where(previous_well_defined, previous_resultant_sq, 1.0)
        )
        previous_key_cosine = (
            tl.where(
                previous_well_defined,
                previous_mean_cos * previous_inv_resultant,
                1.0,
            )
            * scale
        )
        previous_key_sine = (
            tl.where(
                previous_well_defined,
                previous_mean_sin * previous_inv_resultant,
                0.0,
            )
            * scale
        )

    k_base = b_index * k_stride_b + hk_index * k_stride_h + s_offsets[:, None] * k_stride_s
    k_out_base = (
        b_index * k_out_stride_b + hk_index * k_out_stride_h + s_offsets[:, None] * k_out_stride_s
    )
    k_mask = s_valid[:, None] & r_mask[None, :]
    k_first = tl.load(k_ptr + k_base + r_offsets[None, :], mask=k_mask, other=0.0).to(tl.float32)
    k_second = tl.load(
        k_ptr + k_base + rot_half + r_offsets[None, :],
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    k_tail_mask = s_valid[:, None] & tail_mask[None, :]
    k_tail = tl.load(k_ptr + k_base + tail_dim[None, :], mask=k_tail_mask, other=0.0).to(tl.float32)
    if HAS_RMS_NORM:
        k_inv_rms = libdevice.rsqrt(
            (
                tl.sum(k_first * k_first, axis=1)
                + tl.sum(k_second * k_second, axis=1)
                + tl.sum(k_tail * k_tail, axis=1)
            )
            / head_dim
            + rms_norm_eps.to(tl.float32)
        )
        k_first *= k_inv_rms[:, None] * tl.load(
            k_norm_weight_ptr + r_offsets, mask=r_mask, other=0.0
        )[None, :].to(tl.float32)
        k_second *= k_inv_rms[:, None] * tl.load(
            k_norm_weight_ptr + rot_half + r_offsets,
            mask=r_mask,
            other=0.0,
        )[None, :].to(tl.float32)
        k_tail *= k_inv_rms[:, None] * tl.load(
            k_norm_weight_ptr + tail_dim, mask=tail_mask, other=0.0
        )[None, :].to(tl.float32)
    k_rotated_first = k_first * key_cosine - k_second * key_sine
    k_rotated_second = k_second * key_cosine + k_first * key_sine
    if APPLY_MOMENTUM:
        previous_k_base = (
            b_index * k_stride_b + hk_index * k_stride_h + previous_s_offsets[:, None] * k_stride_s
        )
        previous_k_mask = previous_load_valid[:, None] & r_mask[None, :]
        previous_k_first = tl.load(
            k_ptr + previous_k_base + r_offsets[None, :],
            mask=previous_k_mask,
            other=0.0,
        ).to(tl.float32)
        previous_k_second = tl.load(
            k_ptr + previous_k_base + rot_half + r_offsets[None, :],
            mask=previous_k_mask,
            other=0.0,
        ).to(tl.float32)
        previous_k_tail = tl.load(
            k_ptr + previous_k_base + tail_dim[None, :],
            mask=previous_load_valid[:, None] & tail_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_RMS_NORM:
            previous_k_inv_rms = libdevice.rsqrt(
                (
                    tl.sum(previous_k_first * previous_k_first, axis=1)
                    + tl.sum(previous_k_second * previous_k_second, axis=1)
                    + tl.sum(previous_k_tail * previous_k_tail, axis=1)
                )
                / head_dim
                + rms_norm_eps.to(tl.float32)
            )
            previous_k_first *= previous_k_inv_rms[:, None] * tl.load(
                k_norm_weight_ptr + r_offsets, mask=r_mask, other=0.0
            )[None, :].to(tl.float32)
            previous_k_second *= previous_k_inv_rms[:, None] * tl.load(
                k_norm_weight_ptr + rot_half + r_offsets,
                mask=r_mask,
                other=0.0,
            )[None, :].to(tl.float32)
            previous_k_tail *= previous_k_inv_rms[:, None] * tl.load(
                k_norm_weight_ptr + tail_dim, mask=tail_mask, other=0.0
            )[None, :].to(tl.float32)
        previous_k_rotated_first = (
            previous_k_first * previous_key_cosine - previous_k_second * previous_key_sine
        )
        previous_k_rotated_second = (
            previous_k_second * previous_key_cosine + previous_k_first * previous_key_sine
        )
        k_rotated_first += gamma * (k_rotated_first - previous_k_rotated_first)
        k_rotated_second += gamma * (k_rotated_second - previous_k_rotated_second)
    tl.store(
        k_out_ptr + k_out_base + r_offsets[None, :],
        k_rotated_first,
        mask=k_mask,
    )
    tl.store(
        k_out_ptr + k_out_base + rot_half + r_offsets[None, :],
        k_rotated_second,
        mask=k_mask,
    )

    if APPLY_MOMENTUM:
        k_tail += gamma * (k_tail - previous_k_tail)
    tl.store(
        k_out_ptr + k_out_base + tail_dim[None, :],
        k_tail,
        mask=k_tail_mask,
    )


@triton.jit
def _repo_grape_bwd_row_kernel(
    q_ptr,
    k_ptr,
    z_ptr,
    position_ids_ptr,
    inv_freq_ptr,
    alpha_ptr,
    log_scale_ptr,
    grad_q_out_ptr,
    grad_k_out_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_phase_ptr,
    grad_coordinate_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    z_stride_b: tl.constexpr,
    z_stride_h: tl.constexpr,
    z_stride_s: tl.constexpr,
    pid_stride_b: tl.constexpr,
    pid_stride_s: tl.constexpr,
    log_scale_stride_h: tl.constexpr,
    gqo_stride_b: tl.constexpr,
    gqo_stride_h: tl.constexpr,
    gqo_stride_s: tl.constexpr,
    gko_stride_b: tl.constexpr,
    gko_stride_h: tl.constexpr,
    gko_stride_s: tl.constexpr,
    gq_stride_b: tl.constexpr,
    gq_stride_h: tl.constexpr,
    gq_stride_s: tl.constexpr,
    gk_stride_b: tl.constexpr,
    gk_stride_h: tl.constexpr,
    gk_stride_s: tl.constexpr,
    h_q: tl.constexpr,
    h_k: tl.constexpr,
    s_eff: tl.constexpr,
    head_dim: tl.constexpr,
    rot_half: tl.constexpr,
    attention_scaling,
    momentum_gamma,
    HEAD_P: tl.constexpr,
    SEQ_P: tl.constexpr,
    Q_PER_K: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    APPLY_MOMENTUM: tl.constexpr,
    INPUT_BF16: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    row = tl.program_id(0)
    s_index = row % s_eff
    hk_index = (row // s_eff) % h_k
    b_index = row // (h_k * s_eff)
    base_s_index = s_index // SEQ_P
    pseudo_sequence = s_index % SEQ_P
    base_hk_index = hk_index // HEAD_P
    pseudo_head = hk_index % HEAD_P
    next_s_index = s_index + 1
    next_valid = next_s_index < s_eff

    if HAS_POSITION_IDS:
        position = tl.load(
            position_ids_ptr + b_index * pid_stride_b + base_s_index * pid_stride_s
        ).to(tl.float32)
    else:
        position = base_s_index.to(tl.float32)
    r_offsets = tl.arange(0, BLOCK_R)
    r_mask = r_offsets < rot_half
    tail_offsets = tl.arange(0, BLOCK_TAIL)
    tail_dim = 2 * rot_half + tail_offsets
    tail_mask = tail_dim < head_dim
    inv_freq = tl.load(inv_freq_ptr + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    scale = attention_scaling.to(tl.float32)
    gamma = momentum_gamma.to(tl.float32)

    sum_cos = tl.zeros((BLOCK_R,), dtype=tl.float32)
    sum_sin = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for query_in_group in tl.static_range(0, Q_PER_K):
        h_repo_index = base_hk_index * Q_PER_K + query_in_group
        z_value = tl.load(
            z_ptr + b_index * z_stride_b + h_repo_index * z_stride_h + base_s_index * z_stride_s
        ).to(tl.float32)
        alpha = tl.load(alpha_ptr + h_repo_index).to(tl.float32)
        base_coordinate = position + alpha * (z_value - position)
        coordinate = base_coordinate * SEQ_P + pseudo_sequence
        log_scale = tl.load(
            log_scale_ptr + h_repo_index * log_scale_stride_h + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        theta = inv_freq * libdevice.exp(log_scale)
        phase = coordinate * theta
        sum_cos += tl.cos(phase)
        sum_sin += tl.sin(phase)

    mean_cos = sum_cos / Q_PER_K
    mean_sin = sum_sin / Q_PER_K
    resultant_sq = mean_cos * mean_cos + mean_sin * mean_sin
    well_defined = resultant_sq > 1.1920928955078125e-7
    inv_resultant = libdevice.rsqrt(tl.where(well_defined, resultant_sq, 1.0))
    key_cosine = tl.where(well_defined, mean_cos * inv_resultant, 1.0)
    key_sine = tl.where(well_defined, mean_sin * inv_resultant, 0.0)

    k_base = b_index * k_stride_b + hk_index * k_stride_h + s_index * k_stride_s
    gko_base = b_index * gko_stride_b + hk_index * gko_stride_h + s_index * gko_stride_s
    gk_base = b_index * gk_stride_b + hk_index * gk_stride_h + s_index * gk_stride_s
    k_first = tl.load(k_ptr + k_base + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    k_second = tl.load(k_ptr + k_base + rot_half + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    gk_first = tl.load(grad_k_out_ptr + gko_base + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    gk_second = tl.load(
        grad_k_out_ptr + gko_base + rot_half + r_offsets,
        mask=r_mask,
        other=0.0,
    ).to(tl.float32)
    if APPLY_MOMENTUM:
        next_gko_base = (
            b_index * gko_stride_b + hk_index * gko_stride_h + next_s_index * gko_stride_s
        )
        gk_next_first = tl.load(
            grad_k_out_ptr + next_gko_base + r_offsets,
            mask=r_mask & next_valid,
            other=0.0,
        ).to(tl.float32)
        gk_next_second = tl.load(
            grad_k_out_ptr + next_gko_base + rot_half + r_offsets,
            mask=r_mask & next_valid,
            other=0.0,
        ).to(tl.float32)
        gk_first = gk_first + gamma * gk_first - gamma * gk_next_first
        gk_second = gk_second + gamma * gk_second - gamma * gk_next_second
    if INPUT_BF16:
        grad_k_first = (
            (
                (scale * gk_first * key_cosine).to(tl.bfloat16)
                + (scale * gk_second * key_sine).to(tl.bfloat16)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        grad_k_second = (
            (
                (-scale * gk_first * key_sine).to(tl.bfloat16)
                + (scale * gk_second * key_cosine).to(tl.bfloat16)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
    else:
        grad_k_first = scale * (gk_first * key_cosine + gk_second * key_sine)
        grad_k_second = scale * (-gk_first * key_sine + gk_second * key_cosine)
    tl.store(grad_k_ptr + gk_base + r_offsets, grad_k_first, mask=r_mask)
    tl.store(grad_k_ptr + gk_base + rot_half + r_offsets, grad_k_second, mask=r_mask)

    grad_key_cosine = scale * (gk_first * k_first + gk_second * k_second)
    grad_key_sine = scale * (-gk_first * k_second + gk_second * k_first)
    grad_inv_resultant = grad_key_cosine * mean_cos + grad_key_sine * mean_sin
    inv_resultant_cubed = inv_resultant * inv_resultant * inv_resultant
    grad_mean_cos = tl.where(
        well_defined,
        grad_key_cosine * inv_resultant - grad_inv_resultant * mean_cos * inv_resultant_cubed,
        0.0,
    )
    grad_mean_sin = tl.where(
        well_defined,
        grad_key_sine * inv_resultant - grad_inv_resultant * mean_sin * inv_resultant_cubed,
        0.0,
    )
    grad_k_tail = tl.load(grad_k_out_ptr + gko_base + tail_dim, mask=tail_mask, other=0.0).to(
        tl.float32
    )
    if APPLY_MOMENTUM:
        grad_k_tail_next = tl.load(
            grad_k_out_ptr + next_gko_base + tail_dim,
            mask=tail_mask & next_valid,
            other=0.0,
        ).to(tl.float32)
        grad_k_tail = grad_k_tail + gamma * grad_k_tail - gamma * grad_k_tail_next
    tl.store(grad_k_ptr + gk_base + tail_dim, grad_k_tail, mask=tail_mask)

    for query_in_group in tl.static_range(0, Q_PER_K):
        h_repo_index = base_hk_index * Q_PER_K + query_in_group
        hq_index = h_repo_index * HEAD_P + pseudo_head
        z_value = tl.load(
            z_ptr + b_index * z_stride_b + h_repo_index * z_stride_h + base_s_index * z_stride_s
        ).to(tl.float32)
        alpha = tl.load(alpha_ptr + h_repo_index).to(tl.float32)
        base_coordinate = position + alpha * (z_value - position)
        coordinate = base_coordinate * SEQ_P + pseudo_sequence
        log_scale = tl.load(
            log_scale_ptr + h_repo_index * log_scale_stride_h + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        theta = inv_freq * libdevice.exp(log_scale)
        phase = coordinate * theta
        cosine = tl.cos(phase)
        sine = tl.sin(phase)
        q_base = b_index * q_stride_b + hq_index * q_stride_h + s_index * q_stride_s
        gqo_base = b_index * gqo_stride_b + hq_index * gqo_stride_h + s_index * gqo_stride_s
        gq_base = b_index * gq_stride_b + hq_index * gq_stride_h + s_index * gq_stride_s
        q_first = tl.load(q_ptr + q_base + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
        q_second = tl.load(q_ptr + q_base + rot_half + r_offsets, mask=r_mask, other=0.0).to(
            tl.float32
        )
        gq_first = tl.load(grad_q_out_ptr + gqo_base + r_offsets, mask=r_mask, other=0.0).to(
            tl.float32
        )
        gq_second = tl.load(
            grad_q_out_ptr + gqo_base + rot_half + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        if APPLY_MOMENTUM:
            next_gqo_base = (
                b_index * gqo_stride_b + hq_index * gqo_stride_h + next_s_index * gqo_stride_s
            )
            gq_next_first = tl.load(
                grad_q_out_ptr + next_gqo_base + r_offsets,
                mask=r_mask & next_valid,
                other=0.0,
            ).to(tl.float32)
            gq_next_second = tl.load(
                grad_q_out_ptr + next_gqo_base + rot_half + r_offsets,
                mask=r_mask & next_valid,
                other=0.0,
            ).to(tl.float32)
            gq_first = gq_first + gamma * gq_first - gamma * gq_next_first
            gq_second = gq_second + gamma * gq_second - gamma * gq_next_second
        if INPUT_BF16:
            grad_q_first = (
                (
                    (scale * gq_first * cosine).to(tl.bfloat16)
                    + (scale * gq_second * sine).to(tl.bfloat16)
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            grad_q_second = (
                (
                    (-scale * gq_first * sine).to(tl.bfloat16)
                    + (scale * gq_second * cosine).to(tl.bfloat16)
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
        else:
            grad_q_first = scale * (gq_first * cosine + gq_second * sine)
            grad_q_second = scale * (-gq_first * sine + gq_second * cosine)
        tl.store(grad_q_ptr + gq_base + r_offsets, grad_q_first, mask=r_mask)
        tl.store(grad_q_ptr + gq_base + rot_half + r_offsets, grad_q_second, mask=r_mask)

        grad_query_phase = scale * (
            gq_first * (-q_first * sine - q_second * cosine)
            + gq_second * (-q_second * sine + q_first * cosine)
        )
        key_share = (-sine * grad_mean_cos + cosine * grad_mean_sin) / Q_PER_K
        grad_phase = grad_query_phase + key_share
        grad_phase_base = ((b_index * h_q + hq_index) * s_eff + s_index) * rot_half
        tl.store(grad_phase_ptr + grad_phase_base + r_offsets, grad_phase, mask=r_mask)
        grad_coordinate = tl.sum(grad_phase * theta, axis=0)
        grad_coordinate_offset = (b_index * h_q + hq_index) * s_eff + s_index
        tl.store(grad_coordinate_ptr + grad_coordinate_offset, grad_coordinate)

        grad_q_tail = tl.load(grad_q_out_ptr + gqo_base + tail_dim, mask=tail_mask, other=0.0).to(
            tl.float32
        )
        if APPLY_MOMENTUM:
            grad_q_tail_next = tl.load(
                grad_q_out_ptr + next_gqo_base + tail_dim,
                mask=tail_mask & next_valid,
                other=0.0,
            ).to(tl.float32)
            grad_q_tail = grad_q_tail + gamma * grad_q_tail - gamma * grad_q_tail_next
        tl.store(grad_q_ptr + gq_base + tail_dim, grad_q_tail, mask=tail_mask)


@triton.jit
def _repo_grape_bwd_tile_common_kernel(
    q_ptr,
    k_ptr,
    z_ptr,
    position_ids_ptr,
    inv_freq_ptr,
    alpha_ptr,
    log_scale_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    grad_q_out_ptr,
    grad_k_out_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_z_ptr,
    parameter_partial_ptr,
    q_norm_partial_ptr,
    k_norm_partial_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    z_stride_b: tl.constexpr,
    z_stride_h: tl.constexpr,
    z_stride_s: tl.constexpr,
    pid_stride_b: tl.constexpr,
    pid_stride_s: tl.constexpr,
    log_scale_stride_h: tl.constexpr,
    q_norm_partial_columns: tl.constexpr,
    k_norm_partial_columns: tl.constexpr,
    gqo_stride_b: tl.constexpr,
    gqo_stride_h: tl.constexpr,
    gqo_stride_s: tl.constexpr,
    gko_stride_b: tl.constexpr,
    gko_stride_h: tl.constexpr,
    gko_stride_s: tl.constexpr,
    gq_stride_b: tl.constexpr,
    gq_stride_h: tl.constexpr,
    gq_stride_s: tl.constexpr,
    gk_stride_b: tl.constexpr,
    gk_stride_h: tl.constexpr,
    gk_stride_s: tl.constexpr,
    gz_stride_b: tl.constexpr,
    gz_stride_h: tl.constexpr,
    gz_stride_s: tl.constexpr,
    h_k: tl.constexpr,
    s_eff: tl.constexpr,
    head_dim: tl.constexpr,
    rot_half: tl.constexpr,
    partial_columns: tl.constexpr,
    attention_scaling,
    momentum_gamma,
    rms_norm_eps,
    Q_PER_K: tl.constexpr,
    SEQ_P: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    APPLY_MOMENTUM: tl.constexpr,
    HAS_RMS_NORM: tl.constexpr,
    INPUT_BF16: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0)
    sequence_blocks = tl.cdiv(s_eff, BLOCK_S)
    sequence_block = program % sequence_blocks
    hk_index = (program // sequence_blocks) % h_k
    b_index = program // (sequence_blocks * h_k)
    s_offsets = sequence_block * BLOCK_S + tl.arange(0, BLOCK_S)
    s_valid = s_offsets < s_eff
    base_s_offsets = s_offsets // SEQ_P
    pseudo_sequence = s_offsets % SEQ_P
    base_block: tl.constexpr = BLOCK_S // SEQ_P
    next_offsets = s_offsets + 1
    next_valid = next_offsets < s_eff

    if HAS_POSITION_IDS:
        position = tl.load(
            position_ids_ptr + b_index * pid_stride_b + base_s_offsets * pid_stride_s,
            mask=s_valid,
            other=0,
        ).to(tl.float32)
    else:
        position = base_s_offsets.to(tl.float32)

    r_offsets = tl.arange(0, BLOCK_R)
    r_mask = r_offsets < rot_half
    inv_freq = tl.load(inv_freq_ptr + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    scale = attention_scaling.to(tl.float32)
    gamma = momentum_gamma.to(tl.float32)
    if Q_PER_K == 1:
        h_repo_0 = hk_index
        z_value_0 = tl.load(
            z_ptr + b_index * z_stride_b + h_repo_0 * z_stride_h + base_s_offsets * z_stride_s,
            mask=s_valid,
            other=0.0,
        ).to(tl.float32)
        alpha_0 = tl.load(alpha_ptr + h_repo_0).to(tl.float32)
        coordinate_0 = (position + alpha_0 * (z_value_0 - position)) * SEQ_P + pseudo_sequence
        log_scale_0 = tl.load(
            log_scale_ptr + h_repo_0 * log_scale_stride_h + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        theta_0 = inv_freq * libdevice.exp(log_scale_0)
        phase_0 = coordinate_0[:, None] * theta_0[None, :]
        cosine_0 = tl.cos(phase_0)
        sine_0 = tl.sin(phase_0)
        sum_cos = cosine_0
        sum_sin = sine_0
    elif Q_PER_K == 2:
        h_repo_0 = hk_index * 2
        h_repo_1 = h_repo_0 + 1
        z_value_0 = tl.load(
            z_ptr + b_index * z_stride_b + h_repo_0 * z_stride_h + base_s_offsets * z_stride_s,
            mask=s_valid,
            other=0.0,
        ).to(tl.float32)
        z_value_1 = tl.load(
            z_ptr + b_index * z_stride_b + h_repo_1 * z_stride_h + base_s_offsets * z_stride_s,
            mask=s_valid,
            other=0.0,
        ).to(tl.float32)
        alpha_0 = tl.load(alpha_ptr + h_repo_0).to(tl.float32)
        alpha_1 = tl.load(alpha_ptr + h_repo_1).to(tl.float32)
        coordinate_0 = (position + alpha_0 * (z_value_0 - position)) * SEQ_P + pseudo_sequence
        coordinate_1 = (position + alpha_1 * (z_value_1 - position)) * SEQ_P + pseudo_sequence
        log_scale_0 = tl.load(
            log_scale_ptr + h_repo_0 * log_scale_stride_h + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        log_scale_1 = tl.load(
            log_scale_ptr + h_repo_1 * log_scale_stride_h + r_offsets,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)
        theta_0 = inv_freq * libdevice.exp(log_scale_0)
        theta_1 = inv_freq * libdevice.exp(log_scale_1)
        phase_0 = coordinate_0[:, None] * theta_0[None, :]
        phase_1 = coordinate_1[:, None] * theta_1[None, :]
        cosine_0 = tl.cos(phase_0)
        sine_0 = tl.sin(phase_0)
        cosine_1 = tl.cos(phase_1)
        sine_1 = tl.sin(phase_1)
        sum_cos = cosine_0 + cosine_1
        sum_sin = sine_0 + sine_1
    else:
        sum_cos = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
        sum_sin = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
        for query_in_group in tl.static_range(0, Q_PER_K):
            h_repo = hk_index * Q_PER_K + query_in_group
            z_value = tl.load(
                z_ptr + b_index * z_stride_b + h_repo * z_stride_h + base_s_offsets * z_stride_s,
                mask=s_valid,
                other=0.0,
            ).to(tl.float32)
            alpha = tl.load(alpha_ptr + h_repo).to(tl.float32)
            coordinate = (position + alpha * (z_value - position)) * SEQ_P + pseudo_sequence
            log_scale = tl.load(
                log_scale_ptr + h_repo * log_scale_stride_h + r_offsets,
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
            theta = inv_freq * libdevice.exp(log_scale)
            phase = coordinate[:, None] * theta[None, :]
            sum_cos += tl.cos(phase)
            sum_sin += tl.sin(phase)

    mean_cos = sum_cos / Q_PER_K
    mean_sin = sum_sin / Q_PER_K
    resultant_sq = mean_cos * mean_cos + mean_sin * mean_sin
    well_defined = resultant_sq > 1.1920928955078125e-7
    inv_resultant = libdevice.rsqrt(tl.where(well_defined, resultant_sq, 1.0))
    key_cosine = tl.where(well_defined, mean_cos * inv_resultant, 1.0)
    key_sine = tl.where(well_defined, mean_sin * inv_resultant, 0.0)

    k_base = b_index * k_stride_b + hk_index * k_stride_h + s_offsets[:, None] * k_stride_s
    gko_base = b_index * gko_stride_b + hk_index * gko_stride_h + s_offsets[:, None] * gko_stride_s
    gk_base = b_index * gk_stride_b + hk_index * gk_stride_h + s_offsets[:, None] * gk_stride_s
    kr_mask = s_valid[:, None] & r_mask[None, :]
    tail_offsets = tl.arange(0, BLOCK_TAIL)
    tail_dim = 2 * rot_half + tail_offsets
    tail_mask = tail_dim < head_dim
    kt_mask = s_valid[:, None] & tail_mask[None, :]
    partial_column = b_index * sequence_blocks + sequence_block
    k_first_raw = tl.load(k_ptr + k_base + r_offsets[None, :], mask=kr_mask, other=0.0).to(
        tl.float32
    )
    k_second_raw = tl.load(
        k_ptr + k_base + rot_half + r_offsets[None, :], mask=kr_mask, other=0.0
    ).to(tl.float32)
    if HAS_RMS_NORM:
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim
        k_all = tl.load(
            k_ptr + k_base + d_offsets[None, :],
            mask=s_valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        k_tail_raw = tl.load(k_ptr + k_base + tail_dim[None, :], mask=kt_mask, other=0.0).to(
            tl.float32
        )
        k_inv_rms = libdevice.rsqrt(
            tl.sum(k_all * k_all, axis=1) / head_dim + rms_norm_eps.to(tl.float32)
        )
        k_weight_first = tl.load(k_norm_weight_ptr + r_offsets, mask=r_mask, other=0.0).to(
            tl.float32
        )
        k_weight_second = tl.load(
            k_norm_weight_ptr + rot_half + r_offsets, mask=r_mask, other=0.0
        ).to(tl.float32)
        k_weight_tail = tl.load(k_norm_weight_ptr + tail_dim, mask=tail_mask, other=0.0).to(
            tl.float32
        )
        k_first = (k_first_raw * k_inv_rms[:, None]) * k_weight_first[None, :]
        k_second = (k_second_raw * k_inv_rms[:, None]) * k_weight_second[None, :]
        if INPUT_BF16:
            k_first = k_first.to(tl.bfloat16).to(tl.float32)
            k_second = k_second.to(tl.bfloat16).to(tl.float32)
    else:
        k_first = k_first_raw
        k_second = k_second_raw
    gk_out_first = tl.load(
        grad_k_out_ptr + gko_base + r_offsets[None, :], mask=kr_mask, other=0.0
    ).to(tl.float32)
    gk_out_second = tl.load(
        grad_k_out_ptr + gko_base + rot_half + r_offsets[None, :],
        mask=kr_mask,
        other=0.0,
    ).to(tl.float32)
    if APPLY_MOMENTUM:
        gko_next_base = (
            b_index * gko_stride_b + hk_index * gko_stride_h + next_offsets[:, None] * gko_stride_s
        )
        next_r_mask = next_valid[:, None] & r_mask[None, :]
        gk_next_first = tl.load(
            grad_k_out_ptr + gko_next_base + r_offsets[None, :],
            mask=next_r_mask,
            other=0.0,
        ).to(tl.float32)
        gk_next_second = tl.load(
            grad_k_out_ptr + gko_next_base + rot_half + r_offsets[None, :],
            mask=next_r_mask,
            other=0.0,
        ).to(tl.float32)
        gk_out_first = gk_out_first + gamma * gk_out_first - gamma * gk_next_first
        gk_out_second = gk_out_second + gamma * gk_out_second - gamma * gk_next_second
    if INPUT_BF16:
        grad_k_normalized_first = (
            (
                (scale * gk_out_first * key_cosine).to(tl.bfloat16)
                + (scale * gk_out_second * key_sine).to(tl.bfloat16)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        grad_k_normalized_second = (
            (
                (-scale * gk_out_first * key_sine).to(tl.bfloat16)
                + (scale * gk_out_second * key_cosine).to(tl.bfloat16)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
    else:
        grad_k_normalized_first = scale * (gk_out_first * key_cosine + gk_out_second * key_sine)
        grad_k_normalized_second = scale * (-gk_out_first * key_sine + gk_out_second * key_cosine)
    if not HAS_RMS_NORM:
        tl.store(
            grad_k_ptr + gk_base + r_offsets[None, :],
            grad_k_normalized_first,
            mask=kr_mask,
        )
        tl.store(
            grad_k_ptr + gk_base + rot_half + r_offsets[None, :],
            grad_k_normalized_second,
            mask=kr_mask,
        )
    grad_key_cosine = scale * (gk_out_first * k_first + gk_out_second * k_second)
    grad_key_sine = scale * (-gk_out_first * k_second + gk_out_second * k_first)
    grad_inv_resultant = grad_key_cosine * mean_cos + grad_key_sine * mean_sin
    inv_resultant_cubed = inv_resultant * inv_resultant * inv_resultant
    grad_mean_cos = tl.where(
        well_defined,
        grad_key_cosine * inv_resultant - grad_inv_resultant * mean_cos * inv_resultant_cubed,
        0.0,
    )
    grad_mean_sin = tl.where(
        well_defined,
        grad_key_sine * inv_resultant - grad_inv_resultant * mean_sin * inv_resultant_cubed,
        0.0,
    )

    grad_k_normalized_tail = tl.load(
        grad_k_out_ptr + gko_base + tail_dim[None, :], mask=kt_mask, other=0.0
    ).to(tl.float32)
    if APPLY_MOMENTUM:
        next_t_mask = next_valid[:, None] & tail_mask[None, :]
        grad_k_tail_next = tl.load(
            grad_k_out_ptr + gko_next_base + tail_dim[None, :],
            mask=next_t_mask,
            other=0.0,
        ).to(tl.float32)
        grad_k_normalized_tail = (
            grad_k_normalized_tail + gamma * grad_k_normalized_tail - gamma * grad_k_tail_next
        )
    if HAS_RMS_NORM:
        if INPUT_BF16:
            grad_k_normalized_tail = grad_k_normalized_tail.to(tl.bfloat16).to(tl.float32)
        grad_k_weighted_first = grad_k_normalized_first * k_weight_first[None, :]
        grad_k_weighted_second = grad_k_normalized_second * k_weight_second[None, :]
        grad_k_weighted_tail = grad_k_normalized_tail * k_weight_tail[None, :]
        k_projection = (
            tl.sum(grad_k_weighted_first * k_first_raw, axis=1)
            + tl.sum(grad_k_weighted_second * k_second_raw, axis=1)
            + tl.sum(grad_k_weighted_tail * k_tail_raw, axis=1)
        ) / head_dim
        k_projection_scale = k_inv_rms * k_inv_rms * k_inv_rms * k_projection
        tl.store(
            grad_k_ptr + gk_base + r_offsets[None, :],
            k_inv_rms[:, None] * grad_k_weighted_first - k_first_raw * k_projection_scale[:, None],
            mask=kr_mask,
        )
        tl.store(
            grad_k_ptr + gk_base + rot_half + r_offsets[None, :],
            k_inv_rms[:, None] * grad_k_weighted_second
            - k_second_raw * k_projection_scale[:, None],
            mask=kr_mask,
        )
        tl.store(
            grad_k_ptr + gk_base + tail_dim[None, :],
            k_inv_rms[:, None] * grad_k_weighted_tail - k_tail_raw * k_projection_scale[:, None],
            mask=kt_mask,
        )
        k_norm_column = hk_index * partial_columns + partial_column
        tl.store(
            k_norm_partial_ptr + r_offsets * k_norm_partial_columns + k_norm_column,
            tl.sum(
                tl.where(
                    s_valid[:, None],
                    grad_k_normalized_first * k_first_raw * k_inv_rms[:, None],
                    0.0,
                ),
                axis=0,
            ),
            mask=r_mask,
        )
        tl.store(
            k_norm_partial_ptr + (rot_half + r_offsets) * k_norm_partial_columns + k_norm_column,
            tl.sum(
                tl.where(
                    s_valid[:, None],
                    grad_k_normalized_second * k_second_raw * k_inv_rms[:, None],
                    0.0,
                ),
                axis=0,
            ),
            mask=r_mask,
        )
        tl.store(
            k_norm_partial_ptr + tail_dim * k_norm_partial_columns + k_norm_column,
            tl.sum(
                tl.where(
                    s_valid[:, None],
                    grad_k_normalized_tail * k_tail_raw * k_inv_rms[:, None],
                    0.0,
                ),
                axis=0,
            ),
            mask=tail_mask,
        )
    else:
        tl.store(
            grad_k_ptr + gk_base + tail_dim[None, :],
            grad_k_normalized_tail,
            mask=kt_mask,
        )

    for query_in_group in tl.static_range(0, Q_PER_K):
        h_repo = hk_index * Q_PER_K + query_in_group
        if Q_PER_K == 1:
            z_value = z_value_0
            alpha = alpha_0
            coordinate = coordinate_0
            theta = theta_0
            cosine = cosine_0
            sine = sine_0
        elif Q_PER_K == 2:
            if query_in_group == 0:
                z_value = z_value_0
                alpha = alpha_0
                coordinate = coordinate_0
                theta = theta_0
                cosine = cosine_0
                sine = sine_0
            else:
                z_value = z_value_1
                alpha = alpha_1
                coordinate = coordinate_1
                theta = theta_1
                cosine = cosine_1
                sine = sine_1
        else:
            z_value = tl.load(
                z_ptr + b_index * z_stride_b + h_repo * z_stride_h + base_s_offsets * z_stride_s,
                mask=s_valid,
                other=0.0,
            ).to(tl.float32)
            alpha = tl.load(alpha_ptr + h_repo).to(tl.float32)
            coordinate = (position + alpha * (z_value - position)) * SEQ_P + pseudo_sequence
            log_scale = tl.load(
                log_scale_ptr + h_repo * log_scale_stride_h + r_offsets,
                mask=r_mask,
                other=0.0,
            ).to(tl.float32)
            theta = inv_freq * libdevice.exp(log_scale)
            phase = coordinate[:, None] * theta[None, :]
            cosine = tl.cos(phase)
            sine = tl.sin(phase)

        q_base = b_index * q_stride_b + h_repo * q_stride_h + s_offsets[:, None] * q_stride_s
        gqo_base = (
            b_index * gqo_stride_b + h_repo * gqo_stride_h + s_offsets[:, None] * gqo_stride_s
        )
        gq_base = b_index * gq_stride_b + h_repo * gq_stride_h + s_offsets[:, None] * gq_stride_s
        qr_mask = s_valid[:, None] & r_mask[None, :]
        q_first_raw = tl.load(q_ptr + q_base + r_offsets[None, :], mask=qr_mask, other=0.0).to(
            tl.float32
        )
        q_second_raw = tl.load(
            q_ptr + q_base + rot_half + r_offsets[None, :],
            mask=qr_mask,
            other=0.0,
        ).to(tl.float32)
        qt_mask = s_valid[:, None] & tail_mask[None, :]
        if HAS_RMS_NORM:
            q_all = tl.load(
                q_ptr + q_base + d_offsets[None, :],
                mask=s_valid[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            q_tail_raw = tl.load(q_ptr + q_base + tail_dim[None, :], mask=qt_mask, other=0.0).to(
                tl.float32
            )
            q_inv_rms = libdevice.rsqrt(
                tl.sum(q_all * q_all, axis=1) / head_dim + rms_norm_eps.to(tl.float32)
            )
            q_weight_first = tl.load(q_norm_weight_ptr + r_offsets, mask=r_mask, other=0.0).to(
                tl.float32
            )
            q_weight_second = tl.load(
                q_norm_weight_ptr + rot_half + r_offsets, mask=r_mask, other=0.0
            ).to(tl.float32)
            q_weight_tail = tl.load(q_norm_weight_ptr + tail_dim, mask=tail_mask, other=0.0).to(
                tl.float32
            )
            q_first = (q_first_raw * q_inv_rms[:, None]) * q_weight_first[None, :]
            q_second = (q_second_raw * q_inv_rms[:, None]) * q_weight_second[None, :]
            if INPUT_BF16:
                q_first = q_first.to(tl.bfloat16).to(tl.float32)
                q_second = q_second.to(tl.bfloat16).to(tl.float32)
        else:
            q_first = q_first_raw
            q_second = q_second_raw
        gq_out_first = tl.load(
            grad_q_out_ptr + gqo_base + r_offsets[None, :],
            mask=qr_mask,
            other=0.0,
        ).to(tl.float32)
        gq_out_second = tl.load(
            grad_q_out_ptr + gqo_base + rot_half + r_offsets[None, :],
            mask=qr_mask,
            other=0.0,
        ).to(tl.float32)
        if APPLY_MOMENTUM:
            gqo_next_base = (
                b_index * gqo_stride_b
                + h_repo * gqo_stride_h
                + next_offsets[:, None] * gqo_stride_s
            )
            next_qr_mask = next_valid[:, None] & r_mask[None, :]
            gq_next_first = tl.load(
                grad_q_out_ptr + gqo_next_base + r_offsets[None, :],
                mask=next_qr_mask,
                other=0.0,
            ).to(tl.float32)
            gq_next_second = tl.load(
                grad_q_out_ptr + gqo_next_base + rot_half + r_offsets[None, :],
                mask=next_qr_mask,
                other=0.0,
            ).to(tl.float32)
            gq_out_first = gq_out_first + gamma * gq_out_first - gamma * gq_next_first
            gq_out_second = gq_out_second + gamma * gq_out_second - gamma * gq_next_second
        if INPUT_BF16:
            grad_q_normalized_first = (
                (
                    (scale * gq_out_first * cosine).to(tl.bfloat16)
                    + (scale * gq_out_second * sine).to(tl.bfloat16)
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            grad_q_normalized_second = (
                (
                    (-scale * gq_out_first * sine).to(tl.bfloat16)
                    + (scale * gq_out_second * cosine).to(tl.bfloat16)
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
        else:
            grad_q_normalized_first = scale * (gq_out_first * cosine + gq_out_second * sine)
            grad_q_normalized_second = scale * (-gq_out_first * sine + gq_out_second * cosine)
        if not HAS_RMS_NORM:
            tl.store(
                grad_q_ptr + gq_base + r_offsets[None, :],
                grad_q_normalized_first,
                mask=qr_mask,
            )
            tl.store(
                grad_q_ptr + gq_base + rot_half + r_offsets[None, :],
                grad_q_normalized_second,
                mask=qr_mask,
            )
        grad_query_phase = scale * (
            gq_out_first * (-q_first * sine - q_second * cosine)
            + gq_out_second * (-q_second * sine + q_first * cosine)
        )
        key_share = (-sine * grad_mean_cos + cosine * grad_mean_sin) / Q_PER_K
        grad_phase = grad_query_phase + key_share
        grad_coordinate = tl.sum(grad_phase * theta[None, :], axis=1)
        base_offsets = sequence_block * base_block + tl.arange(0, base_block)
        base_valid = base_offsets < (s_eff // SEQ_P)
        grouped_grad_coordinate = tl.reshape(
            tl.where(s_valid, grad_coordinate, 0.0),
            (base_block, SEQ_P),
        )
        grad_base_coordinate = tl.sum(grouped_grad_coordinate, axis=1) * SEQ_P
        gz_base = b_index * gz_stride_b + h_repo * gz_stride_h + base_offsets * gz_stride_s
        tl.store(
            grad_z_ptr + gz_base,
            grad_base_coordinate * alpha,
            mask=base_valid,
        )
        log_contribution = grad_phase * coordinate[:, None] * theta[None, :]
        tl.store(
            parameter_partial_ptr + h_repo * partial_columns + partial_column,
            tl.sum(
                tl.where(
                    s_valid,
                    grad_coordinate * SEQ_P * (z_value - position),
                    0.0,
                ),
                axis=0,
            ),
        )
        tl.store(
            parameter_partial_ptr
            + ((h_k * Q_PER_K) + h_repo * rot_half + r_offsets) * partial_columns
            + partial_column,
            tl.sum(tl.where(s_valid[:, None], log_contribution, 0.0), axis=0),
            mask=r_mask,
        )

        grad_q_normalized_tail = tl.load(
            grad_q_out_ptr + gqo_base + tail_dim[None, :],
            mask=qt_mask,
            other=0.0,
        ).to(tl.float32)
        if APPLY_MOMENTUM:
            next_qt_mask = next_valid[:, None] & tail_mask[None, :]
            grad_q_tail_next = tl.load(
                grad_q_out_ptr + gqo_next_base + tail_dim[None, :],
                mask=next_qt_mask,
                other=0.0,
            ).to(tl.float32)
            grad_q_normalized_tail = (
                grad_q_normalized_tail + gamma * grad_q_normalized_tail - gamma * grad_q_tail_next
            )
        if HAS_RMS_NORM:
            if INPUT_BF16:
                grad_q_normalized_tail = grad_q_normalized_tail.to(tl.bfloat16).to(tl.float32)
            grad_q_weighted_first = grad_q_normalized_first * q_weight_first[None, :]
            grad_q_weighted_second = grad_q_normalized_second * q_weight_second[None, :]
            grad_q_weighted_tail = grad_q_normalized_tail * q_weight_tail[None, :]
            q_projection = (
                tl.sum(grad_q_weighted_first * q_first_raw, axis=1)
                + tl.sum(grad_q_weighted_second * q_second_raw, axis=1)
                + tl.sum(grad_q_weighted_tail * q_tail_raw, axis=1)
            ) / head_dim
            q_projection_scale = q_inv_rms * q_inv_rms * q_inv_rms * q_projection
            tl.store(
                grad_q_ptr + gq_base + r_offsets[None, :],
                q_inv_rms[:, None] * grad_q_weighted_first
                - q_first_raw * q_projection_scale[:, None],
                mask=qr_mask,
            )
            tl.store(
                grad_q_ptr + gq_base + rot_half + r_offsets[None, :],
                q_inv_rms[:, None] * grad_q_weighted_second
                - q_second_raw * q_projection_scale[:, None],
                mask=qr_mask,
            )
            tl.store(
                grad_q_ptr + gq_base + tail_dim[None, :],
                q_inv_rms[:, None] * grad_q_weighted_tail
                - q_tail_raw * q_projection_scale[:, None],
                mask=qt_mask,
            )
            q_norm_column = h_repo * partial_columns + partial_column
            tl.store(
                q_norm_partial_ptr + r_offsets * q_norm_partial_columns + q_norm_column,
                tl.sum(
                    tl.where(
                        s_valid[:, None],
                        grad_q_normalized_first * q_first_raw * q_inv_rms[:, None],
                        0.0,
                    ),
                    axis=0,
                ),
                mask=r_mask,
            )
            tl.store(
                q_norm_partial_ptr
                + (rot_half + r_offsets) * q_norm_partial_columns
                + q_norm_column,
                tl.sum(
                    tl.where(
                        s_valid[:, None],
                        grad_q_normalized_second * q_second_raw * q_inv_rms[:, None],
                        0.0,
                    ),
                    axis=0,
                ),
                mask=r_mask,
            )
            tl.store(
                q_norm_partial_ptr + tail_dim * q_norm_partial_columns + q_norm_column,
                tl.sum(
                    tl.where(
                        s_valid[:, None],
                        grad_q_normalized_tail * q_tail_raw * q_inv_rms[:, None],
                        0.0,
                    ),
                    axis=0,
                ),
                mask=tail_mask,
            )
        else:
            tl.store(
                grad_q_ptr + gq_base + tail_dim[None, :],
                grad_q_normalized_tail,
                mask=qt_mask,
            )


@triton.jit
def _repo_grape_dz_alpha_contrib_kernel(
    z_ptr,
    position_ids_ptr,
    alpha_ptr,
    grad_coordinate_ptr,
    grad_z_ptr,
    alpha_contrib_ptr,
    total_elements: tl.constexpr,
    batch: tl.constexpr,
    s_base: tl.constexpr,
    h_repo: tl.constexpr,
    s_eff: tl.constexpr,
    z_stride_b: tl.constexpr,
    z_stride_h: tl.constexpr,
    z_stride_s: tl.constexpr,
    grad_z_stride_b: tl.constexpr,
    grad_z_stride_h: tl.constexpr,
    grad_z_stride_s: tl.constexpr,
    pid_stride_b: tl.constexpr,
    pid_stride_s: tl.constexpr,
    HEAD_P: tl.constexpr,
    SEQ_P: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    element_mask = offsets < total_elements
    s_index = offsets % s_base
    h_index = (offsets // s_base) % h_repo
    b_index = offsets // (s_base * h_repo)

    summed_coordinate_grad = tl.zeros((BLOCK,), dtype=tl.float32)
    for pseudo_head in tl.static_range(0, HEAD_P):
        hq_index = h_index * HEAD_P + pseudo_head
        for pseudo_sequence in tl.static_range(0, SEQ_P):
            s_effective = s_index * SEQ_P + pseudo_sequence
            grad_offset = (b_index * (h_repo * HEAD_P) + hq_index) * s_eff + s_effective
            summed_coordinate_grad += tl.load(
                grad_coordinate_ptr + grad_offset,
                mask=element_mask,
                other=0.0,
            )
    grad_base_coordinate = summed_coordinate_grad * SEQ_P
    alpha = tl.load(alpha_ptr + h_index, mask=element_mask, other=0.0).to(tl.float32)
    z_offset = b_index * z_stride_b + h_index * z_stride_h + s_index * z_stride_s
    grad_z_offset = (
        b_index * grad_z_stride_b + h_index * grad_z_stride_h + s_index * grad_z_stride_s
    )
    z_value = tl.load(z_ptr + z_offset, mask=element_mask, other=0.0).to(tl.float32)
    if HAS_POSITION_IDS:
        position = tl.load(
            position_ids_ptr + b_index * pid_stride_b + s_index * pid_stride_s,
            mask=element_mask,
            other=0,
        ).to(tl.float32)
    else:
        position = s_index.to(tl.float32)
    tl.store(
        grad_z_ptr + grad_z_offset,
        grad_base_coordinate * alpha,
        mask=element_mask,
    )
    alpha_offset = h_index * (batch * s_base) + b_index * s_base + s_index
    tl.store(
        alpha_contrib_ptr + alpha_offset,
        grad_base_coordinate * (z_value - position),
        mask=element_mask,
    )


@triton.jit
def _repo_grape_row_partial_kernel(
    input_ptr,
    partial_ptr,
    columns: tl.constexpr,
    chunks: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        input_ptr + row * columns + offsets,
        mask=offsets < columns,
        other=0.0,
    ).to(tl.float32)
    tl.store(partial_ptr + row * chunks + chunk, tl.sum(values, axis=0))


@triton.jit
def _repo_grape_log_scale_partial_kernel(
    grad_phase_ptr,
    z_ptr,
    position_ids_ptr,
    inv_freq_ptr,
    alpha_ptr,
    log_scale_ptr,
    partial_ptr,
    batch: tl.constexpr,
    h_repo: tl.constexpr,
    s_base: tl.constexpr,
    s_eff: tl.constexpr,
    rot_half: tl.constexpr,
    reduction_size: tl.constexpr,
    chunks: tl.constexpr,
    z_stride_b: tl.constexpr,
    z_stride_h: tl.constexpr,
    z_stride_s: tl.constexpr,
    pid_stride_b: tl.constexpr,
    pid_stride_s: tl.constexpr,
    log_scale_stride_h: tl.constexpr,
    HEAD_P: tl.constexpr,
    SEQ_P: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    h_index = row // rot_half
    r_index = row % rot_half
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < reduction_size
    s_index = offsets % s_eff
    pseudo_head = (offsets // s_eff) % HEAD_P
    b_index = offsets // (s_eff * HEAD_P)
    s_original = s_index // SEQ_P
    pseudo_sequence = s_index % SEQ_P
    hq_index = h_index * HEAD_P + pseudo_head
    phase_offset = ((b_index * (h_repo * HEAD_P) + hq_index) * s_eff + s_index) * rot_half + r_index
    grad_phase = tl.load(grad_phase_ptr + phase_offset, mask=mask, other=0.0).to(tl.float32)
    z_value = tl.load(
        z_ptr + b_index * z_stride_b + h_index * z_stride_h + s_original * z_stride_s,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_POSITION_IDS:
        position = tl.load(
            position_ids_ptr + b_index * pid_stride_b + s_original * pid_stride_s,
            mask=mask,
            other=0,
        ).to(tl.float32)
    else:
        position = s_original.to(tl.float32)
    alpha = tl.load(alpha_ptr + h_index).to(tl.float32)
    base_coordinate = position + alpha * (z_value - position)
    coordinate = base_coordinate * SEQ_P + pseudo_sequence.to(tl.float32)
    inv_freq = tl.load(inv_freq_ptr + r_index).to(tl.float32)
    log_scale = tl.load(log_scale_ptr + h_index * log_scale_stride_h + r_index).to(tl.float32)
    theta = inv_freq * libdevice.exp(log_scale)
    contribution = grad_phase * coordinate * theta
    tl.store(
        partial_ptr + row * chunks + chunk,
        tl.sum(contribution, axis=0),
    )


@triton.jit
def _repo_grape_reduce_contiguous_chunks_kernel(
    input_ptr,
    output_ptr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(
        input_ptr + row * columns + offsets,
        mask=offsets < columns,
        other=0.0,
    ).to(tl.float32)
    tl.store(output_ptr + row, tl.sum(values, axis=0))


@triton.jit
def _repo_grape_reduce_parameter_chunks_kernel(
    input_ptr,
    output_ptr,
    compact_rows: tl.constexpr,
    columns: tl.constexpr,
    h_repo: tl.constexpr,
    rot_half: tl.constexpr,
    log_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_row = tl.program_id(0)
    is_alpha = output_row < h_repo
    log_row = output_row - h_repo
    log_head = log_row // log_width
    log_dimension = log_row % log_width
    is_active_log = (~is_alpha) & (log_dimension < rot_half)
    compact_row = tl.where(
        is_alpha,
        output_row,
        h_repo + log_head * rot_half + log_dimension,
    )
    offsets = tl.arange(0, BLOCK)
    values = tl.load(
        input_ptr + compact_row * columns + offsets,
        mask=(is_alpha | is_active_log) & (compact_row < compact_rows) & (offsets < columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(output_ptr + output_row, tl.sum(values, axis=0))


@triton.jit
def _repo_grape_reduce_small_backward_epilogue_kernel(
    parameter_partial_ptr,
    q_norm_partial_ptr,
    k_norm_partial_ptr,
    parameter_output_ptr,
    q_norm_output_ptr,
    k_norm_output_ptr,
    compact_parameter_rows: tl.constexpr,
    parameter_columns: tl.constexpr,
    q_norm_columns: tl.constexpr,
    k_norm_columns: tl.constexpr,
    h_repo: tl.constexpr,
    rot_half: tl.constexpr,
    log_width: tl.constexpr,
    head_dim: tl.constexpr,
    PARAMETER_BLOCK: tl.constexpr,
    Q_NORM_BLOCK: tl.constexpr,
    K_NORM_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    parameter_output_rows: tl.constexpr = h_repo + h_repo * log_width
    is_parameter_row = row < parameter_output_rows
    is_alpha = row < h_repo
    log_row = row - h_repo
    log_head = log_row // log_width
    log_dimension = log_row % log_width
    is_active_log = (~is_alpha) & (log_dimension < rot_half)
    compact_row = tl.where(
        is_alpha,
        row,
        h_repo + log_head * rot_half + log_dimension,
    )
    compact_row = tl.where(is_parameter_row, compact_row, 0)
    parameter_offsets = tl.arange(0, PARAMETER_BLOCK)
    parameter_values = tl.load(
        parameter_partial_ptr + compact_row * parameter_columns + parameter_offsets,
        mask=(
            is_parameter_row
            & (is_alpha | is_active_log)
            & (compact_row < compact_parameter_rows)
            & (parameter_offsets < parameter_columns)
        ),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        parameter_output_ptr + row,
        tl.sum(parameter_values, axis=0),
        mask=is_parameter_row,
    )

    is_norm_row = row < head_dim
    q_offsets = tl.arange(0, Q_NORM_BLOCK)
    q_values = tl.load(
        q_norm_partial_ptr + row * q_norm_columns + q_offsets,
        mask=is_norm_row & (q_offsets < q_norm_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        q_norm_output_ptr + row,
        tl.sum(q_values, axis=0),
        mask=is_norm_row,
    )

    k_offsets = tl.arange(0, K_NORM_BLOCK)
    k_values = tl.load(
        k_norm_partial_ptr + row * k_norm_columns + k_offsets,
        mask=is_norm_row & (k_offsets < k_norm_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        k_norm_output_ptr + row,
        tl.sum(k_values, axis=0),
        mask=is_norm_row,
    )


@triton.jit
def _rms_norm_bwd_rows_kernel(
    x_ptr,
    grad_y_ptr,
    weight_ptr,
    grad_x_ptr,
    grad_weight_partial_ptr,
    rows: tl.constexpr,
    heads: tl.constexpr,
    sequence: tl.constexpr,
    head_dim: tl.constexpr,
    x_stride_b: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_s: tl.constexpr,
    gy_stride_b: tl.constexpr,
    gy_stride_h: tl.constexpr,
    gy_stride_s: tl.constexpr,
    gx_stride_b: tl.constexpr,
    gx_stride_h: tl.constexpr,
    gx_stride_s: tl.constexpr,
    eps,
    chunks: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    chunk = tl.program_id(0)
    row_offsets = chunk * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row_offsets < rows
    s_index = row_offsets % sequence
    h_index = (row_offsets // sequence) % heads
    b_index = row_offsets // (heads * sequence)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    mask = row_mask[:, None] & d_mask[None, :]
    x_base = (
        b_index[:, None] * x_stride_b
        + h_index[:, None] * x_stride_h
        + s_index[:, None] * x_stride_s
    )
    gy_base = (
        b_index[:, None] * gy_stride_b
        + h_index[:, None] * gy_stride_h
        + s_index[:, None] * gy_stride_s
    )
    gx_base = (
        b_index[:, None] * gx_stride_b
        + h_index[:, None] * gx_stride_h
        + s_index[:, None] * gx_stride_s
    )
    x = tl.load(x_ptr + x_base + d_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
    grad_y = tl.load(grad_y_ptr + gy_base + d_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    inv_rms = libdevice.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps.to(tl.float32))
    grad_weighted = grad_y * weight[None, :]
    projection = tl.sum(grad_weighted * x, axis=1) / head_dim
    grad_x = (
        inv_rms[:, None] * grad_weighted - x * (inv_rms * inv_rms * inv_rms * projection)[:, None]
    )
    tl.store(grad_x_ptr + gx_base + d_offsets[None, :], grad_x, mask=mask)
    grad_weight = tl.sum(grad_y * x * inv_rms[:, None], axis=0)
    tl.store(
        grad_weight_partial_ptr + d_offsets * chunks + chunk,
        grad_weight,
        mask=d_mask,
    )


def _power_of_two_at_least_one(value: int) -> int:
    return max(1, triton.next_power_of_2(value))


def _select_forward_num_warps(
    sequence_block: int,
    head_dim: int,
    *,
    has_rms_norm: bool,
) -> int:
    del sequence_block, head_dim, has_rms_norm
    return 1


def rms_norm_bwd(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    block_rows: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 4 or grad_y.shape != x.shape:
        raise ValueError("x and grad_y must have matching [B,H,S,D] shapes")
    batch, heads, sequence, head_dim = x.shape
    if weight.shape != (head_dim,):
        raise ValueError("weight must match the final x dimension")
    rows = batch * heads * sequence
    chunks = triton.cdiv(rows, block_rows)
    block_d = _power_of_two_at_least_one(head_dim)
    grad_x = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
    partial = torch.empty((head_dim, chunks), dtype=torch.float32, device=x.device)
    torch.library.wrap_triton(_rms_norm_bwd_rows_kernel)[(chunks,)](
        x,
        grad_y,
        weight,
        grad_x,
        partial,
        rows,
        heads,
        sequence,
        head_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        grad_y.stride(0),
        grad_y.stride(1),
        grad_y.stride(2),
        grad_x.stride(0),
        grad_x.stride(1),
        grad_x.stride(2),
        eps,
        chunks,
        BLOCK_ROWS=block_rows,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    reduction_block = 1024
    reduced_chunks = triton.cdiv(chunks, reduction_block)
    if reduced_chunks > 1:
        reduced = torch.empty((head_dim, reduced_chunks), dtype=torch.float32, device=x.device)
        torch.library.wrap_triton(_repo_grape_row_partial_kernel)[(head_dim, reduced_chunks)](
            partial,
            reduced,
            chunks,
            reduced_chunks,
            BLOCK=reduction_block,
            num_warps=8,
        )
    else:
        reduced = partial
    grad_weight = torch.empty_like(weight, dtype=torch.float32)
    final_columns = reduced.shape[1]
    final_block = _power_of_two_at_least_one(final_columns)
    torch.library.wrap_triton(_repo_grape_reduce_contiguous_chunks_kernel)[(head_dim,)](
        reduced,
        grad_weight,
        head_dim,
        final_columns,
        BLOCK=final_block,
        num_warps=min(8, max(1, final_block // 32)),
    )
    return grad_x, grad_weight


def _reduce_contiguous_partials(partial: torch.Tensor) -> torch.Tensor:
    """Deterministically reduce an FP32 [rows, columns] partial buffer."""
    rows, columns = partial.shape
    reduction_block = 1024
    reduced_columns = triton.cdiv(columns, reduction_block)
    if reduced_columns > 1:
        reduced = torch.empty((rows, reduced_columns), dtype=torch.float32, device=partial.device)
        torch.library.wrap_triton(_repo_grape_row_partial_kernel)[(rows, reduced_columns)](
            partial,
            reduced,
            columns,
            reduced_columns,
            BLOCK=reduction_block,
            num_warps=8,
        )
    else:
        reduced = partial
    result = torch.empty((rows,), dtype=torch.float32, device=partial.device)
    final_columns = reduced.shape[1]
    final_block = _power_of_two_at_least_one(final_columns)
    torch.library.wrap_triton(_repo_grape_reduce_contiguous_chunks_kernel)[(rows,)](
        reduced,
        result,
        rows,
        final_columns,
        BLOCK=final_block,
        num_warps=min(8, max(1, final_block // 32)),
    )
    return result


@torch.library.triton_op(
    "cut_cross_entropy::repo_grape_forward",
    mutates_args=(),
)
def _repo_grape_forward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    z: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    alpha: torch.Tensor,
    log_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    attention_scaling: float,
    momentum_gamma: float,
    rms_norm_eps: float,
    head_p: int,
    sequence_pseudo_factor: int,
    q_per_k: int,
    has_position_ids: bool,
    has_rms_norm: bool,
    sequence_block: int,
    output_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, _h_q, s_eff, head_dim = q.shape
    h_k = k.shape[1]
    rot_half = inv_freq.numel()
    output_dtype = torch.bfloat16 if output_bf16 else q.dtype
    q_out = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=output_dtype, device=k.device)
    block_r = _power_of_two_at_least_one(rot_half)
    block_tail = _power_of_two_at_least_one(head_dim - 2 * rot_half)
    block_d = _power_of_two_at_least_one(head_dim)
    selected_num_warps = _select_forward_num_warps(
        sequence_block,
        head_dim,
        has_rms_norm=has_rms_norm,
    )
    pid_stride_b, pid_stride_s = position_ids.stride() if has_position_ids else (0, 0)
    common_args = (
        q,
        k,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        q_norm_weight,
        k_norm_weight,
        q_out,
        k_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        z.stride(0),
        z.stride(1),
        z.stride(2),
        pid_stride_b,
        pid_stride_s,
        log_scale.stride(0),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        h_k,
        s_eff,
        head_dim,
        rot_half,
        attention_scaling,
        momentum_gamma,
        rms_norm_eps,
    )
    common_meta = dict(
        HEAD_P=head_p,
        SEQ_P=sequence_pseudo_factor,
        Q_PER_K=q_per_k,
        HAS_POSITION_IDS=has_position_ids,
        APPLY_MOMENTUM=momentum_gamma != 0.0,
        HAS_RMS_NORM=has_rms_norm,
        BLOCK_R=block_r,
        BLOCK_TAIL=block_tail,
        BLOCK_D=block_d,
        num_warps=selected_num_warps,
        num_stages=1,
    )
    use_stream = (
        sequence_block > 1
        and has_rms_norm
        and momentum_gamma != 0.0
        and head_p == 1
        and sequence_pseudo_factor in (1, 2, 4)
        and q_per_k in (1, 2, 4)
    )
    if sequence_block == 1:
        rows = b * h_k * s_eff
        torch.library.wrap_triton(_repo_grape_fwd_row_kernel)[(rows,)](*common_args, **common_meta)
    elif use_stream:
        rows = b * h_k * triton.cdiv(s_eff, sequence_block)
        torch.library.wrap_triton(_repo_grape_fwd_stream_qpk124_kernel)[(rows,)](
            *common_args,
            BLOCK_S=sequence_block,
            INPUT_BF16=q.dtype == torch.bfloat16,
            **common_meta,
        )
    else:
        rows = b * h_k * triton.cdiv(s_eff, sequence_block)
        torch.library.wrap_triton(_repo_grape_fwd_tile_kernel)[(rows,)](
            *common_args,
            BLOCK_S=sequence_block,
            **common_meta,
        )
    return q_out, k_out


def _select_forward_geometry(
    sequence: int,
    head_dim: int,
    *,
    has_rms_norm: bool,
    supports_stream: bool = False,
) -> int:
    if sequence <= 4:
        return 1
    if supports_stream:
        return 8 if sequence >= 1024 else 4
    if has_rms_norm:
        # A row CTA keeps the RMS reduction local and exposes enough independent
        # rows to hide its latency on SM120. Wider tiles retain too many FP32
        # normalization values and lose occupancy.
        return 1
    if head_dim > 64:
        return 2
    if sequence >= 1024:
        return 8
    return 4


def _select_backward_geometry(batch: int, sequence: int, head_dim: int) -> int:
    if sequence <= 4:
        return 1
    if head_dim > 64:
        return 4
    if sequence <= 32:
        return 4
    if batch <= 2 and sequence <= 128:
        return 8
    if sequence >= 1536:
        return 16
    if sequence >= 768 or batch >= 32:
        return 8
    return 16


def _select_backward_num_warps(
    sequence_block: int,
    head_dim: int,
    *,
    has_rms_norm: bool = False,
) -> int:
    if has_rms_norm:
        return 1
    if sequence_block >= 16 or head_dim > 64:
        return 2
    return 1


def repo_grape_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    z: torch.Tensor,
    inv_freq: torch.Tensor,
    alpha: torch.Tensor,
    log_scale: torch.Tensor,
    *,
    sequence_pseudo_factor: int = 1,
    q_norm_weight: torch.Tensor | None = None,
    k_norm_weight: torch.Tensor | None = None,
) -> bool:
    if not (q.is_cuda and k.is_cuda and z.is_cuda):
        return False
    if not (q.device == k.device == z.device):
        return False
    if q.dtype not in (torch.bfloat16, torch.float32) or k.dtype != q.dtype:
        return False
    if z.dtype not in (torch.bfloat16, torch.float32):
        return False
    if q.ndim != 4 or k.ndim != 4 or z.ndim != 3:
        return False
    batch, q_heads, sequence, head_dim = q.shape
    if k.shape[0] != batch or k.shape[2:] != (sequence, head_dim):
        return False
    if sequence_pseudo_factor < 1 or sequence != z.shape[2] * sequence_pseudo_factor:
        return False
    repo_heads = z.shape[1]
    if repo_heads < 1 or q_heads % repo_heads:
        return False
    head_pseudo_factor = q_heads // repo_heads
    if k.shape[1] % head_pseudo_factor:
        return False
    base_k_heads = k.shape[1] // head_pseudo_factor
    if base_k_heads < 1 or repo_heads % base_k_heads:
        return False
    rot_half = inv_freq.numel()
    if rot_half < 1 or 2 * rot_half > head_dim or head_dim > 256:
        return False
    if alpha.shape != (repo_heads,) or log_scale.ndim != 2:
        return False
    if log_scale.shape[0] != repo_heads or log_scale.shape[1] < rot_half:
        return False
    if not all(
        tensor.is_cuda and tensor.device == q.device and tensor.dtype == torch.float32
        for tensor in (inv_freq, alpha, log_scale)
    ):
        return False
    if (q_norm_weight is None) != (k_norm_weight is None):
        return False
    if q_norm_weight is not None:
        if q_norm_weight.shape != (head_dim,) or k_norm_weight.shape != (head_dim,):
            return False
        if not (
            q_norm_weight.is_cuda
            and k_norm_weight.is_cuda
            and q_norm_weight.device == q.device
            and k_norm_weight.device == q.device
            and q_norm_weight.dtype == torch.float32
            and k_norm_weight.dtype == torch.float32
        ):
            return False
    return True


def repo_grape(
    q: torch.Tensor,
    k: torch.Tensor,
    z: torch.Tensor,
    position_ids: torch.Tensor | None,
    inv_freq: torch.Tensor,
    alpha: torch.Tensor,
    log_scale: torch.Tensor,
    attention_scaling: float,
    *,
    sequence_pseudo_factor: int = 1,
    momentum_gamma: float = 0.0,
    output_dtype: torch.dtype | None = None,
    q_norm_weight: torch.Tensor | None = None,
    k_norm_weight: torch.Tensor | None = None,
    rms_norm_eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4 or z.ndim != 3:
        raise ValueError("q/k must be rank 4 and z must be rank 3")
    b, h_q, s_eff, head_dim = q.shape
    if k.shape[0] != b or k.shape[2:] != (s_eff, head_dim):
        raise ValueError("q and k batch/sequence/head dimensions must match")
    if z.shape[0] != b or s_eff != z.shape[2] * sequence_pseudo_factor:
        raise ValueError("z length and sequence_pseudo_factor do not match q/k")
    h_repo = z.shape[1]
    if h_q % h_repo:
        raise ValueError("q heads must be divisible by REPO heads")
    head_p = h_q // h_repo
    h_k = k.shape[1]
    if h_k % head_p:
        raise ValueError("k heads must be divisible by the head pseudo factor")
    h_k_base = h_k // head_p
    if h_repo % h_k_base:
        raise ValueError("REPO heads must be divisible by base k heads")
    q_per_k = h_repo // h_k_base
    rot_half = inv_freq.numel()
    if 2 * rot_half > head_dim:
        raise ValueError("rotary dimension exceeds head dimension")
    if alpha.shape != (h_repo,) or log_scale.shape[0] != h_repo:
        raise ValueError("alpha/log_scale head dimensions do not match z")
    if log_scale.shape[1] < rot_half:
        raise ValueError("log_scale does not cover the active rotary planes")
    if (q_norm_weight is None) != (k_norm_weight is None):
        raise ValueError("q_norm_weight and k_norm_weight must be supplied together")
    has_rms_norm = q_norm_weight is not None
    if has_rms_norm:
        if q_norm_weight.shape != (head_dim,) or k_norm_weight.shape != (head_dim,):
            raise ValueError("RMSNorm weights must match head_dim")
        q_norm_weight_arg = q_norm_weight
        k_norm_weight_arg = k_norm_weight
    else:
        q_norm_weight_arg = inv_freq
        k_norm_weight_arg = inv_freq
    if position_ids is None:
        position_ids_arg = z
        has_position_ids = False
        pid_stride_b = 0
        pid_stride_s = 0
    else:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1 and b != 1:
            position_ids = position_ids.expand(b, -1)
        if position_ids.shape != (b, z.shape[2]):
            raise ValueError("position_ids must resolve to [B, S_base]")
        position_ids_arg = position_ids
        has_position_ids = True
        pid_stride_b, pid_stride_s = position_ids.stride()

    if q.dtype != k.dtype:
        raise ValueError("q and k must have the same dtype")
    resolved_output_dtype = output_dtype or q.dtype
    if resolved_output_dtype not in (q.dtype, torch.bfloat16):
        raise ValueError("output_dtype must be the input dtype or torch.bfloat16")
    supports_stream = (
        has_rms_norm
        and momentum_gamma != 0.0
        and head_p == 1
        and sequence_pseudo_factor in (1, 2, 4)
        and q_per_k in (1, 2, 4)
    )
    sequence_block = _select_forward_geometry(
        s_eff,
        head_dim,
        has_rms_norm=has_rms_norm,
        supports_stream=supports_stream,
    )
    if supports_stream and sequence_pseudo_factor == 2:
        # IHA(P=2) interleaves both pseudo-slots of each source token. Four
        # effective rows preserve that pair locality while retaining occupancy;
        # it measured best from short through 4K effective positions on SM120.
        sequence_block = 4
    elif supports_stream and b <= 2 and s_eff <= 128:
        # Tiny decode/prefill workloads need more independent CTAs than the
        # throughput geometry. Two rows measured best at B<=2 through S=128.
        sequence_block = 2
    return _repo_grape_forward_op(
        q,
        k,
        z,
        position_ids_arg,
        inv_freq,
        alpha,
        log_scale,
        q_norm_weight_arg,
        k_norm_weight_arg,
        attention_scaling,
        momentum_gamma,
        rms_norm_eps,
        head_p,
        sequence_pseudo_factor,
        q_per_k,
        has_position_ids,
        has_rms_norm,
        sequence_block,
        resolved_output_dtype == torch.bfloat16,
    )


def repo_grape_bwd_stage1(
    q: torch.Tensor,
    k: torch.Tensor,
    z: torch.Tensor,
    position_ids: torch.Tensor | None,
    inv_freq: torch.Tensor,
    alpha: torch.Tensor,
    log_scale: torch.Tensor,
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    attention_scaling: float,
    *,
    sequence_pseudo_factor: int = 1,
    momentum_gamma: float = 0.0,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, h_q, s_eff, head_dim = q.shape
    h_repo = z.shape[1]
    head_p = h_q // h_repo
    h_k = k.shape[1]
    h_k_base = h_k // head_p
    q_per_k = h_repo // h_k_base
    rot_half = inv_freq.numel()
    if position_ids is None:
        position_ids_arg = z
        has_position_ids = False
        pid_stride_b = 0
        pid_stride_s = 0
    else:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1 and b != 1:
            position_ids = position_ids.expand(b, -1)
        position_ids_arg = position_ids
        has_position_ids = True
        pid_stride_b, pid_stride_s = position_ids.stride()

    grad_q = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
    grad_k = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
    grad_phase = torch.empty((b, h_q, s_eff, rot_half), dtype=torch.float32, device=q.device)
    grad_coordinate = torch.empty((b, h_q, s_eff), dtype=torch.float32, device=q.device)
    block_r = _power_of_two_at_least_one(rot_half)
    block_tail = _power_of_two_at_least_one(head_dim - 2 * rot_half)
    rows = b * h_k * s_eff
    torch.library.wrap_triton(_repo_grape_bwd_row_kernel)[(rows,)](
        q,
        k,
        z,
        position_ids_arg,
        inv_freq,
        alpha,
        log_scale,
        grad_q_out,
        grad_k_out,
        grad_q,
        grad_k,
        grad_phase,
        grad_coordinate,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        z.stride(0),
        z.stride(1),
        z.stride(2),
        pid_stride_b,
        pid_stride_s,
        log_scale.stride(0),
        grad_q_out.stride(0),
        grad_q_out.stride(1),
        grad_q_out.stride(2),
        grad_k_out.stride(0),
        grad_k_out.stride(1),
        grad_k_out.stride(2),
        grad_q.stride(0),
        grad_q.stride(1),
        grad_q.stride(2),
        grad_k.stride(0),
        grad_k.stride(1),
        grad_k.stride(2),
        h_q,
        h_k,
        s_eff,
        head_dim,
        rot_half,
        attention_scaling,
        momentum_gamma,
        HEAD_P=head_p,
        SEQ_P=sequence_pseudo_factor,
        Q_PER_K=q_per_k,
        HAS_POSITION_IDS=has_position_ids,
        APPLY_MOMENTUM=momentum_gamma != 0.0,
        INPUT_BF16=q.dtype == torch.bfloat16,
        BLOCK_R=block_r,
        BLOCK_TAIL=block_tail,
        num_warps=num_warps,
        num_stages=1,
    )
    return grad_q, grad_k, grad_phase, grad_coordinate


def _reduce_repo_grape_partials(
    partial: torch.Tensor,
    h_repo: int,
    rot_half: int,
    log_width: int,
) -> torch.Tensor:
    compact_rows = h_repo + h_repo * rot_half
    columns = partial.shape[1]
    reduction_block = 1024
    reduced_columns = triton.cdiv(columns, reduction_block)
    if reduced_columns > 1:
        reduced = torch.empty(
            (compact_rows, reduced_columns),
            dtype=torch.float32,
            device=partial.device,
        )
        torch.library.wrap_triton(_repo_grape_row_partial_kernel)[(compact_rows, reduced_columns)](
            partial,
            reduced,
            columns,
            reduced_columns,
            BLOCK=reduction_block,
            num_warps=8,
        )
    else:
        reduced = partial
    output_rows = h_repo + h_repo * log_width
    result = torch.empty((output_rows,), dtype=torch.float32, device=partial.device)
    final_columns = reduced.shape[1]
    final_block = _power_of_two_at_least_one(final_columns)
    torch.library.wrap_triton(_repo_grape_reduce_parameter_chunks_kernel)[(output_rows,)](
        reduced,
        result,
        compact_rows,
        final_columns,
        h_repo,
        rot_half,
        log_width,
        BLOCK=final_block,
        num_warps=min(8, max(1, final_block // 32)),
    )
    return result


def _reduce_small_backward_epilogue(
    parameter_partial: torch.Tensor,
    q_norm_partial: torch.Tensor,
    k_norm_partial: torch.Tensor,
    h_repo: int,
    rot_half: int,
    log_width: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce the three small backward partial buffers with one launch."""
    compact_parameter_rows = h_repo + h_repo * rot_half
    parameter_output_rows = h_repo + h_repo * log_width
    parameter_grad = torch.empty(
        (parameter_output_rows,), dtype=torch.float32, device=parameter_partial.device
    )
    grad_q_norm_weight = torch.empty((head_dim,), dtype=torch.float32, device=q_norm_partial.device)
    grad_k_norm_weight = torch.empty((head_dim,), dtype=torch.float32, device=k_norm_partial.device)
    parameter_columns = parameter_partial.shape[1]
    q_norm_columns = q_norm_partial.shape[1]
    k_norm_columns = k_norm_partial.shape[1]
    parameter_block = _power_of_two_at_least_one(parameter_columns)
    q_norm_block = _power_of_two_at_least_one(q_norm_columns)
    k_norm_block = _power_of_two_at_least_one(k_norm_columns)
    rows = max(parameter_output_rows, head_dim)
    torch.library.wrap_triton(_repo_grape_reduce_small_backward_epilogue_kernel)[(rows,)](
        parameter_partial,
        q_norm_partial,
        k_norm_partial,
        parameter_grad,
        grad_q_norm_weight,
        grad_k_norm_weight,
        compact_parameter_rows,
        parameter_columns,
        q_norm_columns,
        k_norm_columns,
        h_repo,
        rot_half,
        log_width,
        head_dim,
        PARAMETER_BLOCK=parameter_block,
        Q_NORM_BLOCK=q_norm_block,
        K_NORM_BLOCK=k_norm_block,
        num_warps=1,
        num_stages=1,
    )
    return parameter_grad, grad_q_norm_weight, grad_k_norm_weight


def repo_grape_bwd_common(
    q: torch.Tensor,
    k: torch.Tensor,
    z: torch.Tensor,
    position_ids: torch.Tensor | None,
    inv_freq: torch.Tensor,
    alpha: torch.Tensor,
    log_scale: torch.Tensor,
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    attention_scaling: float,
    *,
    sequence_pseudo_factor: int = 1,
    momentum_gamma: float = 0.0,
    q_norm_weight: torch.Tensor | None = None,
    k_norm_weight: torch.Tensor | None = None,
    rms_norm_eps: float = 1.0e-6,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    b, h_q, s_eff, head_dim = q.shape
    h_repo = z.shape[1]
    h_k = k.shape[1]
    q_per_k = h_repo // h_k
    rot_half = inv_freq.numel()
    if position_ids is None:
        position_ids_arg = z
        has_position_ids = False
        pid_stride_b = 0
        pid_stride_s = 0
    else:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1 and b != 1:
            position_ids = position_ids.expand(b, -1)
        position_ids_arg = position_ids
        has_position_ids = True
        pid_stride_b, pid_stride_s = position_ids.stride()

    sequence_block = _select_backward_geometry(b, s_eff, head_dim)
    sequence_block = max(sequence_block, sequence_pseudo_factor)
    # GQA with four queries per key retains twice as many per-head phase and
    # gradient vectors. Eight rows measured better than sixteen on SM120 by
    # preserving occupancy while still amortizing the parameter reductions.
    if q_norm_weight is not None and q_per_k == 4 and s_eff > 32:
        sequence_block = 8
    selected_num_warps = _select_backward_num_warps(
        sequence_block,
        head_dim,
        has_rms_norm=q_norm_weight is not None,
    )
    sequence_blocks = triton.cdiv(s_eff, sequence_block)
    partial_columns = b * sequence_blocks
    grad_q = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
    grad_k = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
    grad_z = torch.empty_strided(z.shape, z.stride(), dtype=z.dtype, device=z.device)
    log_rows = h_repo * rot_half
    partial_rows = h_repo + log_rows
    parameter_partial = torch.empty(
        (partial_rows, partial_columns),
        dtype=torch.float32,
        device=z.device,
    )
    has_rms_norm = q_norm_weight is not None
    if has_rms_norm != (k_norm_weight is not None):
        raise ValueError("q_norm_weight and k_norm_weight must be provided together")
    if has_rms_norm:
        assert q_norm_weight is not None and k_norm_weight is not None
        q_norm_partial_columns = h_repo * partial_columns
        k_norm_partial_columns = h_k * partial_columns
        q_norm_partial = torch.empty(
            (head_dim, q_norm_partial_columns), dtype=torch.float32, device=q.device
        )
        k_norm_partial = torch.empty(
            (head_dim, k_norm_partial_columns), dtype=torch.float32, device=k.device
        )
        q_norm_weight_arg = q_norm_weight
        k_norm_weight_arg = k_norm_weight
    else:
        q_norm_partial_columns = 1
        k_norm_partial_columns = 1
        q_norm_partial = parameter_partial
        k_norm_partial = parameter_partial
        q_norm_weight_arg = inv_freq
        k_norm_weight_arg = inv_freq
    block_r = _power_of_two_at_least_one(rot_half)
    block_tail = _power_of_two_at_least_one(head_dim - 2 * rot_half)
    block_d = _power_of_two_at_least_one(head_dim)
    grid = (b * h_k * sequence_blocks,)
    torch.library.wrap_triton(_repo_grape_bwd_tile_common_kernel)[grid](
        q,
        k,
        z,
        position_ids_arg,
        inv_freq,
        alpha,
        log_scale,
        q_norm_weight_arg,
        k_norm_weight_arg,
        grad_q_out,
        grad_k_out,
        grad_q,
        grad_k,
        grad_z,
        parameter_partial,
        q_norm_partial,
        k_norm_partial,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        z.stride(0),
        z.stride(1),
        z.stride(2),
        pid_stride_b,
        pid_stride_s,
        log_scale.stride(0),
        q_norm_partial_columns,
        k_norm_partial_columns,
        grad_q_out.stride(0),
        grad_q_out.stride(1),
        grad_q_out.stride(2),
        grad_k_out.stride(0),
        grad_k_out.stride(1),
        grad_k_out.stride(2),
        grad_q.stride(0),
        grad_q.stride(1),
        grad_q.stride(2),
        grad_k.stride(0),
        grad_k.stride(1),
        grad_k.stride(2),
        grad_z.stride(0),
        grad_z.stride(1),
        grad_z.stride(2),
        h_k,
        s_eff,
        head_dim,
        rot_half,
        partial_columns,
        attention_scaling,
        momentum_gamma,
        rms_norm_eps,
        Q_PER_K=q_per_k,
        SEQ_P=sequence_pseudo_factor,
        HAS_POSITION_IDS=has_position_ids,
        APPLY_MOMENTUM=momentum_gamma != 0.0,
        HAS_RMS_NORM=has_rms_norm,
        INPUT_BF16=q.dtype == torch.bfloat16,
        BLOCK_S=sequence_block,
        BLOCK_R=block_r,
        BLOCK_TAIL=block_tail,
        BLOCK_D=block_d,
        num_warps=selected_num_warps,
        num_stages=1,
    )
    use_small_epilogue = (
        has_rms_norm
        and q_norm_partial_columns <= 256
        and k_norm_partial_columns <= 256
        and partial_columns <= 256
    )
    if use_small_epilogue:
        parameter_grad, grad_q_norm_weight, grad_k_norm_weight = _reduce_small_backward_epilogue(
            parameter_partial,
            q_norm_partial,
            k_norm_partial,
            h_repo,
            rot_half,
            log_scale.shape[1],
            head_dim,
        )
    else:
        parameter_grad = _reduce_repo_grape_partials(
            parameter_partial,
            h_repo,
            rot_half,
            log_scale.shape[1],
        )
        if has_rms_norm:
            grad_q_norm_weight = _reduce_contiguous_partials(q_norm_partial)
            grad_k_norm_weight = _reduce_contiguous_partials(k_norm_partial)
        else:
            grad_q_norm_weight = None
            grad_k_norm_weight = None
    grad_alpha = parameter_grad[:h_repo]
    grad_log_scale = parameter_grad[h_repo:].reshape_as(log_scale)
    return (
        grad_q,
        grad_k,
        grad_z,
        grad_alpha,
        grad_log_scale,
        grad_q_norm_weight,
        grad_k_norm_weight,
    )


def repo_grape_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    z: torch.Tensor,
    position_ids: torch.Tensor | None,
    inv_freq: torch.Tensor,
    alpha: torch.Tensor,
    log_scale: torch.Tensor,
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    attention_scaling: float,
    *,
    sequence_pseudo_factor: int = 1,
    momentum_gamma: float = 0.0,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h_repo = z.shape[1]
    head_p = q.shape[1] // h_repo
    if head_p == 1 and sequence_pseudo_factor in (1, 2, 4):
        common_grads = repo_grape_bwd_common(
            q,
            k,
            z,
            position_ids,
            inv_freq,
            alpha,
            log_scale,
            grad_q_out,
            grad_k_out,
            attention_scaling,
            sequence_pseudo_factor=sequence_pseudo_factor,
            momentum_gamma=momentum_gamma,
        )
        return common_grads[:5]
    grad_q, grad_k, grad_phase, grad_coordinate = repo_grape_bwd_stage1(
        q,
        k,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        grad_q_out,
        grad_k_out,
        attention_scaling,
        sequence_pseudo_factor=sequence_pseudo_factor,
        momentum_gamma=momentum_gamma,
        num_warps=num_warps,
    )
    b, h_q, s_eff, _ = q.shape
    s_base = z.shape[2]
    rot_half = inv_freq.numel()
    if position_ids is None:
        position_ids_arg = z
        has_position_ids = False
        pid_stride_b = 0
        pid_stride_s = 0
    else:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1 and b != 1:
            position_ids = position_ids.expand(b, -1)
        position_ids_arg = position_ids
        has_position_ids = True
        pid_stride_b, pid_stride_s = position_ids.stride()

    grad_z = torch.empty_strided(z.shape, z.stride(), dtype=z.dtype, device=z.device)
    alpha_contrib = torch.empty((h_repo, b * s_base), device=z.device, dtype=torch.float32)
    total_z = b * h_repo * s_base
    point_block = 256
    torch.library.wrap_triton(_repo_grape_dz_alpha_contrib_kernel)[
        (triton.cdiv(total_z, point_block),)
    ](
        z,
        position_ids_arg,
        alpha,
        grad_coordinate,
        grad_z,
        alpha_contrib,
        total_z,
        b,
        s_base,
        h_repo,
        s_eff,
        z.stride(0),
        z.stride(1),
        z.stride(2),
        grad_z.stride(0),
        grad_z.stride(1),
        grad_z.stride(2),
        pid_stride_b,
        pid_stride_s,
        HEAD_P=head_p,
        SEQ_P=sequence_pseudo_factor,
        HAS_POSITION_IDS=has_position_ids,
        BLOCK=point_block,
        num_warps=4,
    )

    reduction_block = 256
    alpha_columns = b * s_base
    alpha_chunks = triton.cdiv(alpha_columns, reduction_block)
    alpha_partial = torch.empty((h_repo, alpha_chunks), device=z.device, dtype=torch.float32)
    torch.library.wrap_triton(_repo_grape_row_partial_kernel)[(h_repo, alpha_chunks)](
        alpha_contrib,
        alpha_partial,
        alpha_columns,
        alpha_chunks,
        BLOCK=reduction_block,
        num_warps=8,
    )
    grad_alpha = torch.empty_like(alpha, dtype=torch.float32)
    alpha_final_block = _power_of_two_at_least_one(alpha_chunks)
    torch.library.wrap_triton(_repo_grape_reduce_contiguous_chunks_kernel)[(h_repo,)](
        alpha_partial,
        grad_alpha,
        h_repo,
        alpha_chunks,
        BLOCK=alpha_final_block,
        num_warps=min(8, max(1, alpha_final_block // 32)),
    )

    log_rows = h_repo * rot_half
    log_reduction_size = b * head_p * s_eff
    log_chunks = triton.cdiv(log_reduction_size, reduction_block)
    log_partial = torch.empty((log_rows, log_chunks), device=z.device, dtype=torch.float32)
    torch.library.wrap_triton(_repo_grape_log_scale_partial_kernel)[(log_rows, log_chunks)](
        grad_phase,
        z,
        position_ids_arg,
        inv_freq,
        alpha,
        log_scale,
        log_partial,
        b,
        h_repo,
        s_base,
        s_eff,
        rot_half,
        log_reduction_size,
        log_chunks,
        z.stride(0),
        z.stride(1),
        z.stride(2),
        pid_stride_b,
        pid_stride_s,
        log_scale.stride(0),
        HEAD_P=head_p,
        SEQ_P=sequence_pseudo_factor,
        HAS_POSITION_IDS=has_position_ids,
        BLOCK=reduction_block,
        num_warps=8,
    )
    grad_log_active = torch.empty((log_rows,), device=z.device, dtype=torch.float32)
    log_final_block = _power_of_two_at_least_one(log_chunks)
    torch.library.wrap_triton(_repo_grape_reduce_contiguous_chunks_kernel)[(log_rows,)](
        log_partial,
        grad_log_active,
        log_rows,
        log_chunks,
        BLOCK=log_final_block,
        num_warps=min(8, max(1, log_final_block // 32)),
    )
    grad_log_scale = torch.zeros_like(log_scale, dtype=torch.float32)
    grad_log_scale[:, :rot_half] = grad_log_active.reshape(h_repo, rot_half)
    return grad_q, grad_k, grad_z, grad_alpha, grad_log_scale


def repo_grape_bwd_with_norm(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    z: torch.Tensor,
    position_ids: torch.Tensor | None,
    inv_freq: torch.Tensor,
    alpha: torch.Tensor,
    log_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    attention_scaling: float,
    *,
    rms_norm_eps: float,
    sequence_pseudo_factor: int = 1,
    momentum_gamma: float = 0.0,
    num_warps: int = 1,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    h_repo = z.shape[1]
    head_p = q_raw.shape[1] // h_repo
    if head_p == 1 and sequence_pseudo_factor in (1, 2, 4):
        common_grads = repo_grape_bwd_common(
            q_raw,
            k_raw,
            z,
            position_ids,
            inv_freq,
            alpha,
            log_scale,
            grad_q_out,
            grad_k_out,
            attention_scaling,
            sequence_pseudo_factor=sequence_pseudo_factor,
            momentum_gamma=momentum_gamma,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            rms_norm_eps=rms_norm_eps,
        )
        assert common_grads[5] is not None and common_grads[6] is not None
        return (
            common_grads[0],
            common_grads[1],
            common_grads[2],
            common_grads[3],
            common_grads[4],
            common_grads[5],
            common_grads[6],
        )
    q_normalized = torch.nn.functional.rms_norm(
        q_raw, (q_raw.shape[-1],), q_norm_weight, rms_norm_eps
    )
    k_normalized = torch.nn.functional.rms_norm(
        k_raw, (k_raw.shape[-1],), k_norm_weight, rms_norm_eps
    )
    grad_q_normalized, grad_k_normalized, grad_z, grad_alpha, grad_log_scale = repo_grape_bwd(
        q_normalized,
        k_normalized,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        grad_q_out,
        grad_k_out,
        attention_scaling,
        sequence_pseudo_factor=sequence_pseudo_factor,
        momentum_gamma=momentum_gamma,
        num_warps=num_warps,
    )
    grad_q, grad_q_norm_weight = rms_norm_bwd(
        q_raw,
        grad_q_normalized,
        q_norm_weight,
        rms_norm_eps,
    )
    grad_k, grad_k_norm_weight = rms_norm_bwd(
        k_raw,
        grad_k_normalized,
        k_norm_weight,
        rms_norm_eps,
    )
    return (
        grad_q,
        grad_k,
        grad_z,
        grad_alpha,
        grad_log_scale,
        grad_q_norm_weight,
        grad_k_norm_weight,
    )


def _repo_grape_setup_context(ctx, inputs, output) -> None:
    del output
    (
        q,
        k,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        q_norm_weight,
        k_norm_weight,
        attention_scaling,
        momentum_gamma,
        rms_norm_eps,
        _head_p,
        sequence_pseudo_factor,
        _q_per_k,
        _has_position_ids,
        has_rms_norm,
        _sequence_block,
        _output_bf16,
    ) = inputs
    ctx.save_for_backward(
        q,
        k,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        q_norm_weight,
        k_norm_weight,
    )
    ctx.attention_scaling = attention_scaling
    ctx.momentum_gamma = momentum_gamma
    ctx.rms_norm_eps = rms_norm_eps
    ctx.sequence_pseudo_factor = sequence_pseudo_factor
    ctx.has_rms_norm = has_rms_norm


def _repo_grape_autograd_backward(ctx, grad_q_out, grad_k_out):
    (
        q,
        k,
        z,
        position_ids,
        inv_freq,
        alpha,
        log_scale,
        q_norm_weight,
        k_norm_weight,
    ) = ctx.saved_tensors
    if ctx.has_rms_norm:
        (
            grad_q,
            grad_k,
            grad_z,
            grad_alpha,
            grad_log_scale,
            grad_q_norm_weight,
            grad_k_norm_weight,
        ) = repo_grape_bwd_with_norm(
            q,
            k,
            z,
            position_ids,
            inv_freq,
            alpha,
            log_scale,
            q_norm_weight,
            k_norm_weight,
            grad_q_out,
            grad_k_out,
            ctx.attention_scaling,
            rms_norm_eps=ctx.rms_norm_eps,
            sequence_pseudo_factor=ctx.sequence_pseudo_factor,
            momentum_gamma=ctx.momentum_gamma,
            num_warps=1,
        )
    else:
        grad_q, grad_k, grad_z, grad_alpha, grad_log_scale = repo_grape_bwd(
            q,
            k,
            z,
            position_ids,
            inv_freq,
            alpha,
            log_scale,
            grad_q_out,
            grad_k_out,
            ctx.attention_scaling,
            sequence_pseudo_factor=ctx.sequence_pseudo_factor,
            momentum_gamma=ctx.momentum_gamma,
            num_warps=1,
        )
        grad_q_norm_weight = None
        grad_k_norm_weight = None
    return (
        grad_q,
        grad_k,
        grad_z,
        None,
        None,
        grad_alpha,
        grad_log_scale,
        grad_q_norm_weight,
        grad_k_norm_weight,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    _repo_grape_forward_op,
    _repo_grape_autograd_backward,
    setup_context=_repo_grape_setup_context,
)
