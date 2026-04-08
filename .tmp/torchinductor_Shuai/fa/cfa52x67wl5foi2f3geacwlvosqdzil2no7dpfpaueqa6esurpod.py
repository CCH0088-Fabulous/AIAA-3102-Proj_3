

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=8,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 7, 'KERNEL_W': 7, 'STRIDE_H': 4, 'STRIDE_W': 4, 'PADDING_H': 3, 'PADDING_W': 3, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}},

)
@triton.jit
def triton_convolution2d(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 7
    KERNEL_W : tl.constexpr = 7
    STRIDE_H : tl.constexpr = 4
    STRIDE_W : tl.constexpr = 4
    PADDING_H : tl.constexpr = 3
    PADDING_W : tl.constexpr = 3
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 1
    IN_C = 3
    IN_H = 1024
    IN_W = 1024
    OUT_C = 144
    OUT_H = 256
    OUT_W = 256

    # Strides:
    stride_xn = 3145728
    stride_xc = 1048576
    stride_xh = 1024
    stride_xw = 1
    stride_wc_out = 147
    stride_wc_in = 49
    stride_wh = 7
    stride_ww = 1

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W
        
        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 256*idx_h + 65536*idx_c + 9437184*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_w + 256*idx_h + 65536*idx_c, [BLOCK_M, BLOCK_N])), acc, mask)