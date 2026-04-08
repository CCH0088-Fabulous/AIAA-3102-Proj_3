
from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch



import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=8,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.052502528, 'kernel_flop': 4831838208, 'config_args': {'KERNEL_H': 1, 'KERNEL_W': 1, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 0, 'PADDING_W': 0, 'GROUPS': 1, 'UNROLL': True, 'ALLOW_TF32': False, 'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}},

)
@triton.jit
def triton_(arg_X, arg_W, in_ptr2, out_ptr1):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 32
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 1
    IN_C = 144
    IN_H = 256
    IN_W = 256
    OUT_C = 256
    OUT_H = 256
    OUT_W = 256

    # Strides:
    stride_xn = 9437184
    stride_xc = 65536
    stride_xh = 256
    stride_xw = 1
    stride_wc_out = 144
    stride_wc_in = 1
    stride_wh = 1
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




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):
        
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
    xindex = idx_w + 256*idx_h + 65536*idx_c + 16777216*idx_n
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_c, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr1 + (tl.broadcast_to(idx_w + 256*idx_h + 65536*idx_c, [BLOCK_M, BLOCK_N])), tmp1, mask)


def get_args():
    arg_0 = rand_strided((1, 144, 256, 256), (9437184, 65536, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((256, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((1, 256, 256, 256), (16777216, 65536, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, 1024, 1, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.052502528
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
