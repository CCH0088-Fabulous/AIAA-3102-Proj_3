
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_clone_convolution_native_layer_norm_permute_repeat_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 132157440, 'r0_': 56624544}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_clone_convolution_native_layer_norm_permute_repeat_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp12_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x3 + 65536*r0_2), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_out_ptr0 + (x3 + 65536*r0_2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (8*((x1 % 8)) + 64*r0_2 + ((x0 % 8))), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r0_2 + 144*((x0 % 8)) + 1152*((x1 % 8)) + 9216*(x0 // 8) + 294912*(x1 // 8)), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 + tmp6
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(r0_mask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(r0_mask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(r0_mask, tmp12_weight_next, tmp12_weight)
        tl.store(in_out_ptr0 + (x3 + 65536*r0_2), tmp10, r0_mask)
    tmp13, tmp14, tmp15 = triton_helpers.welford(tmp12_mean, tmp12_m2, tmp12_weight, 1)
    tmp12 = tmp13[:, None]
    tmp16 = tmp14[:, None]
    tmp17 = tmp15[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp18 = tl.load(in_out_ptr0 + (x3 + 65536*r0_2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp18 - tmp12
        tmp20 = tl.full([1, 1], 144.0, tl.float32)
        tmp21 = (tmp16 / tmp20)
        tmp22 = tl.full([1, 1], 1e-06, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tmp30 = tmp29.to(tl.float32)
        tl.store(out_ptr2 + (r0_2 + 144*x3), tmp30, r0_mask)
