
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 47104}}
)
@triton.jit
def triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([1, 1], 256, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp20 = tmp3 - tmp13
    tmp21 = tl.full([1, 1], 256.0, tl.float32)
    tmp22 = (tmp19 / tmp21)
    tmp23 = tl.full([1, 1], 1e-05, tl.float32)
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = x0
    tmp32 = tl.full([1, 1], 0, tl.int64)
    tmp33 = tmp31 >= tmp32
    tmp34 = tl.full([1, 1], 6, tl.int64)
    tmp35 = tmp31 < tmp34
    tmp36 = tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])
    tmp37 = tl.full([1, 1], 0, tl.int64)
    tmp38 = tmp36 >= tmp37
    tmp39 = tl.full([1, 1], 1, tl.int64)
    tmp40 = tmp36 < tmp39
    tmp41 = tmp40 & tmp35
    tmp42 = tl.load(in_ptr3 + (tl.broadcast_to(r0_1, [XBLOCK, R0_BLOCK])), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp36 >= tmp39
    tmp44 = tl.full([1, 1], 2, tl.int64)
    tmp45 = tmp36 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tmp46 & tmp35
    tmp48 = tl.load(in_ptr4 + (tl.broadcast_to(r0_1, [XBLOCK, R0_BLOCK])), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp36 >= tmp44
    tmp50 = tl.full([1, 1], 6, tl.int64)
    tmp51 = tmp36 < tmp50
    tmp52 = tmp49 & tmp35
    tmp53 = tl.load(in_ptr5 + (r0_1 + 256*((-2) + (x0))), tmp52 & xmask, other=0.0)
    tmp54 = tl.where(tmp46, tmp48, tmp53)
    tmp55 = tl.where(tmp40, tmp42, tmp54)
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp35, tmp55, tmp56)
    tmp58 = tmp31 >= tmp34
    tmp59 = tl.full([1, 1], 8, tl.int64)
    tmp60 = tmp31 < tmp59
    tmp61 = tl.load(in_ptr6 + (r0_1 + 256*((-6) + x0)), tmp58 & xmask, other=0.0)
    tmp62 = tl.where(tmp35, tmp57, tmp61)
    tmp63 = tmp30 + tmp62
    tmp64 = tmp63.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp30, xmask)
    tl.store(out_ptr3 + (r0_1 + 256*x0), tmp64, xmask)
