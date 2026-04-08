
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 38912}}
)
@triton.jit
def triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 256, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tmp1 - tmp11
    tmp19 = tl.full([1, 1], 256.0, tl.float32)
    tmp20 = (tmp17 / tmp19)
    tmp21 = tl.full([1, 1], 1e-05, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = x0
    tmp30 = tl.full([1, 1], 0, tl.int64)
    tmp31 = tmp29 >= tmp30
    tmp32 = tl.full([1, 1], 6, tl.int64)
    tmp33 = tmp29 < tmp32
    tmp34 = tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])
    tmp35 = tl.full([1, 1], 0, tl.int64)
    tmp36 = tmp34 >= tmp35
    tmp37 = tl.full([1, 1], 1, tl.int64)
    tmp38 = tmp34 < tmp37
    tmp39 = tmp38 & tmp33
    tmp40 = tl.load(in_ptr3 + (tl.broadcast_to(r0_1, [XBLOCK, R0_BLOCK])), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp34 >= tmp37
    tmp42 = tl.full([1, 1], 2, tl.int64)
    tmp43 = tmp34 < tmp42
    tmp44 = tmp41 & tmp43
    tmp45 = tmp44 & tmp33
    tmp46 = tl.load(in_ptr4 + (tl.broadcast_to(r0_1, [XBLOCK, R0_BLOCK])), tmp45 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp34 >= tmp42
    tmp48 = tl.full([1, 1], 6, tl.int64)
    tmp49 = tmp34 < tmp48
    tmp50 = tmp47 & tmp33
    tmp51 = tl.load(in_ptr5 + (r0_1 + 256*((-2) + (x0))), tmp50 & xmask, other=0.0)
    tmp52 = tl.where(tmp44, tmp46, tmp51)
    tmp53 = tl.where(tmp38, tmp40, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp33, tmp53, tmp54)
    tmp56 = tmp29 >= tmp32
    tmp57 = tl.full([1, 1], 8, tl.int64)
    tmp58 = tmp29 < tmp57
    tmp59 = tl.load(in_ptr6 + (r0_1 + 256*((-6) + x0)), tmp56 & xmask, other=0.0)
    tmp60 = tl.where(tmp33, tmp55, tmp59)
    tmp61 = tmp28 + tmp60
    tmp62 = tmp61.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp28, xmask)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp62, xmask)
