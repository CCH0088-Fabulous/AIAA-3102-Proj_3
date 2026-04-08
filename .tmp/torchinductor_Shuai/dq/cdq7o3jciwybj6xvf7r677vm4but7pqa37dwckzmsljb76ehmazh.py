
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_cat_cos_eq_mul_neg_sin_unsqueeze_where_zeros_like_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 10240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_cat_cos_eq_mul_neg_sin_unsqueeze_where_zeros_like_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 256
    x0 = (xindex % 256)
    x2 = xindex
    tmp20 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.where(tmp4, tmp6, 0)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.full([1], -1.0, tl.float32)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp10, tmp16)
    tmp18 = tl.full([1], -1.0, tl.float32)
    tmp19 = tmp17 == tmp18
    tmp21 = tl.full([1], 0.0, tl.float32)
    tmp22 = tmp21 + tmp20
    tmp23 = x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tl.full([1], 128, tl.int64)
    tmp26 = tmp23 < tmp25
    tmp27 = tl.load(in_ptr2 + (128*x1 + (x0)), tmp26 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.full([1], 6.283185307179586, tl.float32)
    tmp29 = tmp27 * tmp28
    tmp30 = tl_math.sin(tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp26, tmp30, tmp31)
    tmp33 = tmp23 >= tmp25
    tmp34 = tl.full([1], 256, tl.int64)
    tmp35 = tmp23 < tmp34
    tmp36 = tl.load(in_ptr2 + (128*x1 + ((-128) + x0)), tmp33 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tl.full([1], 6.283185307179586, tl.float32)
    tmp38 = tmp36 * tmp37
    tmp39 = tl_math.cos(tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp33, tmp39, tmp40)
    tmp42 = tl.where(tmp26, tmp32, tmp41)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tl.where(tmp19, tmp22, tmp43)
    tmp45 = tl.full([1], 2.0, tl.float32)
    tmp46 = tmp17 == tmp45
    tmp47 = tl.full([1], 1.0, tl.float32)
    tmp48 = tmp17 == tmp47
    tmp49 = tmp17 == tmp21
    tmp51 = tmp44 + tmp50
    tmp52 = tl.where(tmp49, tmp51, tmp44)
    tmp54 = tmp52 + tmp53
    tmp55 = tl.where(tmp48, tmp54, tmp52)
    tmp57 = tmp55 + tmp56
    tmp58 = tl.where(tmp46, tmp57, tmp55)
    tmp59 = tl.full([1], 3.0, tl.float32)
    tmp60 = tmp17 == tmp59
    tmp62 = tmp58 + tmp61
    tmp63 = tl.where(tmp60, tmp62, tmp58)
    tl.store(in_out_ptr0 + (x2), tmp63, xmask)
