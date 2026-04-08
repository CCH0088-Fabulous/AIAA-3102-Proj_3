
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cat_copy_div_mul_select_sub_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 12}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_copy_div_mul_select_sub_zeros_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tl.load(in_ptr0 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.where(tmp9, tmp11, 0.0)
    tmp13 = tl.full([1], 0.5, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp9, tmp14, tmp15)
    tmp17 = tmp5 >= tmp8
    tmp18 = tl.full([1], 2, tl.int64)
    tmp19 = tmp5 < tmp18
    tmp20 = tl.full([1], 0.0, tl.float32)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp9, tmp16, tmp22)
    tmp24 = tl.full([1], 0.0009765625, tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.load(in_ptr0 + (1))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp28 = tl.where(tmp9, tmp27, 0.0)
    tmp29 = tmp28 + tmp13
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp9, tmp29, tmp30)
    tmp32 = tl.where(tmp9, tmp31, tmp22)
    tmp33 = tl.where(tmp4, tmp25, tmp32)
    tmp34 = tmp33 * tmp24
    tmp35 = tmp0 == tmp3
    tmp36 = tl.load(in_ptr0 + (x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp36 + tmp13
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp9, tmp37, tmp38)
    tmp40 = tl.where(tmp9, tmp39, tmp22)
    tmp41 = tl.where(tmp35, tmp25, tmp40)
    tmp42 = tl.where(tmp2, tmp34, tmp41)
    tmp43 = tl.full([1], 2.0, tl.float32)
    tmp44 = tmp42 * tmp43
    tmp45 = tl.full([1], 1.0, tl.float32)
    tmp46 = tmp44 - tmp45
    tmp47 = tmp46.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp47, xmask)
