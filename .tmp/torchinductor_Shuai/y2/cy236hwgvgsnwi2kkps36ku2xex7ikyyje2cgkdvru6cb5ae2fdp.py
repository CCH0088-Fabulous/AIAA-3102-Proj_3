
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_mean_pow_sub_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 5242880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_mean_pow_sub_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (262144 + x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (1))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp13 = tl.load(in_ptr0 + (524288 + x0), None).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (2))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (786432 + x0), None).to(tl.float32)
    tmp21 = tl.load(in_ptr1 + (3))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp5 + tmp11
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp13 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp12 + tmp18
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp19 + tmp25
    tmp27 = tl.full([1], 4.0, tl.float32)
    tmp28 = (tmp26 / tmp27)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp4 - tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31 * tmp31
    tmp33 = tmp10 - tmp29
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp34
    tmp36 = tmp32 + tmp35
    tmp37 = tmp17 - tmp29
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp38 * tmp38
    tmp40 = tmp36 + tmp39
    tmp41 = tmp24 - tmp29
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 * tmp42
    tmp44 = tmp40 + tmp43
    tmp45 = (tmp44 / tmp27)
    tl.store(out_ptr0 + (x0), tmp29, None)
    tl.store(out_ptr1 + (x0), tmp45, None)
