
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_clamp_convolution_floor_mul_native_layer_norm_permute_repeat_rsub_sub_unsqueeze_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 2, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 95457280, 'r0_': 288}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_clamp_convolution_floor_mul_native_layer_norm_permute_repeat_rsub_sub_unsqueeze_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 256
    x0 = (xindex % 256)
    x3 = xindex
    tmp166_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp166_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp166_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp158 = tl.load(in_ptr1 + (x3 + 65536*r0_2), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp159 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp162 = tl.load(in_ptr3 + (8*((x1 % 8)) + 64*r0_2 + ((x0 % 8))), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp0 = x1
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.full([1, 1], 0.5, tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tl.full([1, 1], 0.02734375, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tmp5 - tmp2
        tmp7 = libdevice.floor(tmp6)
        tmp8 = tmp7.to(tl.int32)
        tmp9 = tl.full([1, 1], 1, tl.int64)
        tmp10 = tmp8 - tmp9
        tmp11 = tl.full([1, 1], 0, tl.int64)
        tmp12 = triton_helpers.maximum(tmp10, tmp11)
        tmp13 = tl.full([1, 1], 6, tl.int64)
        tmp14 = triton_helpers.minimum(tmp12, tmp13)
        tmp15 = x0
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16 + tmp2
        tmp18 = tmp17 * tmp4
        tmp19 = tmp18 - tmp2
        tmp20 = libdevice.floor(tmp19)
        tmp21 = tmp20.to(tl.int32)
        tmp22 = tmp21 - tmp9
        tmp23 = triton_helpers.maximum(tmp22, tmp11)
        tmp24 = triton_helpers.minimum(tmp23, tmp13)
        tmp25 = tl.load(in_ptr0 + (tmp24 + 7*tmp14 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp26 = tmp19 - tmp20
        tmp27 = tl.full([1, 1], 0.0, tl.float32)
        tmp28 = triton_helpers.maximum(tmp26, tmp27)
        tmp29 = tl.full([1, 1], 1.0, tl.float32)
        tmp30 = triton_helpers.minimum(tmp28, tmp29)
        tmp31 = tmp30 + tmp29
        tmp32 = tl.full([1, 1], -0.75, tl.float32)
        tmp33 = tmp31 * tmp32
        tmp34 = tl.full([1, 1], -3.75, tl.float32)
        tmp35 = tmp33 - tmp34
        tmp36 = tmp35 * tmp31
        tmp37 = tl.full([1, 1], -6.0, tl.float32)
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38 * tmp31
        tmp40 = tl.full([1, 1], -3.0, tl.float32)
        tmp41 = tmp39 - tmp40
        tmp42 = tmp25 * tmp41
        tmp43 = triton_helpers.maximum(tmp21, tmp11)
        tmp44 = triton_helpers.minimum(tmp43, tmp13)
        tmp45 = tl.load(in_ptr0 + (tmp44 + 7*tmp14 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.full([1, 1], 1.25, tl.float32)
        tmp47 = tmp30 * tmp46
        tmp48 = tl.full([1, 1], 2.25, tl.float32)
        tmp49 = tmp47 - tmp48
        tmp50 = tmp49 * tmp30
        tmp51 = tmp50 * tmp30
        tmp52 = tmp51 + tmp29
        tmp53 = tmp45 * tmp52
        tmp54 = tmp42 + tmp53
        tmp55 = tmp21 + tmp9
        tmp56 = triton_helpers.maximum(tmp55, tmp11)
        tmp57 = triton_helpers.minimum(tmp56, tmp13)
        tmp58 = tl.load(in_ptr0 + (tmp57 + 7*tmp14 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp59 = tmp29 - tmp30
        tmp60 = tmp59 * tmp46
        tmp61 = tmp60 - tmp48
        tmp62 = tmp61 * tmp59
        tmp63 = tmp62 * tmp59
        tmp64 = tmp63 + tmp29
        tmp65 = tmp58 * tmp64
        tmp66 = tmp54 + tmp65
        tmp67 = tl.full([1, 1], 2, tl.int64)
        tmp68 = tmp21 + tmp67
        tmp69 = triton_helpers.maximum(tmp68, tmp11)
        tmp70 = triton_helpers.minimum(tmp69, tmp13)
        tmp71 = tl.load(in_ptr0 + (tmp70 + 7*tmp14 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp72 = tl.full([1, 1], 2.0, tl.float32)
        tmp73 = tmp72 - tmp30
        tmp74 = tmp73 * tmp32
        tmp75 = tmp74 - tmp34
        tmp76 = tmp75 * tmp73
        tmp77 = tmp76 + tmp37
        tmp78 = tmp77 * tmp73
        tmp79 = tmp78 - tmp40
        tmp80 = tmp71 * tmp79
        tmp81 = tmp66 + tmp80
        tmp82 = triton_helpers.maximum(tmp8, tmp11)
        tmp83 = triton_helpers.minimum(tmp82, tmp13)
        tmp84 = tl.load(in_ptr0 + (tmp24 + 7*tmp83 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp85 = tmp84 * tmp41
        tmp86 = tl.load(in_ptr0 + (tmp44 + 7*tmp83 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp87 = tmp86 * tmp52
        tmp88 = tmp85 + tmp87
        tmp89 = tl.load(in_ptr0 + (tmp57 + 7*tmp83 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp90 = tmp89 * tmp64
        tmp91 = tmp88 + tmp90
        tmp92 = tl.load(in_ptr0 + (tmp70 + 7*tmp83 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp93 = tmp92 * tmp79
        tmp94 = tmp91 + tmp93
        tmp95 = tmp6 - tmp7
        tmp96 = triton_helpers.maximum(tmp95, tmp27)
        tmp97 = triton_helpers.minimum(tmp96, tmp29)
        tmp98 = tmp97 + tmp29
        tmp99 = tmp98 * tmp32
        tmp100 = tmp99 - tmp34
        tmp101 = tmp100 * tmp98
        tmp102 = tmp101 + tmp37
        tmp103 = tmp102 * tmp98
        tmp104 = tmp103 - tmp40
        tmp105 = tmp81 * tmp104
        tmp106 = tmp97 * tmp46
        tmp107 = tmp106 - tmp48
        tmp108 = tmp107 * tmp97
        tmp109 = tmp108 * tmp97
        tmp110 = tmp109 + tmp29
        tmp111 = tmp94 * tmp110
        tmp112 = tmp105 + tmp111
        tmp113 = tmp8 + tmp9
        tmp114 = triton_helpers.maximum(tmp113, tmp11)
        tmp115 = triton_helpers.minimum(tmp114, tmp13)
        tmp116 = tl.load(in_ptr0 + (tmp24 + 7*tmp115 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp117 = tmp116 * tmp41
        tmp118 = tl.load(in_ptr0 + (tmp44 + 7*tmp115 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp119 = tmp118 * tmp52
        tmp120 = tmp117 + tmp119
        tmp121 = tl.load(in_ptr0 + (tmp57 + 7*tmp115 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp122 = tmp121 * tmp64
        tmp123 = tmp120 + tmp122
        tmp124 = tl.load(in_ptr0 + (tmp70 + 7*tmp115 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp125 = tmp124 * tmp79
        tmp126 = tmp123 + tmp125
        tmp127 = tmp8 + tmp67
        tmp128 = triton_helpers.maximum(tmp127, tmp11)
        tmp129 = triton_helpers.minimum(tmp128, tmp13)
        tmp130 = tl.load(in_ptr0 + (tmp24 + 7*tmp129 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp131 = tmp130 * tmp41
        tmp132 = tl.load(in_ptr0 + (tmp44 + 7*tmp129 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp133 = tmp132 * tmp52
        tmp134 = tmp131 + tmp133
        tmp135 = tl.load(in_ptr0 + (tmp57 + 7*tmp129 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp136 = tmp135 * tmp64
        tmp137 = tmp134 + tmp136
        tmp138 = tl.load(in_ptr0 + (tmp70 + 7*tmp129 + 49*r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp139 = tmp138 * tmp79
        tmp140 = tmp137 + tmp139
        tmp141 = tmp29 - tmp97
        tmp142 = tmp141 * tmp46
        tmp143 = tmp142 - tmp48
        tmp144 = tmp143 * tmp141
        tmp145 = tmp144 * tmp141
        tmp146 = tmp145 + tmp29
        tmp147 = tmp126 * tmp146
        tmp148 = tmp112 + tmp147
        tmp149 = tmp72 - tmp97
        tmp150 = tmp149 * tmp32
        tmp151 = tmp150 - tmp34
        tmp152 = tmp151 * tmp149
        tmp153 = tmp152 + tmp37
        tmp154 = tmp153 * tmp149
        tmp155 = tmp154 - tmp40
        tmp156 = tmp140 * tmp155
        tmp157 = tmp148 + tmp156
        tmp160 = tmp158 + tmp159
        tmp161 = tmp160.to(tl.float32)
        tmp163 = tmp157 + tmp162
        tmp164 = tmp161 + tmp163
        tmp165 = tl.broadcast_to(tmp164, [XBLOCK, R0_BLOCK])
        tmp166_mean_next, tmp166_m2_next, tmp166_weight_next = triton_helpers.welford_reduce(
            tmp165, tmp166_mean, tmp166_m2, tmp166_weight, roffset == 0
        )
        tmp166_mean = tl.where(r0_mask, tmp166_mean_next, tmp166_mean)
        tmp166_m2 = tl.where(r0_mask, tmp166_m2_next, tmp166_m2)
        tmp166_weight = tl.where(r0_mask, tmp166_weight_next, tmp166_weight)
        tl.store(in_out_ptr0 + (x3 + 65536*r0_2), tmp157, r0_mask)
    tmp167, tmp168, tmp169 = triton_helpers.welford(tmp166_mean, tmp166_m2, tmp166_weight, 1)
    tmp166 = tmp167[:, None]
    tmp170 = tmp168[:, None]
    tmp171 = tmp169[:, None]
    tl.store(out_ptr0 + (x3), tmp166, None)
    tl.store(out_ptr1 + (x3), tmp170, None)
