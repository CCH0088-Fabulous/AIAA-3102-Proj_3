# AOT ID: ['5_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch._C import _cuda_getCurrentRawStream as get_raw_stream



# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/3f/c3fk4zjinru7lxyqmxyhgpjh5zba5dwqrpqw24f46bsxdmzojy4c.py
# Topologically Sorted Source Nodes: [mul, output, output_1, tgt2], Original ATen: [aten.mul, aten.add, aten.transpose, aten.native_layer_norm]
# Source node to ATen node mapping:
#   mul => mul
#   output => add_4
#   output_1 => permute
#   tgt2 => var_mean
# Graph fragment:
#   %arg1_1 : Tensor "f32[s14, 1, 256][256, 256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg4_1 : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0" = PlaceHolder[target=arg4_1]
#   %mul : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 0.1), kwargs = {})
#   %add_4 : Tensor "f32[s14, 1, 256][256, 256*s14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %mul), kwargs = {})
#   %permute : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [1, 0, 2]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute, [2]), kwargs = {correction: 0, keepdim: True})
#   return %buf0,%buf1,%buf2
triton_red_fused_add_mul_native_layer_norm_transpose_0 = async_compile.triton('triton_red_fused_add_mul_native_layer_norm_transpose_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'y': 4096, 'x': 2, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i64', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_transpose_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 3, 'num_reduction': 3, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4194304, 'x': 49152, 'r0_': 4194304}}
)
@triton.jit
def triton_red_fused_add_mul_native_layer_norm_transpose_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ks0, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
    rbase = r0_base
    x1 = xindex
    y0 = yindex
    tmp6_mean = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x1 + 256*y0), r0_mask & xmask & ymask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (y0 + ks0*r0_2 + 128*ks0*x1), r0_mask & xmask & ymask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([1, 1, 1], 0.1, tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask & ymask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask & ymask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask & ymask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 2)
    tmp6 = tmp7[:, :, None]
    tmp10 = tmp8[:, :, None]
    tmp11 = tmp9[:, :, None]
    tl.store(out_ptr0 + (x1 + 2*y0), tmp6, xmask & ymask)
    tl.store(out_ptr1 + (x1 + 2*y0), tmp10, xmask & ymask)
    tl.store(out_ptr2 + (x1 + 2*y0), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/4q/c4qr3gqnsl66sjmikfwbjued5ink2cvuubka44dxtdzuhqfkhhx2.py
# Topologically Sorted Source Nodes: [mul, output, output_1, tgt2], Original ATen: [aten.mul, aten.add, aten.transpose, aten.native_layer_norm]
# Source node to ATen node mapping:
#   mul => mul
#   output => add_4
#   output_1 => permute
#   tgt2 => var_mean
# Graph fragment:
#   %buf0 : Tensor "f32[1, s14, 1, 2][2*s14, 2, 2*s14, 1]cuda:0" = PlaceHolder[target=buf0]
#   %buf1 : Tensor "f32[1, s14, 1, 2][2*s14, 2, 2*s14, 1]cuda:0" = PlaceHolder[target=buf1]
#   %buf2 : Tensor "f32[1, s14, 1, 2][2*s14, 2, 2*s14, 1]cuda:0" = PlaceHolder[target=buf2]
#   %mul : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 0.1), kwargs = {})
#   %add_4 : Tensor "f32[s14, 1, 256][256, 256*s14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %mul), kwargs = {})
#   %permute : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [1, 0, 2]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute, [2]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf4
triton_per_fused_add_mul_native_layer_norm_transpose_1 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_transpose_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_transpose_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 65536, 'r0_': 24576}}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_transpose_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 2*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 2*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 2*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/qj/cqjqg6oh26no3qrafnirmn4lxhvrlmz55yy7o6ya2rja2b2y76hs.py
# Topologically Sorted Source Nodes: [mul, output, output_1, tgt2, q, k, v], Original ATen: [aten.mul, aten.add, aten.transpose, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   k => convert_element_type_8
#   mul => mul
#   output => add_4
#   output_1 => permute
#   q => convert_element_type_2
#   tgt2 => add_21, add_22, mul_19, mul_20, rsqrt, sub_6, var_mean
#   v => convert_element_type_14
# Graph fragment:
#   %arg1_1 : Tensor "f32[s14, 1, 256][256, 256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg4_1 : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0" = PlaceHolder[target=arg4_1]
#   %getitem_1 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf4 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf4]
#   %arg6_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %add_22 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_22]
#   %mul : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 0.1), kwargs = {})
#   %add_4 : Tensor "f32[s14, 1, 256][256, 256*s14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %mul), kwargs = {})
#   %permute : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [1, 0, 2]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute, %getitem_1), kwargs = {})
#   %add_21 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %mul_19 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt), kwargs = {})
#   %mul_20 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %arg6_1), kwargs = {})
#   %add_22 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %arg7_1), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.bfloat16), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.bfloat16), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.bfloat16), kwargs = {})
#   return %add_22,%convert_element_type_2,%convert_element_type_8,%convert_element_type_14
triton_poi_fused__to_copy_add_mul_native_layer_norm_transpose_2 = async_compile.triton('triton_poi_fused__to_copy_add_mul_native_layer_norm_transpose_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ks0': 'i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_native_layer_norm_transpose_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 3, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4227072, 'x': 16779264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_native_layer_norm_transpose_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, out_ptr3, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 256
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + ks0*x1), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1, 1], 0.1, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tl.full([1, 1], 256.0, tl.float32)
    tmp9 = (tmp7 / tmp8)
    tmp10 = tl.full([1, 1], 1e-05, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr1 + (x1 + 256*y0), tmp18, xmask & ymask)
    tl.store(out_ptr2 + (x1 + 256*y0), tmp18, xmask & ymask)
    tl.store(out_ptr3 + (x1 + 256*y0), tmp18, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/s5/cs5qi43oc4qorfrbugptjdbxjfeqdoviqvaeskyfjq4qiql5zopm.py
# Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   q => convert_element_type_1
# Graph fragment:
#   %arg8_1 : Tensor "f32[256, 256][256, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convert_element_type_1 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/xz/cxzdtrhp3rhwfge5jwj43qrzdvt7tlme3cqsrzeqob4xaxiyi6he.py
# Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   q => convert_element_type
# Graph fragment:
#   %arg9_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %convert_element_type : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/ax/caxnfqjebcm6qadde7vvcnbfl4jxxaoh7srvh62m4gyq2mnuw4kr.py
# Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   q => addmm, convert_element_type, convert_element_type_1, convert_element_type_2, permute_4, view
# Graph fragment:
#   %convert_element_type : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type]
#   %convert_element_type_2 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=convert_element_type_2]
#   %convert_element_type_1 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_1]
#   %convert_element_type : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.bfloat16), kwargs = {})
#   %view : Tensor "bf16[s14, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_2, [%arg0_1, 256]), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.bfloat16), kwargs = {})
#   %permute_4 : Tensor "bf16[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1, [1, 0]), kwargs = {})
#   %addmm : Tensor "bf16[s14, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type, %view, %permute_4), kwargs = {})
#   return %addmm
triton_tem_fused__to_copy_addmm_t_view_5 = async_compile.triton('triton_tem_fused__to_copy_addmm_t_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=8,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_t_view_5', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_t_view_5(in_ptr0, arg_A, arg_B, out_ptr0, ks0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = ks0
    N = 256
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/ew/cew5irqpg6mqzdf7z2ibofabr2asy6hojbvd2cpda2ep3x6n7fis.py
# Topologically Sorted Source Nodes: [q, x, q_1, float_1, reshape_3, xq_], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_18
#   q => view_1
#   q_1 => permute_7
#   reshape_3 => view_9
#   x => view_6
#   xq_ => view_as_complex
# Graph fragment:
#   %addmm : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_1 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, %arg0_1, 256]), kwargs = {})
#   %view_6 : Tensor "bf16[1, s14, 1, 256][256*s14, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [1, %arg0_1, 1, 256]), kwargs = {})
#   %permute_7 : Tensor "bf16[1, 1, s14, 256][256*s14, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %convert_element_type_18 : Tensor "f32[1, 1, s14, 256][256*s14, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_7, torch.float32), kwargs = {})
#   %view_9 : Tensor "f32[1, 1, s14, 128, 2][256*s14, 256, 256, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_18, [1, 1, %arg0_1, -1, 2]), kwargs = {})
#   %view_as_complex : Tensor "c64[1, 1, s14, 128][128*s14, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.view_as_complex.default](args = (%view_9,), kwargs = {})
#   return %buf11
triton_poi_fused__to_copy_transpose_view_view_as_complex_6 = async_compile.triton('triton_poi_fused__to_copy_transpose_view_view_as_complex_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_transpose_view_view_as_complex_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 10485760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_transpose_view_view_as_complex_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/m5/cm56hsyv3gykh7mbtxjj4qcg6x6hmtqqrh7jav6672yopnzuq6te.py
# Topologically Sorted Source Nodes: [xq_out, type_as, k, x_1, k_1, xk_out, type_as_1, setitem, v, x_2, v_1, out], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   k => view_3
#   k_1 => permute_8
#   out => _scaled_dot_product_flash_attention
#   setitem => copy, permute_10, permute_11, view_16, view_19
#   type_as => convert_element_type_20
#   type_as_1 => convert_element_type_21
#   v => view_5
#   v_1 => permute_9
#   x_1 => view_7
#   x_2 => view_8
#   xk_out => view_13
#   xq_out => view_12
# Graph fragment:
#   %view_as_real : Tensor "f32[1, 1, s14, 128, 2][256*s14, 256*s14, 256, 2, 1]cuda:0" = PlaceHolder[target=view_as_real]
#   %view_12 : Tensor "f32[1, 1, s14, 256][256*s14, 256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_as_real, [1, 1, %arg0_1, 256]), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[1, 1, s14, 256][256*s14, 256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_12, torch.bfloat16), kwargs = {})
#   %view_3 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, %arg0_1, 256]), kwargs = {})
#   %view_7 : Tensor "bf16[1, s14, 1, 256][256*s14, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [1, %arg0_1, 1, 256]), kwargs = {})
#   %permute_8 : Tensor "bf16[1, 1, s14, 256][256*s14, 256, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [0, 2, 1, 3]), kwargs = {})
#   %view_13 : Tensor "f32[1, 1, s14, 256][256*s14, 256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_as_real_1, [1, 1, %arg0_1, 256]), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[1, 1, s14, 256][256*s14, 256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_13, torch.bfloat16), kwargs = {})
#   %copy : Tensor "bf16[1, 1, s14, 256][256*s14, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%permute_8, %convert_element_type_21), kwargs = {})
#   %permute_10 : Tensor "bf16[1, s14, 1, 256][256*s14, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%copy, [0, 2, 1, 3]), kwargs = {})
#   %view_16 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_10, [1, %arg0_1, 256]), kwargs = {})
#   %view_19 : Tensor "bf16[1, s14, 1, 256][256*s14, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_16, [1, %arg0_1, 1, 256]), kwargs = {})
#   %permute_11 : Tensor "bf16[1, 1, s14, 256][256*s14, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_19, [0, 2, 1, 3]), kwargs = {})
#   %view_5 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, %arg0_1, 256]), kwargs = {})
#   %view_8 : Tensor "bf16[1, s14, 1, 256][256*s14, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, %arg0_1, 1, 256]), kwargs = {})
#   %permute_9 : Tensor "bf16[1, 1, s14, 256][256*s14, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%convert_element_type_20, %permute_11, %permute_9), kwargs = {scale: 0.0625})
#   return %buf35
triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/pg/cpgmik4z6am6hy4trrxwdhplrfngxpglxkquf5immsnb5t5njhpy.py
# Topologically Sorted Source Nodes: [mul, output, output_1, out_2, tgt, tgt2_1, q_3], Original ATen: [aten.mul, aten.add, aten.transpose, aten.view, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   mul => mul
#   out_2 => view_22
#   output => add_4
#   output_1 => permute
#   q_3 => convert_element_type_29
#   tgt => add_174
#   tgt2_1 => add_178, add_179, mul_215, mul_216, rsqrt_1, sub_56, var_mean_1
# Graph fragment:
#   %arg1_1 : Tensor "f32[s14, 1, 256][256, 256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg4_1 : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0" = PlaceHolder[target=arg4_1]
#   %addmm_3 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %getitem_12 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_12]
#   %buf47 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf47]
#   %arg18_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %mul : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 0.1), kwargs = {})
#   %add_4 : Tensor "f32[s14, 1, 256][256, 256*s14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %mul), kwargs = {})
#   %permute : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [1, 0, 2]), kwargs = {})
#   %view_22 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, %arg0_1, 256]), kwargs = {})
#   %add_174 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %view_22), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_174, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_56 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_174, %getitem_12), kwargs = {})
#   %add_178 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_178,), kwargs = {})
#   %mul_215 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %rsqrt_1), kwargs = {})
#   %mul_216 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_215, %arg18_1), kwargs = {})
#   %add_179 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_216, %arg19_1), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_179, torch.bfloat16), kwargs = {})
#   return %getitem_12,%buf47,%convert_element_type_29
triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_8 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4194304, 'r0_': 10487808}}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + ks0*r0_1), xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tl.full([1, 1], 0.1, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp15 = tl.full([1, 1], 256, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp24 = tmp7 - tmp17
    tmp25 = tl.full([1, 1], 256.0, tl.float32)
    tmp26 = (tmp23 / tmp25)
    tmp27 = tl.full([1, 1], 1e-05, tl.float32)
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp35, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/pn/cpnd73vr7d4nzollrczi6it73a56agoikw3qe4vs4escpvelfif4.py
# Topologically Sorted Source Nodes: [memory, memory_pos, add_2, k_2, v_2], Original ATen: [aten.transpose, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   add_2 => add_189
#   k_2 => convert_element_type_35
#   memory => permute_2
#   memory_pos => permute_3
#   v_2 => convert_element_type_41
# Graph fragment:
#   %arg3_1 : Tensor "f32[s98, 1, 64][64, 64, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg5_1 : Tensor "f32[s98, 1, 64][64, 64, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %permute_2 : Tensor "f32[1, s98, 64][64, 64, 1]cuda:0"[num_users=8] = call_function[target=torch.ops.aten.permute.default](args = (%arg3_1, [1, 0, 2]), kwargs = {})
#   %permute_3 : Tensor "f32[1, s98, 64][64, 64, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%arg5_1, [1, 0, 2]), kwargs = {})
#   %add_189 : Tensor "f32[1, s98, 64][64*s98, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_2, %permute_3), kwargs = {})
#   %convert_element_type_35 : Tensor "bf16[1, s98, 64][64*s98, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_189, torch.bfloat16), kwargs = {})
#   %convert_element_type_41 : Tensor "bf16[1, s98, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_2, torch.bfloat16), kwargs = {})
#   return %convert_element_type_35,%convert_element_type_41
triton_poi_fused__to_copy_add_transpose_9 = async_compile.triton('triton_poi_fused__to_copy_add_transpose_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_transpose_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4198400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_transpose_9(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/ij/cijwe6uuucvcurlobjwv4zfzauizax4dec474dwzho6dckubs3sf.py
# Topologically Sorted Source Nodes: [k_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   k_2 => convert_element_type_34
# Graph fragment:
#   %arg22_1 : Tensor "f32[256, 64][64, 1]cuda:0" = PlaceHolder[target=arg22_1]
#   %convert_element_type_34 : Tensor "bf16[256, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg22_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_34
triton_poi_fused__to_copy_10 = async_compile.triton('triton_poi_fused__to_copy_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/ac/cactm3z57x7iwcqhlofok3nwsrbvnbu6kbf4vlfivr3k5lkesp5e.py
# Topologically Sorted Source Nodes: [k_2, memory, memory_pos, add_2], Original ATen: [aten._to_copy, aten.transpose, aten.add, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   add_2 => add_189
#   k_2 => addmm_5, convert_element_type_33, convert_element_type_34, convert_element_type_35, permute_15, view_25
#   memory => permute_2
#   memory_pos => permute_3
# Graph fragment:
#   %convert_element_type_33 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type_33]
#   %convert_element_type_35 : Tensor "bf16[1, s98, 64][64*s98, 64, 1]cuda:0" = PlaceHolder[target=convert_element_type_35]
#   %convert_element_type_34 : Tensor "bf16[256, 64][64, 1]cuda:0" = PlaceHolder[target=convert_element_type_34]
#   %convert_element_type_33 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg23_1, torch.bfloat16), kwargs = {})
#   %permute_2 : Tensor "f32[1, s98, 64][64, 64, 1]cuda:0"[num_users=8] = call_function[target=torch.ops.aten.permute.default](args = (%arg3_1, [1, 0, 2]), kwargs = {})
#   %permute_3 : Tensor "f32[1, s98, 64][64, 64, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%arg5_1, [1, 0, 2]), kwargs = {})
#   %add_189 : Tensor "f32[1, s98, 64][64*s98, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_2, %permute_3), kwargs = {})
#   %convert_element_type_35 : Tensor "bf16[1, s98, 64][64*s98, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_189, torch.bfloat16), kwargs = {})
#   %view_25 : Tensor "bf16[s98, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_35, [%arg2_1, 64]), kwargs = {})
#   %convert_element_type_34 : Tensor "bf16[256, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg22_1, torch.bfloat16), kwargs = {})
#   %permute_15 : Tensor "bf16[64, 256][1, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_34, [1, 0]), kwargs = {})
#   %addmm_5 : Tensor "bf16[s98, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_33, %view_25, %permute_15), kwargs = {})
#   return %addmm_5
triton_tem_fused__to_copy_add_addmm_t_transpose_view_11 = async_compile.triton('triton_tem_fused__to_copy_add_addmm_t_transpose_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=8,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_addmm_t_transpose_view_11', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_add_addmm_t_transpose_view_11(in_ptr0, arg_A, arg_B, out_ptr0, ks0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = ks0
    N = 256
    K = 64
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 64
    stride_ak = 1
    stride_bk = 1
    stride_bn = 64

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 64*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 64*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/na/cnaycw6begbrnuxymicjlsju37ugsy7rhbputkfted3xubca5t5m.py
# Topologically Sorted Source Nodes: [xq_out_1, type_as_2, setitem_1, k_2, x_5, k_3, xk_out_1, type_as_3, out_3, v_2, x_6, v_3], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   k_2 => view_26
#   k_3 => permute_18
#   out_3 => _scaled_dot_product_flash_attention_1, permute_23, view_44
#   setitem_1 => copy_1, permute_20, permute_21, slice_2, view_37, view_38, view_39
#   type_as_2 => convert_element_type_47
#   type_as_3 => convert_element_type_48
#   v_2 => view_28
#   v_3 => permute_19
#   x_5 => view_30
#   x_6 => view_31
#   xk_out_1 => view_36
#   xq_out_1 => view_35
# Graph fragment:
#   %view_as_real_3 : Tensor "f32[1, 1, -s46 + s98, 128, 2][256*Max(1, -s46 + s98), 256*Max(1, -s46 + s98), 256, 2, 1]cuda:0" = PlaceHolder[target=view_as_real_3]
#   %addmm_5 : Tensor "bf16[s98, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %view_35 : Tensor "f32[1, 1, s14, 256][256*s14, 256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_as_real_2, [1, 1, %arg0_1, 256]), kwargs = {})
#   %convert_element_type_47 : Tensor "bf16[1, 1, s14, 256][256*s14, 256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_35, torch.bfloat16), kwargs = {})
#   %view_37 : Tensor "bf16[1, s98, 256][256*s98, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, %arg2_1, 256]), kwargs = {})
#   %view_38 : Tensor "bf16[1, s98, 1, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_37, [1, %arg2_1, 1, 256]), kwargs = {})
#   %permute_20 : Tensor "bf16[1, 1, s98, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_38, [0, 2, 1, 3]), kwargs = {})
#   %view_26 : Tensor "bf16[1, s98, 256][256*s98, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, %arg2_1, 256]), kwargs = {})
#   %view_30 : Tensor "bf16[1, s98, 1, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_26, [1, %arg2_1, 1, 256]), kwargs = {})
#   %permute_18 : Tensor "bf16[1, 1, s98, 256][256*s98, 256, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_30, [0, 2, 1, 3]), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 1, -s46 + s98, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_18, 2, 0, %sub_79), kwargs = {})
#   %view_36 : Tensor "f32[1, 1, -s46 + s98, 256][256*Max(1, -s46 + s98), 256*Max(1, -s46 + s98), 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_as_real_3, [1, 1, %sub_79, 256]), kwargs = {})
#   %convert_element_type_48 : Tensor "bf16[1, 1, -s46 + s98, 256][256*Max(1, -s46 + s98), 256*Max(1, -s46 + s98), 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_36, torch.bfloat16), kwargs = {})
#   %copy_1 : Tensor "bf16[1, 1, -s46 + s98, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_2, %convert_element_type_48), kwargs = {})
#   %slice_scatter_default : Tensor "bf16[1, 1, s98, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_20, %copy_1, 2, 0, %sub_79), kwargs = {})
#   %permute_21 : Tensor "bf16[1, s98, 1, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_scatter_default, [0, 2, 1, 3]), kwargs = {})
#   %view_39 : Tensor "bf16[1, s98, 256][256*s98, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_21, [1, %arg2_1, 256]), kwargs = {})
#   %view_44 : Tensor "bf16[1, s98, 1, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_39, [1, %arg2_1, 1, 256]), kwargs = {})
#   %permute_23 : Tensor "bf16[1, 1, s98, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_44, [0, 2, 1, 3]), kwargs = {})
#   %view_28 : Tensor "bf16[1, s98, 256][256*s98, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_6, [1, %arg2_1, 256]), kwargs = {})
#   %view_31 : Tensor "bf16[1, s98, 1, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_28, [1, %arg2_1, 1, 256]), kwargs = {})
#   %permute_19 : Tensor "bf16[1, 1, s98, 256][256*s98, 256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_31, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%convert_element_type_47, %permute_23, %permute_19), kwargs = {scale: 0.0625})
#   return %buf80
triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 10491904}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12(in_out_ptr0, in_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 256
    x2 = xindex
    tmp7 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp0 = x1
    tmp1 = ks1 + ((-1)*ks0)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2 & xmask, other=0.0)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tmp8 = tl.where(tmp2, tmp6, tmp7)
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/36/c3627j2wgiwvvadysboc6whxkbdgmxwkxajokjxclrq6hbtlpjig.py
# Topologically Sorted Source Nodes: [mul, output, output_1, out_2, tgt, out_5, tgt_1, tgt2_2, linear_8], Original ATen: [aten.mul, aten.add, aten.transpose, aten.view, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   linear_8 => convert_element_type_56
#   mul => mul
#   out_2 => view_22
#   out_5 => view_47
#   output => add_4
#   output_1 => permute
#   tgt => add_174
#   tgt2_2 => add_345, add_346, mul_420, mul_421, rsqrt_2, sub_110, var_mean_2
#   tgt_1 => add_341
# Graph fragment:
#   %arg1_1 : Tensor "f32[s14, 1, 256][256, 256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg4_1 : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0" = PlaceHolder[target=arg4_1]
#   %addmm_3 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %addmm_7 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_7]
#   %getitem_23 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_23]
#   %buf91 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf91]
#   %arg29_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %arg30_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %mul : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 0.1), kwargs = {})
#   %add_4 : Tensor "f32[s14, 1, 256][256, 256*s14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %mul), kwargs = {})
#   %permute : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [1, 0, 2]), kwargs = {})
#   %view_22 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, %arg0_1, 256]), kwargs = {})
#   %add_174 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %view_22), kwargs = {})
#   %view_47 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_7, [1, %arg0_1, 256]), kwargs = {})
#   %add_341 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_174, %view_47), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_341, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_110 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_341, %getitem_23), kwargs = {})
#   %add_345 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_345,), kwargs = {})
#   %mul_420 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %rsqrt_2), kwargs = {})
#   %mul_421 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_420, %arg29_1), kwargs = {})
#   %add_346 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_421, %arg30_1), kwargs = {})
#   %convert_element_type_56 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_346, torch.bfloat16), kwargs = {})
#   return %getitem_23,%buf91,%convert_element_type_56
triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_13 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4194304, 'r0_': 12584960}}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + ks0*r0_1), xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tl.full([1, 1], 0.1, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tl.full([1, 1], 256, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = (tmp17 / tmp19)
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp25 = tl.where(xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp27 = tmp10 - tmp20
    tmp28 = tl.full([1, 1], 256.0, tl.float32)
    tmp29 = (tmp26 / tmp28)
    tmp30 = tl.full([1, 1], 1e-05, tl.float32)
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp38, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/wz/cwzarrddmy6zr6q4xt2mmih52vgsythfmic4bupua5zcz6v6o7cn.py
# Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_8 => convert_element_type_55
# Graph fragment:
#   %arg31_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=arg31_1]
#   %convert_element_type_55 : Tensor "bf16[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg31_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_55
triton_poi_fused__to_copy_14 = async_compile.triton('triton_poi_fused__to_copy_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/7q/c7qzb4yw5dl6wdkm72wxbdjahgl4dtw2kexkq4vdto4qb3lk3ou4.py
# Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_8 => convert_element_type_54
# Graph fragment:
#   %arg32_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %convert_element_type_54 : Tensor "bf16[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg32_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_54
triton_poi_fused__to_copy_15 = async_compile.triton('triton_poi_fused__to_copy_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/4q/c4qoi2gtccsoargipv675ee4kefzbkj5bhe4yicgrcup4jq2jqeg.py
# Topologically Sorted Source Nodes: [mul, output, output_1, out_2, tgt, out_5, tgt_1, tgt2_2, linear_8], Original ATen: [aten.mul, aten.add, aten.transpose, aten.view, aten.native_layer_norm, aten._to_copy, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   linear_8 => addmm_8, convert_element_type_54, convert_element_type_55, convert_element_type_56, permute_26, view_48
#   mul => mul
#   out_2 => view_22
#   out_5 => view_47
#   output => add_4
#   output_1 => permute
#   tgt => add_174
#   tgt2_2 => add_345, add_346, mul_420, mul_421, rsqrt_2, sub_110, var_mean_2
#   tgt_1 => add_341
# Graph fragment:
#   %convert_element_type_54 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=convert_element_type_54]
#   %convert_element_type_56 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=convert_element_type_56]
#   %convert_element_type_55 : Tensor "bf16[2048, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_55]
#   %mul : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 0.1), kwargs = {})
#   %add_4 : Tensor "f32[s14, 1, 256][256, 256*s14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %mul), kwargs = {})
#   %permute : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [1, 0, 2]), kwargs = {})
#   %view_22 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, %arg0_1, 256]), kwargs = {})
#   %add_174 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %view_22), kwargs = {})
#   %view_47 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_7, [1, %arg0_1, 256]), kwargs = {})
#   %add_341 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_174, %view_47), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_341, [2]), kwargs = {correction: 0, keepdim: True})
#   %convert_element_type_54 : Tensor "bf16[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg32_1, torch.bfloat16), kwargs = {})
#   %sub_110 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_341, %getitem_23), kwargs = {})
#   %add_345 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_345,), kwargs = {})
#   %mul_420 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %rsqrt_2), kwargs = {})
#   %mul_421 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_420, %arg29_1), kwargs = {})
#   %add_346 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_421, %arg30_1), kwargs = {})
#   %convert_element_type_56 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_346, torch.bfloat16), kwargs = {})
#   %view_48 : Tensor "bf16[s14, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_56, [%arg0_1, 256]), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg31_1, torch.bfloat16), kwargs = {})
#   %permute_26 : Tensor "bf16[256, 2048][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_55, [1, 0]), kwargs = {})
#   %addmm_8 : Tensor "bf16[s14, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_54, %view_48, %permute_26), kwargs = {})
#   return %addmm_8
triton_tem_fused__to_copy_add_addmm_mul_native_layer_norm_t_transpose_view_16 = async_compile.triton('triton_tem_fused__to_copy_add_addmm_mul_native_layer_norm_t_transpose_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_addmm_mul_native_layer_norm_t_transpose_view_16', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_add_addmm_mul_native_layer_norm_t_transpose_view_16(in_ptr0, arg_A, arg_B, out_ptr0, ks0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = ks0
    N = 2048
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 2048*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 2048*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/n5/cn5wkgp57qzldk6dbjw7yynggi6wenvloybegwco6ablra63dvmp.py
# Topologically Sorted Source Nodes: [linear_8, relu], Original ATen: [aten.view, aten.relu]
# Source node to ATen node mapping:
#   linear_8 => view_49
#   relu => relu
# Graph fragment:
#   %addmm_8 : Tensor "bf16[s14, 2048][2048, 1]cuda:0" = PlaceHolder[target=addmm_8]
#   %view_49 : Tensor "bf16[1, s14, 2048][2048*s14, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_8, [1, %arg0_1, 2048]), kwargs = {})
#   %relu : Tensor "bf16[1, s14, 2048][2048*s14, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_49,), kwargs = {})
#   return %relu
triton_poi_fused_relu_view_17 = async_compile.triton('triton_poi_fused_relu_view_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_view_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_view_17(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/xo/cxox4upz3axdl6qn2kn2twciz26qcqa2kfhfphd5s5zgh4grhzio.py
# Topologically Sorted Source Nodes: [tgt2_3, linear_8, relu], Original ATen: [aten._to_copy, aten.view, aten.relu, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   linear_8 => view_49
#   relu => relu
#   tgt2_3 => addmm_9, convert_element_type_60, convert_element_type_61, permute_27, view_50
# Graph fragment:
#   %convert_element_type_60 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type_60]
#   %relu : Tensor "bf16[1, s14, 2048][2048*s14, 2048, 1]cuda:0" = PlaceHolder[target=relu]
#   %convert_element_type_61 : Tensor "bf16[256, 2048][2048, 1]cuda:0" = PlaceHolder[target=convert_element_type_61]
#   %convert_element_type_60 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg34_1, torch.bfloat16), kwargs = {})
#   %view_49 : Tensor "bf16[1, s14, 2048][2048*s14, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_8, [1, %arg0_1, 2048]), kwargs = {})
#   %relu : Tensor "bf16[1, s14, 2048][2048*s14, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_49,), kwargs = {})
#   %view_50 : Tensor "bf16[s14, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%relu, [%arg0_1, 2048]), kwargs = {})
#   %convert_element_type_61 : Tensor "bf16[256, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg33_1, torch.bfloat16), kwargs = {})
#   %permute_27 : Tensor "bf16[2048, 256][1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_61, [1, 0]), kwargs = {})
#   %addmm_9 : Tensor "bf16[s14, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_60, %view_50, %permute_27), kwargs = {})
#   return %addmm_9
triton_tem_fused__to_copy_addmm_relu_t_view_18 = async_compile.triton('triton_tem_fused__to_copy_addmm_relu_t_view_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_relu_t_view_18', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_relu_t_view_18(in_ptr0, arg_A, arg_B, out_ptr0, ks0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = ks0
    N = 256
    K = 2048
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2048
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2048

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 2048*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 2048*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 256*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/c7/cc7b6y7gziiynpyohxtnn32pdpdabqsclnykowzg7a5nnrmff6sg.py
# Topologically Sorted Source Nodes: [mul, output, output_1, out_2, tgt, out_5, tgt_1, tgt2_3, tgt_2, tgt2_4, q_6, k_4, v_4], Original ATen: [aten.mul, aten.add, aten.transpose, aten.view, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   k_4 => convert_element_type_73
#   mul => mul
#   out_2 => view_22
#   out_5 => view_47
#   output => add_4
#   output_1 => permute
#   q_6 => convert_element_type_67
#   tgt => add_174
#   tgt2_3 => view_51
#   tgt2_4 => add_390, add_391, mul_466, mul_467, rsqrt_3, sub_125, var_mean_3
#   tgt_1 => add_341
#   tgt_2 => add_386
#   v_4 => convert_element_type_79
# Graph fragment:
#   %arg1_1 : Tensor "f32[s14, 1, 256][256, 256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg4_1 : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0" = PlaceHolder[target=arg4_1]
#   %addmm_3 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %addmm_7 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_7]
#   %addmm_9 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %add_386 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_386]
#   %getitem_25 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_25]
#   %buf103 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf103]
#   %arg35_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %arg36_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg36_1]
#   %add_391 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_391]
#   %mul : Tensor "f32[s14, 1, 256][1, 256*s14, s14]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, 0.1), kwargs = {})
#   %add_4 : Tensor "f32[s14, 1, 256][256, 256*s14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %mul), kwargs = {})
#   %permute : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%add_4, [1, 0, 2]), kwargs = {})
#   %view_22 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, %arg0_1, 256]), kwargs = {})
#   %add_174 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %view_22), kwargs = {})
#   %view_47 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_7, [1, %arg0_1, 256]), kwargs = {})
#   %add_341 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_174, %view_47), kwargs = {})
#   %view_51 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_9, [1, %arg0_1, 256]), kwargs = {})
#   %add_386 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_341, %view_51), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_386, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_125 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_386, %getitem_25), kwargs = {})
#   %add_390 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_390,), kwargs = {})
#   %mul_466 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_125, %rsqrt_3), kwargs = {})
#   %mul_467 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_466, %arg35_1), kwargs = {})
#   %add_391 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_467, %arg36_1), kwargs = {})
#   %convert_element_type_67 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_391, torch.bfloat16), kwargs = {})
#   %convert_element_type_73 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_391, torch.bfloat16), kwargs = {})
#   %convert_element_type_79 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_391, torch.bfloat16), kwargs = {})
#   return %add_386,%getitem_25,%buf103,%add_391,%convert_element_type_67,%convert_element_type_73,%convert_element_type_79
triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_19 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'out_ptr6': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 4, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4194304, 'r0_': 31459328}}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr4, out_ptr5, out_ptr6, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + ks0*r0_1), xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr6 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tl.full([1, 1], 0.1, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tl.full([1, 1], 256, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.float32)
    tmp30 = tmp13 - tmp23
    tmp31 = tl.full([1, 1], 256.0, tl.float32)
    tmp32 = (tmp29 / tmp31)
    tmp33 = tl.full([1, 1], 1e-05, tl.float32)
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tmp40.to(tl.float32)
    tl.store(out_ptr0 + (r0_1 + 256*x0), tmp13, xmask)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp41, xmask)
    tl.store(out_ptr5 + (r0_1 + 256*x0), tmp41, xmask)
    tl.store(out_ptr6 + (r0_1 + 256*x0), tmp41, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/dy/cdyuvxccexc7vjtggh2d2iamxn4j4naqhtwjc3nrw2z3bqwumq5z.py
# Topologically Sorted Source Nodes: [out_8, tgt_3, tgt2_5, q_9], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   out_8 => view_74
#   q_9 => convert_element_type_94
#   tgt2_5 => add_547, add_548, mul_662, mul_663, rsqrt_4, sub_175, var_mean_4
#   tgt_3 => add_543
# Graph fragment:
#   %add_386 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_386]
#   %addmm_13 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_13]
#   %getitem_36 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_36]
#   %buf146 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf146]
#   %arg46_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg46_1]
#   %arg47_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg47_1]
#   %view_74 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_13, [1, %arg0_1, 256]), kwargs = {})
#   %add_543 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_386, %view_74), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_543, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_175 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_543, %getitem_36), kwargs = {})
#   %add_547 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_35, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_547,), kwargs = {})
#   %mul_662 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_175, %rsqrt_4), kwargs = {})
#   %mul_663 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_662, %arg46_1), kwargs = {})
#   %add_548 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_663, %arg47_1), kwargs = {})
#   %convert_element_type_94 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_548, torch.bfloat16), kwargs = {})
#   return %getitem_36,%buf146,%convert_element_type_94
triton_per_fused__to_copy_add_native_layer_norm_view_20 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_view_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 10487808}}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_view_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
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
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp31, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/wh/cwh6ack5xi5kbvpf574g6xnuu5mdpuucfz72tqtltkqldcmvg5wo.py
# Topologically Sorted Source Nodes: [out_8, tgt_3, out_11, tgt_4, tgt2_6, linear_18], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   linear_18 => convert_element_type_121
#   out_11 => view_99
#   out_8 => view_74
#   tgt2_6 => add_714, add_715, mul_867, mul_868, rsqrt_5, sub_228, var_mean_5
#   tgt_3 => add_543
#   tgt_4 => add_710
# Graph fragment:
#   %add_386 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_386]
#   %addmm_13 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_13]
#   %addmm_17 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %getitem_47 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_47]
#   %buf190 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf190]
#   %arg57_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %arg58_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %view_74 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_13, [1, %arg0_1, 256]), kwargs = {})
#   %add_543 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_386, %view_74), kwargs = {})
#   %view_99 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_17, [1, %arg0_1, 256]), kwargs = {})
#   %add_710 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_543, %view_99), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_710, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_228 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_710, %getitem_47), kwargs = {})
#   %add_714 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_5 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_714,), kwargs = {})
#   %mul_867 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_228, %rsqrt_5), kwargs = {})
#   %mul_868 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_867, %arg57_1), kwargs = {})
#   %add_715 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_868, %arg58_1), kwargs = {})
#   %convert_element_type_121 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_715, torch.bfloat16), kwargs = {})
#   return %getitem_47,%buf190,%convert_element_type_121
triton_per_fused__to_copy_add_native_layer_norm_view_21 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_view_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 12584960}}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_view_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp14 = tl.full([1, 1], 256, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tmp6 - tmp16
    tmp24 = tl.full([1, 1], 256.0, tl.float32)
    tmp25 = (tmp22 / tmp24)
    tmp26 = tl.full([1, 1], 1e-05, tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp34, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/tf/ctf32q4qnlr4t4w6sj2feh2hnpfujvu426h5x466z36ibsjeqx2n.py
# Topologically Sorted Source Nodes: [out_8, tgt_3, out_11, tgt_4, tgt2_7, tgt_5, tgt2_8, q_12, k_8, v_8], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   k_8 => convert_element_type_138
#   out_11 => view_99
#   out_8 => view_74
#   q_12 => convert_element_type_132
#   tgt2_7 => view_103
#   tgt2_8 => add_759, add_760, mul_913, mul_914, rsqrt_6, sub_243, var_mean_6
#   tgt_3 => add_543
#   tgt_4 => add_710
#   tgt_5 => add_755
#   v_8 => convert_element_type_144
# Graph fragment:
#   %add_386 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_386]
#   %addmm_13 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_13]
#   %addmm_17 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %addmm_19 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_19]
#   %getitem_49 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_49]
#   %buf201 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf201]
#   %arg63_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg63_1]
#   %arg64_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %add_760 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_760]
#   %view_74 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_13, [1, %arg0_1, 256]), kwargs = {})
#   %add_543 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_386, %view_74), kwargs = {})
#   %view_99 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_17, [1, %arg0_1, 256]), kwargs = {})
#   %add_710 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_543, %view_99), kwargs = {})
#   %view_103 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_19, [1, %arg0_1, 256]), kwargs = {})
#   %add_755 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_710, %view_103), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_755, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_243 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_755, %getitem_49), kwargs = {})
#   %add_759 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_48, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_759,), kwargs = {})
#   %mul_913 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_243, %rsqrt_6), kwargs = {})
#   %mul_914 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_913, %arg63_1), kwargs = {})
#   %add_760 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_914, %arg64_1), kwargs = {})
#   %convert_element_type_132 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_760, torch.bfloat16), kwargs = {})
#   %convert_element_type_138 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_760, torch.bfloat16), kwargs = {})
#   %convert_element_type_144 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_760, torch.bfloat16), kwargs = {})
#   return %getitem_49,%buf201,%add_760,%convert_element_type_132,%convert_element_type_138,%convert_element_type_144
triton_per_fused__to_copy_add_native_layer_norm_view_22 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_view_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_view_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 3, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 23070720}}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_view_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tl.full([1, 1], 256, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp26 = tmp9 - tmp19
    tmp27 = tl.full([1, 1], 256.0, tl.float32)
    tmp28 = (tmp25 / tmp27)
    tmp29 = tl.full([1, 1], 1e-05, tl.float32)
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 256*x0), tmp37, xmask)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp37, xmask)
    tl.store(out_ptr5 + (r0_1 + 256*x0), tmp37, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/63/c63icdnosgann5ip45h3seoqund6wejrft4kykjb5qtthhzcoohi.py
# Topologically Sorted Source Nodes: [out_8, tgt_3, out_11, tgt_4, tgt2_7, tgt_5, out_14, tgt_6, tgt2_9, q_15], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   out_11 => view_99
#   out_14 => view_126
#   out_8 => view_74
#   q_15 => convert_element_type_159
#   tgt2_7 => view_103
#   tgt2_9 => add_916, add_917, mul_1109, mul_1110, rsqrt_7, sub_293, var_mean_7
#   tgt_3 => add_543
#   tgt_4 => add_710
#   tgt_5 => add_755
#   tgt_6 => add_912
# Graph fragment:
#   %add_386 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_386]
#   %addmm_13 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_13]
#   %addmm_17 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %addmm_19 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_19]
#   %addmm_23 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_23]
#   %add_912 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_912]
#   %getitem_60 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_60]
#   %buf245 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf245]
#   %arg74_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg74_1]
#   %arg75_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %view_74 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_13, [1, %arg0_1, 256]), kwargs = {})
#   %add_543 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_386, %view_74), kwargs = {})
#   %view_99 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_17, [1, %arg0_1, 256]), kwargs = {})
#   %add_710 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_543, %view_99), kwargs = {})
#   %view_103 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_19, [1, %arg0_1, 256]), kwargs = {})
#   %add_755 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_710, %view_103), kwargs = {})
#   %view_126 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_23, [1, %arg0_1, 256]), kwargs = {})
#   %add_912 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_755, %view_126), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_912, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_293 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_912, %getitem_60), kwargs = {})
#   %add_916 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_59, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_916,), kwargs = {})
#   %mul_1109 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_293, %rsqrt_7), kwargs = {})
#   %mul_1110 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1109, %arg74_1), kwargs = {})
#   %add_917 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1110, %arg75_1), kwargs = {})
#   %convert_element_type_159 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_917, torch.bfloat16), kwargs = {})
#   return %add_912,%getitem_60,%buf245,%convert_element_type_159
triton_per_fused__to_copy_add_native_layer_norm_view_23 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_view_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 25167872}}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_view_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp4 = tl.load(in_ptr1 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp20 = tl.full([1, 1], 256, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = (tmp19 / tmp21)
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = tmp12 - tmp22
    tmp30 = tl.full([1, 1], 256.0, tl.float32)
    tmp31 = (tmp28 / tmp30)
    tmp32 = tl.full([1, 1], 1e-05, tl.float32)
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp12, xmask)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp40, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/wo/cwomz7pyvsrjvvih6nlsyftqzxa42xphyxdt3di43pgruxbnlgra.py
# Topologically Sorted Source Nodes: [out_17, tgt_7, tgt2_11, tgt_8, tgt2_12, q_18, k_12, v_12], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   k_12 => convert_element_type_203
#   out_17 => view_151
#   q_18 => convert_element_type_197
#   tgt2_11 => view_155
#   tgt2_12 => add_1128, add_1129, mul_1360, mul_1361, rsqrt_9, sub_361, var_mean_9
#   tgt_7 => add_1079
#   tgt_8 => add_1124
#   v_12 => convert_element_type_209
# Graph fragment:
#   %add_912 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_912]
#   %addmm_27 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_27]
#   %addmm_29 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_29]
#   %getitem_73 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_73]
#   %buf300 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf300]
#   %arg91_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg91_1]
#   %arg92_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg92_1]
#   %add_1129 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_1129]
#   %view_151 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_27, [1, %arg0_1, 256]), kwargs = {})
#   %add_1079 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_912, %view_151), kwargs = {})
#   %view_155 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_29, [1, %arg0_1, 256]), kwargs = {})
#   %add_1124 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1079, %view_155), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1124, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_361 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1124, %getitem_73), kwargs = {})
#   %add_1128 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-05), kwargs = {})
#   %rsqrt_9 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1128,), kwargs = {})
#   %mul_1360 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_361, %rsqrt_9), kwargs = {})
#   %mul_1361 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1360, %arg91_1), kwargs = {})
#   %add_1129 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1361, %arg92_1), kwargs = {})
#   %convert_element_type_197 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1129, torch.bfloat16), kwargs = {})
#   %convert_element_type_203 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1129, torch.bfloat16), kwargs = {})
#   %convert_element_type_209 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1129, torch.bfloat16), kwargs = {})
#   return %getitem_73,%buf300,%add_1129,%convert_element_type_197,%convert_element_type_203,%convert_element_type_209
triton_per_fused__to_copy_add_native_layer_norm_view_24 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_view_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 20973568}}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_view_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp14 = tl.full([1, 1], 256, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tmp6 - tmp16
    tmp24 = tl.full([1, 1], 256.0, tl.float32)
    tmp25 = (tmp22 / tmp24)
    tmp26 = tl.full([1, 1], 1e-05, tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 256*x0), tmp34, xmask)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp34, xmask)
    tl.store(out_ptr5 + (r0_1 + 256*x0), tmp34, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/zn/cznko3zyfmkbumbzeehrw5thfmahf4u2t3qljmqqcto3kp3kj5xy.py
# Topologically Sorted Source Nodes: [out_17, tgt_7, tgt2_11, tgt_8, out_20, tgt_9, tgt2_13, q_21], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   out_17 => view_151
#   out_20 => view_178
#   q_21 => convert_element_type_224
#   tgt2_11 => view_155
#   tgt2_13 => add_1285, add_1286, mul_1556, mul_1557, rsqrt_10, sub_411, var_mean_10
#   tgt_7 => add_1079
#   tgt_8 => add_1124
#   tgt_9 => add_1281
# Graph fragment:
#   %add_912 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_912]
#   %addmm_27 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_27]
#   %addmm_29 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_29]
#   %addmm_33 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_33]
#   %getitem_84 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_84]
#   %buf343 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf343]
#   %arg102_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg102_1]
#   %arg103_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg103_1]
#   %view_151 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_27, [1, %arg0_1, 256]), kwargs = {})
#   %add_1079 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_912, %view_151), kwargs = {})
#   %view_155 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_29, [1, %arg0_1, 256]), kwargs = {})
#   %add_1124 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1079, %view_155), kwargs = {})
#   %view_178 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_33, [1, %arg0_1, 256]), kwargs = {})
#   %add_1281 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1124, %view_178), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1281, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_411 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1281, %getitem_84), kwargs = {})
#   %add_1285 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_83, 1e-05), kwargs = {})
#   %rsqrt_10 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1285,), kwargs = {})
#   %mul_1556 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_411, %rsqrt_10), kwargs = {})
#   %mul_1557 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1556, %arg102_1), kwargs = {})
#   %add_1286 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1557, %arg103_1), kwargs = {})
#   %convert_element_type_224 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1286, torch.bfloat16), kwargs = {})
#   return %getitem_84,%buf343,%convert_element_type_224
triton_per_fused__to_copy_add_native_layer_norm_view_25 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_view_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 14682112}}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_view_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tl.full([1, 1], 256, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp26 = tmp9 - tmp19
    tmp27 = tl.full([1, 1], 256.0, tl.float32)
    tmp28 = (tmp25 / tmp27)
    tmp29 = tl.full([1, 1], 1e-05, tl.float32)
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp37, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/5y/c5yvi5n7vibuute4dfrshtzkhe6b4gpwftyn2ndvtp7z6inoaozr.py
# Topologically Sorted Source Nodes: [tgt2_14, linear_38, relu_3], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear_38 => addmm_38, convert_element_type_249, convert_element_type_250, convert_element_type_251, permute_98, view_204, view_205
#   relu_3 => relu_3
#   tgt2_14 => add_1452, add_1453, mul_1761, mul_1762, rsqrt_11, sub_464, var_mean_11
# Graph fragment:
#   %convert_element_type_249 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=convert_element_type_249]
#   %convert_element_type_251 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=convert_element_type_251]
#   %convert_element_type_250 : Tensor "bf16[2048, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_250]
#   %addmm_38 : Tensor "bf16[s14, 2048][2048, 1]cuda:0" = PlaceHolder[target=addmm_38]
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1448, [2]), kwargs = {correction: 0, keepdim: True})
#   %convert_element_type_249 : Tensor "bf16[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg116_1, torch.bfloat16), kwargs = {})
#   %sub_464 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1448, %getitem_95), kwargs = {})
#   %add_1452 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_94, 1e-05), kwargs = {})
#   %rsqrt_11 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1452,), kwargs = {})
#   %mul_1761 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_464, %rsqrt_11), kwargs = {})
#   %mul_1762 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1761, %arg113_1), kwargs = {})
#   %add_1453 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1762, %arg114_1), kwargs = {})
#   %convert_element_type_251 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1453, torch.bfloat16), kwargs = {})
#   %view_204 : Tensor "bf16[s14, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_251, [%arg0_1, 256]), kwargs = {})
#   %convert_element_type_250 : Tensor "bf16[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg115_1, torch.bfloat16), kwargs = {})
#   %permute_98 : Tensor "bf16[256, 2048][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_250, [1, 0]), kwargs = {})
#   %addmm_38 : Tensor "bf16[s14, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_249, %view_204, %permute_98), kwargs = {})
#   %view_205 : Tensor "bf16[1, s14, 2048][2048*s14, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_38, [1, %arg0_1, 2048]), kwargs = {})
#   %relu_3 : Tensor "bf16[1, s14, 2048][2048*s14, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_205,), kwargs = {})
#   return %addmm_38,%relu_3
triton_tem_fused__to_copy_addmm_native_layer_norm_relu_t_view_26 = async_compile.triton('triton_tem_fused__to_copy_addmm_native_layer_norm_relu_t_view_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_native_layer_norm_relu_t_view_26', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_native_layer_norm_relu_t_view_26(in_ptr0, arg_A, arg_B, out_ptr1, ks0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = ks0
    N = 2048
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 256*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 2048*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 2048*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp3, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/5m/c5mlbmg3np7i7hwjeydr36y3wqj3kkyxlgzu5uv3m4yddiyzdsuo.py
# Topologically Sorted Source Nodes: [tgt2_15, tgt_11, normed_output], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   normed_output => add_1497, add_1498, mul_1807, mul_1808, rsqrt_12, sub_479, var_mean_12
#   tgt2_15 => view_207
#   tgt_11 => add_1493
# Graph fragment:
#   %add_1448 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0" = PlaceHolder[target=add_1448]
#   %addmm_39 : Tensor "bf16[s14, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_39]
#   %getitem_97 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=getitem_97]
#   %buf399 : Tensor "f32[1, s14, 1][s14, 1, s14]cuda:0" = PlaceHolder[target=buf399]
#   %arg119_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg119_1]
#   %arg120_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg120_1]
#   %view_207 : Tensor "bf16[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_39, [1, %arg0_1, 256]), kwargs = {})
#   %add_1493 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1448, %view_207), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1493, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_479 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1493, %getitem_97), kwargs = {})
#   %add_1497 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_96, 1e-05), kwargs = {})
#   %rsqrt_12 : Tensor "f32[1, s14, 1][s14, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1497,), kwargs = {})
#   %mul_1807 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_479, %rsqrt_12), kwargs = {})
#   %mul_1808 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1807, %arg119_1), kwargs = {})
#   %add_1498 : Tensor "f32[1, s14, 256][256*s14, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1808, %arg120_1), kwargs = {})
#   return %getitem_97,%buf399,%add_1498
triton_per_fused_add_native_layer_norm_view_27 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 14682112}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp30, xmask)
''', device_str='cuda')

def partition_0(args):
    arg1_1, arg4_1, arg6_1, arg7_1, arg8_1, arg9_1, arg14_1, arg10_1, arg11_1, arg12_1, arg13_1, arg15_1, arg16_1, arg18_1, arg19_1, arg20_1, arg21_1, arg26_1, arg3_1, arg5_1, arg22_1, arg23_1, arg24_1, arg25_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg43_1, arg39_1, arg40_1, arg41_1, arg42_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg54_1, arg50_1, arg51_1, arg52_1, arg53_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg71_1, arg67_1, arg68_1, arg69_1, arg70_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg82_1, arg78_1, arg79_1, arg80_1, arg81_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg99_1, arg95_1, arg96_1, arg97_1, arg98_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg110_1, arg106_1, arg107_1, arg108_1, arg109_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, s14, s46, s98 = args
    args.clear()
    s14 = s14
    s46 = s46
    s98 = s98
    assert_size_stride(arg1_1, (s14, 1, 256), (256, 256, 1))
    assert_size_stride(arg4_1, (s14, 1, 256), (1, 256*s14, s14))
    assert_size_stride(arg6_1, (256, ), (1, ))
    assert_size_stride(arg7_1, (256, ), (1, ))
    assert_size_stride(arg8_1, (256, 256), (256, 1))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (s14, 128), (128, 1))
    assert_size_stride(arg10_1, (256, 256), (256, 1))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, 256), (256, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, 256), (256, 1))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, 256), (256, 1))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (s14, 128), (128, 1))
    assert_size_stride(arg3_1, (s98, 1, 64), (64, 64, 1))
    assert_size_stride(arg5_1, (s98, 1, 64), (64, 64, 1))
    assert_size_stride(arg22_1, (256, 64), (64, 1))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, 64), (64, 1))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, 256), (256, 1))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (2048, 256), (256, 1))
    assert_size_stride(arg32_1, (2048, ), (1, ))
    assert_size_stride(arg33_1, (256, 2048), (2048, 1))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, 256), (256, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (s14, 128), (128, 1))
    assert_size_stride(arg39_1, (256, 256), (256, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, 256), (256, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, 256), (256, 1))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, 256), (256, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (s14, 128), (128, 1))
    assert_size_stride(arg50_1, (256, 64), (64, 1))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, 64), (64, 1))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (256, 256), (256, 1))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (2048, 256), (256, 1))
    assert_size_stride(arg60_1, (2048, ), (1, ))
    assert_size_stride(arg61_1, (256, 2048), (2048, 1))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (256, 256), (256, 1))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (s14, 128), (128, 1))
    assert_size_stride(arg67_1, (256, 256), (256, 1))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (256, 256), (256, 1))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (256, 256), (256, 1))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, ), (1, ))
    assert_size_stride(arg76_1, (256, 256), (256, 1))
    assert_size_stride(arg77_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (s14, 128), (128, 1))
    assert_size_stride(arg78_1, (256, 64), (64, 1))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (256, 64), (64, 1))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, 256), (256, 1))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (2048, 256), (256, 1))
    assert_size_stride(arg88_1, (2048, ), (1, ))
    assert_size_stride(arg89_1, (256, 2048), (2048, 1))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, 256), (256, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (s14, 128), (128, 1))
    assert_size_stride(arg95_1, (256, 256), (256, 1))
    assert_size_stride(arg96_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (256, 256), (256, 1))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg100_1, (256, 256), (256, 1))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (256, 256), (256, 1))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg110_1, (s14, 128), (128, 1))
    assert_size_stride(arg106_1, (256, 64), (64, 1))
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (256, 64), (64, 1))
    assert_size_stride(arg109_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, 256), (256, 1))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (256, ), (1, ))
    assert_size_stride(arg114_1, (256, ), (1, ))
    assert_size_stride(arg115_1, (2048, 256), (256, 1))
    assert_size_stride(arg116_1, (2048, ), (1, ))
    assert_size_stride(arg117_1, (256, 2048), (2048, 1))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (256, ), (1, ))
    assert_size_stride(arg120_1, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s14, 1, 2), (2*s14, 2, 2*s14, 1), torch.float32)
        buf1 = empty_strided_cuda((1, s14, 1, 2), (2*s14, 2, 2*s14, 1), torch.float32)
        buf2 = empty_strided_cuda((1, s14, 1, 2), (2*s14, 2, 2*s14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, output, output_1, tgt2], Original ATen: [aten.mul, aten.add, aten.transpose, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_native_layer_norm_transpose_0.run(arg1_1, arg4_1, buf0, buf1, buf2, s14, s14, 2, 128, stream=stream0)
        buf3 = empty_strided_cuda((1, s14, 1), (s14, 1, s14), torch.float32)
        buf4 = empty_strided_cuda((1, s14, 1), (s14, 1, s14), torch.float32)
        # Topologically Sorted Source Nodes: [mul, output, output_1, tgt2], Original ATen: [aten.mul, aten.add, aten.transpose, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mul_native_layer_norm_transpose_1.run(buf0, buf1, buf2, buf3, buf4, s14, 2, stream=stream0)
        del buf0
        del buf1
        del buf2
        buf7 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        buf20 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        buf31 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul, output, output_1, tgt2, q, k, v], Original ATen: [aten.mul, aten.add, aten.transpose, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mul_native_layer_norm_transpose_2.run(arg1_1, arg4_1, buf3, buf4, arg6_1, arg7_1, buf7, buf20, buf31, s14, s14, 256, stream=stream0)
        del arg6_1
        del arg7_1
        del buf3
        del buf4
        buf8 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg8_1, buf8, 65536, stream=stream0)
        del arg8_1
        buf9 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg9_1, buf9, 256, stream=stream0)
        del arg9_1
        buf10 = empty_strided_cuda((s14, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf9, buf7, buf8, buf10, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf7
        buf11 = empty_strided_cuda((1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q, x, q_1, float_1, reshape_3, xq_], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf10, buf11, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [q, x, q_1, float_1, reshape_3, xq_], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf12 = torch.ops.aten.view_as_complex.default(buf11)
        buf13 = buf12
        assert_size_stride(buf13, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf13, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis], Original ATen: [aten.view]
        buf14 = torch.ops.aten.reshape.default(arg14_1, [1, 1, s14, 128])
        buf15 = buf14
        assert_size_stride(buf15, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.reshape.default')
        assert_alignment(buf15, 16, 'torch.ops.aten.reshape.default')
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        buf16 = torch.ops.aten.mul.Tensor(buf13, buf15)
        del buf12
        del buf13
        buf17 = buf16
        assert_size_stride(buf17, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf17, 16, 'torch.ops.aten.mul.Tensor')
        del buf16
        # Topologically Sorted Source Nodes: [view_as_real], Original ATen: [aten.view_as_real]
        buf18 = torch.ops.aten.view_as_real.default(buf17)
        buf19 = buf18
        assert_size_stride(buf19, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf19, 16, 'torch.ops.aten.view_as_real.default')
        buf21 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [k], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg10_1, buf21, 65536, stream=stream0)
        del arg10_1
        buf22 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [k], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg11_1, buf22, 256, stream=stream0)
        del arg11_1
        buf23 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [k], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf22, buf20, buf21, buf23, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf24 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [k, x_1, k_1, float_2, reshape_4, xk_], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf23, buf24, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [k, x_1, k_1, float_2, reshape_4, xk_], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf25 = torch.ops.aten.view_as_complex.default(buf24)
        buf26 = buf25
        assert_size_stride(buf26, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf26, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [mul_2], Original ATen: [aten.mul]
        buf27 = torch.ops.aten.mul.Tensor(buf26, buf15)
        del buf25
        del buf26
        buf28 = buf27
        assert_size_stride(buf28, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf28, 16, 'torch.ops.aten.mul.Tensor')
        del buf27
        # Topologically Sorted Source Nodes: [view_as_real_1], Original ATen: [aten.view_as_real]
        buf29 = torch.ops.aten.view_as_real.default(buf28)
        buf30 = buf29
        assert_size_stride(buf30, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf30, 16, 'torch.ops.aten.view_as_real.default')
        buf32 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg12_1, buf32, 65536, stream=stream0)
        del arg12_1
        buf33 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg13_1, buf33, 256, stream=stream0)
        del arg13_1
        buf34 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf33, buf31, buf32, buf34, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf35 = reinterpret_tensor(buf31, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [xq_out, type_as, k, x_1, k_1, xk_out, type_as_1, setitem, v, x_2, v_1, out], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf19, buf35, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf17
        del buf18
        del buf19
        buf36 = reinterpret_tensor(buf20, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [xq_out, type_as, k, x_1, k_1, xk_out, type_as_1, setitem, v, x_2, v_1, out], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf30, buf36, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf28
        del buf29
        del buf30
        # Topologically Sorted Source Nodes: [xq_out, type_as, k, x_1, k_1, xk_out, type_as_1, setitem, v, x_2, v_1, out], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        buf37 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf35, buf36, reinterpret_tensor(buf34, (1, 1, s14, 256), (256*s14, 256, 256, 1), 0), scale=0.0625)
        del buf34
        buf38 = buf37[0]
        assert_size_stride(buf38, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf38, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf37
        buf43 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg15_1, buf43, 65536, stream=stream0)
        del arg15_1
        buf44 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg16_1, buf44, 256, stream=stream0)
        del arg16_1
        buf45 = reinterpret_tensor(buf36, (s14, 256), (256, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [out_2, x_3, out_1], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf44, buf38, buf43, buf45, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf49 = reinterpret_tensor(buf38, (1, s14, 256), (256*s14, 256, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [mul, output, output_1, out_2, tgt, tgt2_1, q_3], Original ATen: [aten.mul, aten.add, aten.transpose, aten.view, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_8.run(arg1_1, arg4_1, buf45, arg18_1, arg19_1, buf49, s14, s14, 256, stream=stream0)
        del arg18_1
        del arg19_1
        buf50 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [q_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg20_1, buf50, 65536, stream=stream0)
        del arg20_1
        buf51 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [q_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg21_1, buf51, 256, stream=stream0)
        del arg21_1
        buf52 = reinterpret_tensor(buf35, (s14, 256), (256, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [mul, output, output_1, out_2, tgt, tgt2_1, q_3], Original ATen: [aten.mul, aten.add, aten.transpose, aten.view, aten.native_layer_norm, aten._to_copy, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf51, buf49, buf50, buf52, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf49
        buf53 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [q_3, x_4, q_4, float_3, reshape_9, xq__1], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf52, buf53, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [q_3, x_4, q_4, float_3, reshape_9, xq__1], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf54 = torch.ops.aten.view_as_complex.default(buf53)
        buf55 = buf54
        assert_size_stride(buf55, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf55, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_1], Original ATen: [aten.view]
        buf56 = torch.ops.aten.reshape.default(arg26_1, [1, 1, s14, 128])
        buf57 = buf56
        assert_size_stride(buf57, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.reshape.default')
        assert_alignment(buf57, 16, 'torch.ops.aten.reshape.default')
        # Topologically Sorted Source Nodes: [mul_3], Original ATen: [aten.mul]
        buf58 = torch.ops.aten.mul.Tensor(buf55, buf57)
        del buf53
        del buf54
        del buf55
        buf59 = buf58
        assert_size_stride(buf59, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf59, 16, 'torch.ops.aten.mul.Tensor')
        del buf58
        # Topologically Sorted Source Nodes: [view_as_real_2], Original ATen: [aten.view_as_real]
        buf60 = torch.ops.aten.view_as_real.default(buf59)
        buf61 = buf60
        assert_size_stride(buf61, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf61, 16, 'torch.ops.aten.view_as_real.default')
        buf62 = empty_strided_cuda((1, s98, 64), (64*s98, 64, 1), torch.bfloat16)
        buf75 = empty_strided_cuda((1, s98, 64), (64*s98, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, memory_pos, add_2, k_2, v_2], Original ATen: [aten.transpose, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_transpose_9_xnumel = 64*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_transpose_9.run(arg3_1, arg5_1, buf62, buf75, triton_poi_fused__to_copy_add_transpose_9_xnumel, stream=stream0)
        buf63 = empty_strided_cuda((256, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [k_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(arg22_1, buf63, 16384, stream=stream0)
        del arg22_1
        buf64 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [k_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg23_1, buf64, 256, stream=stream0)
        del arg23_1
        buf65 = empty_strided_cuda((s98, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [k_2, memory, memory_pos, add_2], Original ATen: [aten._to_copy, aten.transpose, aten.add, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_t_transpose_view_11.run(buf64, buf62, buf63, buf65, s98, 4*((31 + s98) // 32), 1, 1, stream=stream0)
        del buf62
        buf66 = empty_strided_cuda((1, 1, s98 + ((-1)*s46), 128, 2), (((-256)*s46) + 256*s98, ((-256)*s46) + 256*s98, 256, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_2, x_5, k_3, getitem_91, float_4, reshape_10, xk__1], Original ATen: [aten.view, aten.transpose, aten.slice, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = ((-256)*s46) + 256*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf65, buf66, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [k_2, x_5, k_3, getitem_91, float_4, reshape_10, xk__1], Original ATen: [aten.view, aten.transpose, aten.slice, aten._to_copy, aten.view_as_complex]
        buf67 = torch.ops.aten.view_as_complex.default(buf66)
        buf68 = buf67
        assert_size_stride(buf68, (1, 1, s98 + ((-1)*s46), 128), (((-128)*s46) + 128*s98, ((-128)*s46) + 128*s98, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf68, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_2], Original ATen: [aten.repeat]
        buf69 = torch.ops.aten.repeat.default(buf57, [1, 1, (s98 + ((-1)*s46)) // s14, 1])
        buf70 = buf69
        assert_size_stride(buf70, (1, 1, s14*((s98 + ((-1)*s46)) // s14), 128), (128*max(1, s14*((s98 + ((-1)*s46)) // s14)), 128*max(1, s14*((s98 + ((-1)*s46)) // s14)), 128, 1), 'torch.ops.aten.repeat.default')
        assert_alignment(buf70, 16, 'torch.ops.aten.repeat.default')
        del buf69
        # Topologically Sorted Source Nodes: [mul_4], Original ATen: [aten.mul]
        buf71 = torch.ops.aten.mul.Tensor(buf68, buf70)
        del buf66
        del buf67
        del buf68
        del buf70
        buf72 = buf71
        assert_size_stride(buf72, (1, 1, s98 + ((-1)*s46), 128), (128*max(1, s98 + ((-1)*s46)), 128*max(1, s98 + ((-1)*s46)), 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf72, 16, 'torch.ops.aten.mul.Tensor')
        del buf71
        # Topologically Sorted Source Nodes: [view_as_real_3], Original ATen: [aten.view_as_real]
        buf73 = torch.ops.aten.view_as_real.default(buf72)
        buf74 = buf73
        assert_size_stride(buf74, (1, 1, s98 + ((-1)*s46), 128, 2), (256*max(1, s98 + ((-1)*s46)), 256*max(1, s98 + ((-1)*s46)), 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf74, 16, 'torch.ops.aten.view_as_real.default')
        buf76 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [v_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(arg24_1, buf76, 16384, stream=stream0)
        del arg24_1
        buf77 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [v_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg25_1, buf77, 256, stream=stream0)
        del arg25_1
        buf78 = empty_strided_cuda((s98, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, v_2], Original ATen: [aten.transpose, aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_t_transpose_view_11.run(buf77, buf75, buf76, buf78, s98, 4*((31 + s98) // 32), 1, 1, stream=stream0)
        buf79 = reinterpret_tensor(buf52, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [xq_out_1, type_as_2, setitem_1, k_2, x_5, k_3, xk_out_1, type_as_3, out_3, v_2, x_6, v_3], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf61, buf79, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf59
        del buf60
        del buf61
        buf80 = reinterpret_tensor(buf65, (1, 1, s98, 256), (256*s98, 256*s98, 256, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [xq_out_1, type_as_2, setitem_1, k_2, x_5, k_3, xk_out_1, type_as_3, out_3, v_2, x_6, v_3], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12_xnumel = 256*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12.run(buf80, buf74, s46, s98, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12_xnumel, stream=stream0)
        del buf72
        del buf73
        del buf74
        # Topologically Sorted Source Nodes: [xq_out_1, type_as_2, setitem_1, k_2, x_5, k_3, xk_out_1, type_as_3, out_3, v_2, x_6, v_3], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        buf81 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf79, buf80, reinterpret_tensor(buf78, (1, 1, s98, 256), (256*s98, 256, 256, 1), 0), scale=0.0625)
        del buf78
        del buf80
        buf82 = buf81[0]
        assert_size_stride(buf82, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf82, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf81
        buf87 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg27_1, buf87, 65536, stream=stream0)
        del arg27_1
        buf88 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg28_1, buf88, 256, stream=stream0)
        del arg28_1
        buf89 = reinterpret_tensor(buf79, (s14, 256), (256, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [out_5, x_7, out_4], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf88, buf82, buf87, buf89, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf93 = reinterpret_tensor(buf82, (1, s14, 256), (256*s14, 256, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [mul, output, output_1, out_2, tgt, out_5, tgt_1, tgt2_2, linear_8], Original ATen: [aten.mul, aten.add, aten.transpose, aten.view, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_13.run(arg1_1, arg4_1, buf45, buf89, arg29_1, arg30_1, buf93, s14, s14, 256, stream=stream0)
        del arg29_1
        del arg30_1
        buf94 = empty_strided_cuda((2048, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(arg31_1, buf94, 524288, stream=stream0)
        del arg31_1
        buf95 = empty_strided_cuda((2048, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(arg32_1, buf95, 2048, stream=stream0)
        del arg32_1
        buf96 = empty_strided_cuda((s14, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul, output, output_1, out_2, tgt, out_5, tgt_1, tgt2_2, linear_8], Original ATen: [aten.mul, aten.add, aten.transpose, aten.view, aten.native_layer_norm, aten._to_copy, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_mul_native_layer_norm_t_transpose_view_16.run(buf95, buf93, buf94, buf96, s14, 32*((127 + s14) // 128), 1, 1, stream=stream0)
        buf97 = reinterpret_tensor(buf96, (1, s14, 2048), (2048*s14, 2048, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [linear_8, relu], Original ATen: [aten.view, aten.relu]
        triton_poi_fused_relu_view_17_xnumel = 2048*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_view_17.run(buf97, triton_poi_fused_relu_view_17_xnumel, stream=stream0)
        buf98 = reinterpret_tensor(buf94, (256, 2048), (2048, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [tgt2_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(arg33_1, buf98, 524288, stream=stream0)
        del arg33_1
        buf99 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [tgt2_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg34_1, buf99, 256, stream=stream0)
        del arg34_1
        buf100 = reinterpret_tensor(buf93, (s14, 256), (256, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [tgt2_3, linear_8, relu], Original ATen: [aten._to_copy, aten.view, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_view_18.run(buf99, buf97, buf98, buf100, s14, 2*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf97
        buf101 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.float32)
        buf106 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        buf119 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        buf130 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul, output, output_1, out_2, tgt, out_5, tgt_1, tgt2_3, tgt_2, tgt2_4, q_6, k_4, v_4], Original ATen: [aten.mul, aten.add, aten.transpose, aten.view, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_native_layer_norm_transpose_view_19.run(arg1_1, arg4_1, buf45, buf89, buf100, arg35_1, arg36_1, buf101, buf106, buf119, buf130, s14, s14, 256, stream=stream0)
        del arg1_1
        del arg35_1
        del arg36_1
        del arg4_1
        del buf100
        del buf45
        buf107 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [q_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg37_1, buf107, 65536, stream=stream0)
        del arg37_1
        buf108 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [q_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg38_1, buf108, 256, stream=stream0)
        del arg38_1
        buf109 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [q_6], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf108, buf106, buf107, buf109, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf106
        buf110 = empty_strided_cuda((1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_6, x_8, q_7, float_5, reshape_15, xq__2], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf109, buf110, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [q_6, x_8, q_7, float_5, reshape_15, xq__2], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf111 = torch.ops.aten.view_as_complex.default(buf110)
        buf112 = buf111
        assert_size_stride(buf112, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf112, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_3], Original ATen: [aten.view]
        buf113 = torch.ops.aten.reshape.default(arg43_1, [1, 1, s14, 128])
        buf114 = buf113
        assert_size_stride(buf114, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.reshape.default')
        assert_alignment(buf114, 16, 'torch.ops.aten.reshape.default')
        # Topologically Sorted Source Nodes: [mul_5], Original ATen: [aten.mul]
        buf115 = torch.ops.aten.mul.Tensor(buf112, buf114)
        del buf111
        del buf112
        buf116 = buf115
        assert_size_stride(buf116, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf116, 16, 'torch.ops.aten.mul.Tensor')
        del buf115
        # Topologically Sorted Source Nodes: [view_as_real_4], Original ATen: [aten.view_as_real]
        buf117 = torch.ops.aten.view_as_real.default(buf116)
        buf118 = buf117
        assert_size_stride(buf118, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf118, 16, 'torch.ops.aten.view_as_real.default')
        buf120 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [k_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg39_1, buf120, 65536, stream=stream0)
        del arg39_1
        buf121 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [k_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg40_1, buf121, 256, stream=stream0)
        del arg40_1
        buf122 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [k_4], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf121, buf119, buf120, buf122, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf123 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [k_4, x_9, k_5, float_6, reshape_16, xk__2], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf122, buf123, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [k_4, x_9, k_5, float_6, reshape_16, xk__2], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf124 = torch.ops.aten.view_as_complex.default(buf123)
        buf125 = buf124
        assert_size_stride(buf125, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf125, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [mul_6], Original ATen: [aten.mul]
        buf126 = torch.ops.aten.mul.Tensor(buf125, buf114)
        del buf124
        del buf125
        buf127 = buf126
        assert_size_stride(buf127, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf127, 16, 'torch.ops.aten.mul.Tensor')
        del buf126
        # Topologically Sorted Source Nodes: [view_as_real_5], Original ATen: [aten.view_as_real]
        buf128 = torch.ops.aten.view_as_real.default(buf127)
        buf129 = buf128
        assert_size_stride(buf129, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf129, 16, 'torch.ops.aten.view_as_real.default')
        buf131 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg41_1, buf131, 65536, stream=stream0)
        del arg41_1
        buf132 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg42_1, buf132, 256, stream=stream0)
        del arg42_1
        buf133 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf132, buf130, buf131, buf133, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf134 = reinterpret_tensor(buf130, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [xq_out_2, type_as_4, k_4, x_9, k_5, xk_out_2, type_as_5, setitem_2, v_4, x_10, v_5, out_6], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf118, buf134, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf116
        del buf117
        del buf118
        buf135 = reinterpret_tensor(buf119, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [xq_out_2, type_as_4, k_4, x_9, k_5, xk_out_2, type_as_5, setitem_2, v_4, x_10, v_5, out_6], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf129, buf135, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf127
        del buf128
        del buf129
        # Topologically Sorted Source Nodes: [xq_out_2, type_as_4, k_4, x_9, k_5, xk_out_2, type_as_5, setitem_2, v_4, x_10, v_5, out_6], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        buf136 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf134, buf135, reinterpret_tensor(buf133, (1, 1, s14, 256), (256*s14, 256, 256, 1), 0), scale=0.0625)
        del buf133
        buf137 = buf136[0]
        assert_size_stride(buf137, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf137, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf136
        buf142 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg44_1, buf142, 65536, stream=stream0)
        del arg44_1
        buf143 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg45_1, buf143, 256, stream=stream0)
        del arg45_1
        buf144 = reinterpret_tensor(buf135, (s14, 256), (256, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [out_8, x_11, out_7], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf143, buf137, buf142, buf144, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf148 = reinterpret_tensor(buf137, (1, s14, 256), (256*s14, 256, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [out_8, tgt_3, tgt2_5, q_9], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_view_20.run(buf101, buf144, arg46_1, arg47_1, buf148, s14, 256, stream=stream0)
        del arg46_1
        del arg47_1
        buf149 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [q_9], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg48_1, buf149, 65536, stream=stream0)
        del arg48_1
        buf150 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [q_9], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg49_1, buf150, 256, stream=stream0)
        del arg49_1
        buf151 = reinterpret_tensor(buf134, (s14, 256), (256, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [out_8, tgt_3, tgt2_5, q_9], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf150, buf148, buf149, buf151, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf148
        buf152 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [q_9, x_12, q_10, float_7, reshape_21, xq__3], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf151, buf152, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [q_9, x_12, q_10, float_7, reshape_21, xq__3], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf153 = torch.ops.aten.view_as_complex.default(buf152)
        buf154 = buf153
        assert_size_stride(buf154, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf154, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_4], Original ATen: [aten.view]
        buf155 = torch.ops.aten.reshape.default(arg54_1, [1, 1, s14, 128])
        buf156 = buf155
        assert_size_stride(buf156, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.reshape.default')
        assert_alignment(buf156, 16, 'torch.ops.aten.reshape.default')
        # Topologically Sorted Source Nodes: [mul_7], Original ATen: [aten.mul]
        buf157 = torch.ops.aten.mul.Tensor(buf154, buf156)
        del buf152
        del buf153
        del buf154
        buf158 = buf157
        assert_size_stride(buf158, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf158, 16, 'torch.ops.aten.mul.Tensor')
        del buf157
        # Topologically Sorted Source Nodes: [view_as_real_6], Original ATen: [aten.view_as_real]
        buf159 = torch.ops.aten.view_as_real.default(buf158)
        buf160 = buf159
        assert_size_stride(buf160, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf160, 16, 'torch.ops.aten.view_as_real.default')
        buf161 = buf75; del buf75  # reuse
        buf174 = empty_strided_cuda((1, s98, 64), (64*s98, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, memory_pos, add_6, k_6, v_6], Original ATen: [aten.transpose, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_transpose_9_xnumel = 64*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_transpose_9.run(arg3_1, arg5_1, buf161, buf174, triton_poi_fused__to_copy_add_transpose_9_xnumel, stream=stream0)
        buf162 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [k_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(arg50_1, buf162, 16384, stream=stream0)
        del arg50_1
        buf163 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [k_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg51_1, buf163, 256, stream=stream0)
        del arg51_1
        buf164 = empty_strided_cuda((s98, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, memory_pos, k_6, add_6], Original ATen: [aten.transpose, aten._to_copy, aten.add, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_t_transpose_view_11.run(buf163, buf161, buf162, buf164, s98, 4*((31 + s98) // 32), 1, 1, stream=stream0)
        del buf161
        buf165 = empty_strided_cuda((1, 1, s98 + ((-1)*s46), 128, 2), (((-256)*s46) + 256*s98, ((-256)*s46) + 256*s98, 256, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_6, x_13, k_7, getitem_215, float_8, reshape_22, xk__3], Original ATen: [aten.view, aten.transpose, aten.slice, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = ((-256)*s46) + 256*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf164, buf165, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [k_6, x_13, k_7, getitem_215, float_8, reshape_22, xk__3], Original ATen: [aten.view, aten.transpose, aten.slice, aten._to_copy, aten.view_as_complex]
        buf166 = torch.ops.aten.view_as_complex.default(buf165)
        buf167 = buf166
        assert_size_stride(buf167, (1, 1, s98 + ((-1)*s46), 128), (((-128)*s46) + 128*s98, ((-128)*s46) + 128*s98, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf167, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_5], Original ATen: [aten.repeat]
        buf168 = torch.ops.aten.repeat.default(buf156, [1, 1, (s98 + ((-1)*s46)) // s14, 1])
        buf169 = buf168
        assert_size_stride(buf169, (1, 1, s14*((s98 + ((-1)*s46)) // s14), 128), (128*max(1, s14*((s98 + ((-1)*s46)) // s14)), 128*max(1, s14*((s98 + ((-1)*s46)) // s14)), 128, 1), 'torch.ops.aten.repeat.default')
        assert_alignment(buf169, 16, 'torch.ops.aten.repeat.default')
        del buf168
        # Topologically Sorted Source Nodes: [mul_8], Original ATen: [aten.mul]
        buf170 = torch.ops.aten.mul.Tensor(buf167, buf169)
        del buf165
        del buf166
        del buf167
        del buf169
        buf171 = buf170
        assert_size_stride(buf171, (1, 1, s98 + ((-1)*s46), 128), (128*max(1, s98 + ((-1)*s46)), 128*max(1, s98 + ((-1)*s46)), 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf171, 16, 'torch.ops.aten.mul.Tensor')
        del buf170
        # Topologically Sorted Source Nodes: [view_as_real_7], Original ATen: [aten.view_as_real]
        buf172 = torch.ops.aten.view_as_real.default(buf171)
        buf173 = buf172
        assert_size_stride(buf173, (1, 1, s98 + ((-1)*s46), 128, 2), (256*max(1, s98 + ((-1)*s46)), 256*max(1, s98 + ((-1)*s46)), 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf173, 16, 'torch.ops.aten.view_as_real.default')
        buf175 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [v_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(arg52_1, buf175, 16384, stream=stream0)
        del arg52_1
        buf176 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [v_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg53_1, buf176, 256, stream=stream0)
        del arg53_1
        buf177 = empty_strided_cuda((s98, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, v_6], Original ATen: [aten.transpose, aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_t_transpose_view_11.run(buf176, buf174, buf175, buf177, s98, 4*((31 + s98) // 32), 1, 1, stream=stream0)
        del buf174
        del buf175
        buf178 = reinterpret_tensor(buf151, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [xq_out_3, type_as_6, setitem_3, k_6, x_13, k_7, xk_out_3, type_as_7, out_9, v_6, x_14, v_7], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf160, buf178, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf158
        del buf159
        del buf160
        buf179 = reinterpret_tensor(buf164, (1, 1, s98, 256), (256*s98, 256*s98, 256, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [xq_out_3, type_as_6, setitem_3, k_6, x_13, k_7, xk_out_3, type_as_7, out_9, v_6, x_14, v_7], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12_xnumel = 256*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12.run(buf179, buf173, s46, s98, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12_xnumel, stream=stream0)
        del buf171
        del buf172
        del buf173
        # Topologically Sorted Source Nodes: [xq_out_3, type_as_6, setitem_3, k_6, x_13, k_7, xk_out_3, type_as_7, out_9, v_6, x_14, v_7], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        buf180 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf178, buf179, reinterpret_tensor(buf177, (1, 1, s98, 256), (256*s98, 256, 256, 1), 0), scale=0.0625)
        del buf177
        del buf179
        buf181 = buf180[0]
        assert_size_stride(buf181, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf181, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf180
        buf186 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg55_1, buf186, 65536, stream=stream0)
        del arg55_1
        buf187 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg56_1, buf187, 256, stream=stream0)
        del arg56_1
        buf188 = reinterpret_tensor(buf178, (s14, 256), (256, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [out_11, x_15, out_10], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf187, buf181, buf186, buf188, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf186
        del buf187
        buf192 = reinterpret_tensor(buf181, (1, s14, 256), (256*s14, 256, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [out_8, tgt_3, out_11, tgt_4, tgt2_6, linear_18], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_view_21.run(buf101, buf144, buf188, arg57_1, arg58_1, buf192, s14, 256, stream=stream0)
        del arg57_1
        del arg58_1
        buf193 = reinterpret_tensor(buf98, (2048, 256), (256, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(arg59_1, buf193, 524288, stream=stream0)
        del arg59_1
        buf194 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(arg60_1, buf194, 2048, stream=stream0)
        del arg60_1
        buf195 = empty_strided_cuda((s14, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_8, tgt_3, out_11, tgt_4, tgt2_6, linear_18], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_mul_native_layer_norm_t_transpose_view_16.run(buf194, buf192, buf193, buf195, s14, 32*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf192
        del buf193
        del buf194
        buf196 = reinterpret_tensor(buf195, (1, s14, 2048), (2048*s14, 2048, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [linear_18, relu_1], Original ATen: [aten.view, aten.relu]
        triton_poi_fused_relu_view_17_xnumel = 2048*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_view_17.run(buf196, triton_poi_fused_relu_view_17_xnumel, stream=stream0)
        buf197 = empty_strided_cuda((256, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [tgt2_7], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(arg61_1, buf197, 524288, stream=stream0)
        del arg61_1
        buf198 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [tgt2_7], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg62_1, buf198, 256, stream=stream0)
        del arg62_1
        buf199 = empty_strided_cuda((s14, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [tgt2_7, linear_18, relu_1], Original ATen: [aten._to_copy, aten.view, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_view_18.run(buf198, buf196, buf197, buf199, s14, 2*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf196
        buf204 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        buf217 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        buf228 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_8, tgt_3, out_11, tgt_4, tgt2_7, tgt_5, tgt2_8, q_12, k_8, v_8], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_view_22.run(buf101, buf144, buf188, buf199, arg63_1, arg64_1, buf204, buf217, buf228, s14, 256, stream=stream0)
        del arg63_1
        del arg64_1
        buf205 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg65_1, buf205, 65536, stream=stream0)
        del arg65_1
        buf206 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [q_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg66_1, buf206, 256, stream=stream0)
        del arg66_1
        buf207 = empty_strided_cuda((s14, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_12], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf206, buf204, buf205, buf207, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf204
        buf208 = empty_strided_cuda((1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_12, x_16, q_13, float_9, reshape_27, xq__4], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf207, buf208, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [q_12, x_16, q_13, float_9, reshape_27, xq__4], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf209 = torch.ops.aten.view_as_complex.default(buf208)
        buf210 = buf209
        assert_size_stride(buf210, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf210, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_6], Original ATen: [aten.view]
        buf211 = torch.ops.aten.reshape.default(arg71_1, [1, 1, s14, 128])
        buf212 = buf211
        assert_size_stride(buf212, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.reshape.default')
        assert_alignment(buf212, 16, 'torch.ops.aten.reshape.default')
        # Topologically Sorted Source Nodes: [mul_9], Original ATen: [aten.mul]
        buf213 = torch.ops.aten.mul.Tensor(buf210, buf212)
        del buf209
        del buf210
        buf214 = buf213
        assert_size_stride(buf214, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf214, 16, 'torch.ops.aten.mul.Tensor')
        del buf213
        # Topologically Sorted Source Nodes: [view_as_real_8], Original ATen: [aten.view_as_real]
        buf215 = torch.ops.aten.view_as_real.default(buf214)
        buf216 = buf215
        assert_size_stride(buf216, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf216, 16, 'torch.ops.aten.view_as_real.default')
        buf218 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [k_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg67_1, buf218, 65536, stream=stream0)
        del arg67_1
        buf219 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [k_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg68_1, buf219, 256, stream=stream0)
        del arg68_1
        buf220 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [k_8], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf219, buf217, buf218, buf220, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf221 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [k_8, x_17, k_9, float_10, reshape_28, xk__4], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf220, buf221, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [k_8, x_17, k_9, float_10, reshape_28, xk__4], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf222 = torch.ops.aten.view_as_complex.default(buf221)
        buf223 = buf222
        assert_size_stride(buf223, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf223, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [mul_10], Original ATen: [aten.mul]
        buf224 = torch.ops.aten.mul.Tensor(buf223, buf212)
        del buf222
        del buf223
        buf225 = buf224
        assert_size_stride(buf225, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf225, 16, 'torch.ops.aten.mul.Tensor')
        del buf224
        # Topologically Sorted Source Nodes: [view_as_real_9], Original ATen: [aten.view_as_real]
        buf226 = torch.ops.aten.view_as_real.default(buf225)
        buf227 = buf226
        assert_size_stride(buf227, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf227, 16, 'torch.ops.aten.view_as_real.default')
        buf229 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [v_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg69_1, buf229, 65536, stream=stream0)
        del arg69_1
        buf230 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [v_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg70_1, buf230, 256, stream=stream0)
        del arg70_1
        buf231 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [v_8], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf230, buf228, buf229, buf231, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf232 = reinterpret_tensor(buf228, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [xq_out_4, type_as_8, k_8, x_17, k_9, xk_out_4, type_as_9, setitem_4, v_8, x_18, v_9, out_12], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf216, buf232, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf214
        del buf215
        del buf216
        buf233 = reinterpret_tensor(buf217, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [xq_out_4, type_as_8, k_8, x_17, k_9, xk_out_4, type_as_9, setitem_4, v_8, x_18, v_9, out_12], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf227, buf233, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf225
        del buf226
        del buf227
        # Topologically Sorted Source Nodes: [xq_out_4, type_as_8, k_8, x_17, k_9, xk_out_4, type_as_9, setitem_4, v_8, x_18, v_9, out_12], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        buf234 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf232, buf233, reinterpret_tensor(buf231, (1, 1, s14, 256), (256*s14, 256, 256, 1), 0), scale=0.0625)
        del buf231
        del buf232
        buf235 = buf234[0]
        assert_size_stride(buf235, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf235, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf234
        buf240 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg72_1, buf240, 65536, stream=stream0)
        del arg72_1
        buf241 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg73_1, buf241, 256, stream=stream0)
        del arg73_1
        buf242 = reinterpret_tensor(buf233, (s14, 256), (256, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [out_14, x_19, out_13], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf241, buf235, buf240, buf242, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf243 = buf101; del buf101  # reuse
        buf247 = reinterpret_tensor(buf235, (1, s14, 256), (256*s14, 256, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [out_8, tgt_3, out_11, tgt_4, tgt2_7, tgt_5, out_14, tgt_6, tgt2_9, q_15], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_view_23.run(buf243, buf144, buf188, buf199, buf242, arg74_1, arg75_1, buf247, s14, 256, stream=stream0)
        del arg74_1
        del arg75_1
        del buf144
        del buf188
        del buf199
        buf248 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [q_15], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg76_1, buf248, 65536, stream=stream0)
        del arg76_1
        buf249 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [q_15], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg77_1, buf249, 256, stream=stream0)
        del arg77_1
        buf250 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [tgt2_9, q_15], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf249, buf247, buf248, buf250, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf247
        buf251 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [q_15, x_20, q_16, float_11, reshape_33, xq__5], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf250, buf251, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [q_15, x_20, q_16, float_11, reshape_33, xq__5], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf252 = torch.ops.aten.view_as_complex.default(buf251)
        buf253 = buf252
        assert_size_stride(buf253, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf253, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_7], Original ATen: [aten.view]
        buf254 = torch.ops.aten.reshape.default(arg82_1, [1, 1, s14, 128])
        buf255 = buf254
        assert_size_stride(buf255, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.reshape.default')
        assert_alignment(buf255, 16, 'torch.ops.aten.reshape.default')
        # Topologically Sorted Source Nodes: [mul_11], Original ATen: [aten.mul]
        buf256 = torch.ops.aten.mul.Tensor(buf253, buf255)
        del buf251
        del buf252
        del buf253
        buf257 = buf256
        assert_size_stride(buf257, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf257, 16, 'torch.ops.aten.mul.Tensor')
        del buf256
        # Topologically Sorted Source Nodes: [view_as_real_10], Original ATen: [aten.view_as_real]
        buf258 = torch.ops.aten.view_as_real.default(buf257)
        buf259 = buf258
        assert_size_stride(buf259, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf259, 16, 'torch.ops.aten.view_as_real.default')
        buf260 = empty_strided_cuda((1, s98, 64), (64*s98, 64, 1), torch.bfloat16)
        buf273 = empty_strided_cuda((1, s98, 64), (64*s98, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, memory_pos, add_10, k_10, v_10], Original ATen: [aten.transpose, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_transpose_9_xnumel = 64*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_transpose_9.run(arg3_1, arg5_1, buf260, buf273, triton_poi_fused__to_copy_add_transpose_9_xnumel, stream=stream0)
        buf261 = empty_strided_cuda((256, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [k_10], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(arg78_1, buf261, 16384, stream=stream0)
        del arg78_1
        buf262 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [k_10], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg79_1, buf262, 256, stream=stream0)
        del arg79_1
        buf263 = empty_strided_cuda((s98, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, memory_pos, k_10, add_10], Original ATen: [aten.transpose, aten._to_copy, aten.add, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_t_transpose_view_11.run(buf262, buf260, buf261, buf263, s98, 4*((31 + s98) // 32), 1, 1, stream=stream0)
        del buf260
        buf264 = empty_strided_cuda((1, 1, s98 + ((-1)*s46), 128, 2), (((-256)*s46) + 256*s98, ((-256)*s46) + 256*s98, 256, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_10, x_21, k_11, getitem_339, float_12, reshape_34, xk__5], Original ATen: [aten.view, aten.transpose, aten.slice, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = ((-256)*s46) + 256*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf263, buf264, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [k_10, x_21, k_11, getitem_339, float_12, reshape_34, xk__5], Original ATen: [aten.view, aten.transpose, aten.slice, aten._to_copy, aten.view_as_complex]
        buf265 = torch.ops.aten.view_as_complex.default(buf264)
        buf266 = buf265
        assert_size_stride(buf266, (1, 1, s98 + ((-1)*s46), 128), (((-128)*s46) + 128*s98, ((-128)*s46) + 128*s98, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf266, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_8], Original ATen: [aten.repeat]
        buf267 = torch.ops.aten.repeat.default(buf255, [1, 1, (s98 + ((-1)*s46)) // s14, 1])
        buf268 = buf267
        assert_size_stride(buf268, (1, 1, s14*((s98 + ((-1)*s46)) // s14), 128), (128*max(1, s14*((s98 + ((-1)*s46)) // s14)), 128*max(1, s14*((s98 + ((-1)*s46)) // s14)), 128, 1), 'torch.ops.aten.repeat.default')
        assert_alignment(buf268, 16, 'torch.ops.aten.repeat.default')
        del buf267
        # Topologically Sorted Source Nodes: [mul_12], Original ATen: [aten.mul]
        buf269 = torch.ops.aten.mul.Tensor(buf266, buf268)
        del buf264
        del buf265
        del buf266
        del buf268
        buf270 = buf269
        assert_size_stride(buf270, (1, 1, s98 + ((-1)*s46), 128), (128*max(1, s98 + ((-1)*s46)), 128*max(1, s98 + ((-1)*s46)), 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf270, 16, 'torch.ops.aten.mul.Tensor')
        del buf269
        # Topologically Sorted Source Nodes: [view_as_real_11], Original ATen: [aten.view_as_real]
        buf271 = torch.ops.aten.view_as_real.default(buf270)
        buf272 = buf271
        assert_size_stride(buf272, (1, 1, s98 + ((-1)*s46), 128, 2), (256*max(1, s98 + ((-1)*s46)), 256*max(1, s98 + ((-1)*s46)), 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf272, 16, 'torch.ops.aten.view_as_real.default')
        buf274 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [v_10], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(arg80_1, buf274, 16384, stream=stream0)
        del arg80_1
        buf275 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [v_10], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg81_1, buf275, 256, stream=stream0)
        del arg81_1
        buf276 = empty_strided_cuda((s98, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, v_10], Original ATen: [aten.transpose, aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_t_transpose_view_11.run(buf275, buf273, buf274, buf276, s98, 4*((31 + s98) // 32), 1, 1, stream=stream0)
        buf277 = reinterpret_tensor(buf250, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [xq_out_5, type_as_10, setitem_5, k_10, x_21, k_11, xk_out_5, type_as_11, out_15, v_10, x_22, v_11], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf259, buf277, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf257
        del buf258
        del buf259
        buf278 = reinterpret_tensor(buf263, (1, 1, s98, 256), (256*s98, 256*s98, 256, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [xq_out_5, type_as_10, setitem_5, k_10, x_21, k_11, xk_out_5, type_as_11, out_15, v_10, x_22, v_11], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12_xnumel = 256*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12.run(buf278, buf272, s46, s98, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12_xnumel, stream=stream0)
        del buf270
        del buf271
        del buf272
        # Topologically Sorted Source Nodes: [xq_out_5, type_as_10, setitem_5, k_10, x_21, k_11, xk_out_5, type_as_11, out_15, v_10, x_22, v_11], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        buf279 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf277, buf278, reinterpret_tensor(buf276, (1, 1, s98, 256), (256*s98, 256, 256, 1), 0), scale=0.0625)
        del buf276
        del buf278
        buf280 = buf279[0]
        assert_size_stride(buf280, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf280, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf279
        buf285 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg83_1, buf285, 65536, stream=stream0)
        del arg83_1
        buf286 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg84_1, buf286, 256, stream=stream0)
        del arg84_1
        buf287 = reinterpret_tensor(buf277, (s14, 256), (256, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [out_17, x_23, out_16], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf286, buf280, buf285, buf287, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf291 = reinterpret_tensor(buf280, (1, s14, 256), (256*s14, 256, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [out_17, tgt_7, tgt2_10, linear_28], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_view_20.run(buf243, buf287, arg85_1, arg86_1, buf291, s14, 256, stream=stream0)
        del arg85_1
        del arg86_1
        buf292 = reinterpret_tensor(buf197, (2048, 256), (256, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(arg87_1, buf292, 524288, stream=stream0)
        del arg87_1
        buf293 = empty_strided_cuda((2048, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(arg88_1, buf293, 2048, stream=stream0)
        del arg88_1
        buf294 = empty_strided_cuda((s14, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_17, tgt_7, tgt2_10, linear_28], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_mul_native_layer_norm_t_transpose_view_16.run(buf293, buf291, buf292, buf294, s14, 32*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf291
        buf295 = reinterpret_tensor(buf294, (1, s14, 2048), (2048*s14, 2048, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [linear_28, relu_2], Original ATen: [aten.view, aten.relu]
        triton_poi_fused_relu_view_17_xnumel = 2048*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_view_17.run(buf295, triton_poi_fused_relu_view_17_xnumel, stream=stream0)
        buf296 = reinterpret_tensor(buf292, (256, 2048), (2048, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [tgt2_11], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(arg89_1, buf296, 524288, stream=stream0)
        del arg89_1
        buf297 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [tgt2_11], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg90_1, buf297, 256, stream=stream0)
        del arg90_1
        buf298 = empty_strided_cuda((s14, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [tgt2_11, linear_28, relu_2], Original ATen: [aten._to_copy, aten.view, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_view_18.run(buf297, buf295, buf296, buf298, s14, 2*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf295
        buf303 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        buf316 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        buf327 = empty_strided_cuda((1, s14, 256), (256*s14, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_17, tgt_7, tgt2_11, tgt_8, tgt2_12, q_18, k_12, v_12], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_view_24.run(buf243, buf287, buf298, arg91_1, arg92_1, buf303, buf316, buf327, s14, 256, stream=stream0)
        del arg91_1
        del arg92_1
        buf304 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [q_18], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg93_1, buf304, 65536, stream=stream0)
        del arg93_1
        buf305 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [q_18], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg94_1, buf305, 256, stream=stream0)
        del arg94_1
        buf306 = empty_strided_cuda((s14, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_18], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf305, buf303, buf304, buf306, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf303
        buf307 = empty_strided_cuda((1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_18, x_24, q_19, float_13, reshape_39, xq__6], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf306, buf307, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [q_18, x_24, q_19, float_13, reshape_39, xq__6], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf308 = torch.ops.aten.view_as_complex.default(buf307)
        buf309 = buf308
        assert_size_stride(buf309, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf309, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_9], Original ATen: [aten.view]
        buf310 = torch.ops.aten.reshape.default(arg99_1, [1, 1, s14, 128])
        buf311 = buf310
        assert_size_stride(buf311, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.reshape.default')
        assert_alignment(buf311, 16, 'torch.ops.aten.reshape.default')
        # Topologically Sorted Source Nodes: [mul_13], Original ATen: [aten.mul]
        buf312 = torch.ops.aten.mul.Tensor(buf309, buf311)
        del buf308
        del buf309
        buf313 = buf312
        assert_size_stride(buf313, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf313, 16, 'torch.ops.aten.mul.Tensor')
        del buf312
        # Topologically Sorted Source Nodes: [view_as_real_12], Original ATen: [aten.view_as_real]
        buf314 = torch.ops.aten.view_as_real.default(buf313)
        buf315 = buf314
        assert_size_stride(buf315, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf315, 16, 'torch.ops.aten.view_as_real.default')
        buf317 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [k_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg95_1, buf317, 65536, stream=stream0)
        del arg95_1
        buf318 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [k_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg96_1, buf318, 256, stream=stream0)
        del arg96_1
        buf319 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [k_12], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf318, buf316, buf317, buf319, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf320 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [k_12, x_25, k_13, float_14, reshape_40, xk__6], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf319, buf320, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [k_12, x_25, k_13, float_14, reshape_40, xk__6], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf321 = torch.ops.aten.view_as_complex.default(buf320)
        buf322 = buf321
        assert_size_stride(buf322, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf322, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [mul_14], Original ATen: [aten.mul]
        buf323 = torch.ops.aten.mul.Tensor(buf322, buf311)
        del buf321
        del buf322
        buf324 = buf323
        assert_size_stride(buf324, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf324, 16, 'torch.ops.aten.mul.Tensor')
        del buf323
        # Topologically Sorted Source Nodes: [view_as_real_13], Original ATen: [aten.view_as_real]
        buf325 = torch.ops.aten.view_as_real.default(buf324)
        buf326 = buf325
        assert_size_stride(buf326, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf326, 16, 'torch.ops.aten.view_as_real.default')
        buf328 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [v_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg97_1, buf328, 65536, stream=stream0)
        del arg97_1
        buf329 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [v_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg98_1, buf329, 256, stream=stream0)
        del arg98_1
        buf330 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [v_12], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf329, buf327, buf328, buf330, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf331 = reinterpret_tensor(buf327, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [xq_out_6, type_as_12, k_12, x_25, k_13, xk_out_6, type_as_13, setitem_6, v_12, x_26, v_13, out_18], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf315, buf331, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf313
        del buf314
        del buf315
        buf332 = reinterpret_tensor(buf316, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [xq_out_6, type_as_12, k_12, x_25, k_13, xk_out_6, type_as_13, setitem_6, v_12, x_26, v_13, out_18], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf326, buf332, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf324
        del buf325
        del buf326
        # Topologically Sorted Source Nodes: [xq_out_6, type_as_12, k_12, x_25, k_13, xk_out_6, type_as_13, setitem_6, v_12, x_26, v_13, out_18], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.copy, aten._scaled_dot_product_flash_attention]
        buf333 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf331, buf332, reinterpret_tensor(buf330, (1, 1, s14, 256), (256*s14, 256, 256, 1), 0), scale=0.0625)
        del buf330
        buf334 = buf333[0]
        assert_size_stride(buf334, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf334, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf333
        buf339 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg100_1, buf339, 65536, stream=stream0)
        del arg100_1
        buf340 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg101_1, buf340, 256, stream=stream0)
        del arg101_1
        buf341 = reinterpret_tensor(buf332, (s14, 256), (256, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [out_20, x_27, out_19], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf340, buf334, buf339, buf341, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        buf345 = reinterpret_tensor(buf334, (1, s14, 256), (256*s14, 256, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [out_17, tgt_7, tgt2_11, tgt_8, out_20, tgt_9, tgt2_13, q_21], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_view_25.run(buf243, buf287, buf298, buf341, arg102_1, arg103_1, buf345, s14, 256, stream=stream0)
        del arg102_1
        del arg103_1
        buf346 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [q_21], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg104_1, buf346, 65536, stream=stream0)
        del arg104_1
        buf347 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [q_21], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg105_1, buf347, 256, stream=stream0)
        del arg105_1
        buf348 = reinterpret_tensor(buf331, (s14, 256), (256, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [out_17, tgt_7, tgt2_11, tgt_8, out_20, tgt_9, tgt2_13, q_21], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf347, buf345, buf346, buf348, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf345
        buf349 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [q_21, x_28, q_22, float_15, reshape_45, xq__7], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf348, buf349, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [q_21, x_28, q_22, float_15, reshape_45, xq__7], Original ATen: [aten.view, aten.transpose, aten._to_copy, aten.view_as_complex]
        buf350 = torch.ops.aten.view_as_complex.default(buf349)
        buf351 = buf350
        assert_size_stride(buf351, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf351, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_10], Original ATen: [aten.view]
        buf352 = torch.ops.aten.reshape.default(arg110_1, [1, 1, s14, 128])
        buf353 = buf352
        assert_size_stride(buf353, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.reshape.default')
        assert_alignment(buf353, 16, 'torch.ops.aten.reshape.default')
        # Topologically Sorted Source Nodes: [mul_15], Original ATen: [aten.mul]
        buf354 = torch.ops.aten.mul.Tensor(buf351, buf353)
        del buf349
        del buf350
        del buf351
        buf355 = buf354
        assert_size_stride(buf355, (1, 1, s14, 128), (128*s14, 128*s14, 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf355, 16, 'torch.ops.aten.mul.Tensor')
        del buf354
        # Topologically Sorted Source Nodes: [view_as_real_14], Original ATen: [aten.view_as_real]
        buf356 = torch.ops.aten.view_as_real.default(buf355)
        buf357 = buf356
        assert_size_stride(buf357, (1, 1, s14, 128, 2), (256*s14, 256*s14, 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf357, 16, 'torch.ops.aten.view_as_real.default')
        buf358 = buf273; del buf273  # reuse
        buf371 = empty_strided_cuda((1, s98, 64), (64*s98, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, memory_pos, add_14, k_14, v_14], Original ATen: [aten.transpose, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_transpose_9_xnumel = 64*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_transpose_9.run(arg3_1, arg5_1, buf358, buf371, triton_poi_fused__to_copy_add_transpose_9_xnumel, stream=stream0)
        del arg3_1
        del arg5_1
        buf359 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [k_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(arg106_1, buf359, 16384, stream=stream0)
        del arg106_1
        buf360 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [k_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg107_1, buf360, 256, stream=stream0)
        del arg107_1
        buf361 = empty_strided_cuda((s98, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, memory_pos, k_14, add_14], Original ATen: [aten.transpose, aten._to_copy, aten.add, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_t_transpose_view_11.run(buf360, buf358, buf359, buf361, s98, 4*((31 + s98) // 32), 1, 1, stream=stream0)
        del buf358
        buf362 = empty_strided_cuda((1, 1, s98 + ((-1)*s46), 128, 2), (((-256)*s46) + 256*s98, ((-256)*s46) + 256*s98, 256, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_14, x_29, k_15, getitem_463, float_16, reshape_46, xk__7], Original ATen: [aten.view, aten.transpose, aten.slice, aten._to_copy, aten.view_as_complex]
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel = ((-256)*s46) + 256*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_transpose_view_view_as_complex_6.run(buf361, buf362, triton_poi_fused__to_copy_transpose_view_view_as_complex_6_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [k_14, x_29, k_15, getitem_463, float_16, reshape_46, xk__7], Original ATen: [aten.view, aten.transpose, aten.slice, aten._to_copy, aten.view_as_complex]
        buf363 = torch.ops.aten.view_as_complex.default(buf362)
        buf364 = buf363
        assert_size_stride(buf364, (1, 1, s98 + ((-1)*s46), 128), (((-128)*s46) + 128*s98, ((-128)*s46) + 128*s98, 128, 1), 'torch.ops.aten.view_as_complex.default')
        assert_alignment(buf364, 16, 'torch.ops.aten.view_as_complex.default')
        # Topologically Sorted Source Nodes: [freqs_cis_11], Original ATen: [aten.repeat]
        buf365 = torch.ops.aten.repeat.default(buf353, [1, 1, (s98 + ((-1)*s46)) // s14, 1])
        buf366 = buf365
        assert_size_stride(buf366, (1, 1, s14*((s98 + ((-1)*s46)) // s14), 128), (128*max(1, s14*((s98 + ((-1)*s46)) // s14)), 128*max(1, s14*((s98 + ((-1)*s46)) // s14)), 128, 1), 'torch.ops.aten.repeat.default')
        assert_alignment(buf366, 16, 'torch.ops.aten.repeat.default')
        del buf365
        # Topologically Sorted Source Nodes: [mul_16], Original ATen: [aten.mul]
        buf367 = torch.ops.aten.mul.Tensor(buf364, buf366)
        del buf362
        del buf363
        del buf364
        del buf366
        buf368 = buf367
        assert_size_stride(buf368, (1, 1, s98 + ((-1)*s46), 128), (128*max(1, s98 + ((-1)*s46)), 128*max(1, s98 + ((-1)*s46)), 128, 1), 'torch.ops.aten.mul.Tensor')
        assert_alignment(buf368, 16, 'torch.ops.aten.mul.Tensor')
        del buf367
        # Topologically Sorted Source Nodes: [view_as_real_15], Original ATen: [aten.view_as_real]
        buf369 = torch.ops.aten.view_as_real.default(buf368)
        buf370 = buf369
        assert_size_stride(buf370, (1, 1, s98 + ((-1)*s46), 128, 2), (256*max(1, s98 + ((-1)*s46)), 256*max(1, s98 + ((-1)*s46)), 256, 2, 1), 'torch.ops.aten.view_as_real.default')
        assert_alignment(buf370, 16, 'torch.ops.aten.view_as_real.default')
        buf372 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [v_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(arg108_1, buf372, 16384, stream=stream0)
        del arg108_1
        buf373 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [v_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg109_1, buf373, 256, stream=stream0)
        del arg109_1
        buf374 = empty_strided_cuda((s98, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [memory, v_14], Original ATen: [aten.transpose, aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_t_transpose_view_11.run(buf373, buf371, buf372, buf374, s98, 4*((31 + s98) // 32), 1, 1, stream=stream0)
        del buf371
        del buf372
        buf375 = reinterpret_tensor(buf348, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [xq_out_7, type_as_14, setitem_7, k_14, x_29, k_15, xk_out_7, type_as_15, out_21, v_14, x_30, v_15], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel = 256*s14
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7.run(buf357, buf375, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_transpose_view_7_xnumel, stream=stream0)
        del buf355
        del buf356
        del buf357
        buf376 = reinterpret_tensor(buf361, (1, 1, s98, 256), (256*s98, 256*s98, 256, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [xq_out_7, type_as_14, setitem_7, k_14, x_29, k_15, xk_out_7, type_as_15, out_21, v_14, x_30, v_15], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12_xnumel = 256*s98
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12.run(buf376, buf370, s46, s98, triton_poi_fused__scaled_dot_product_flash_attention__to_copy_copy_slice_transpose_view_12_xnumel, stream=stream0)
        del buf368
        del buf369
        del buf370
        # Topologically Sorted Source Nodes: [xq_out_7, type_as_14, setitem_7, k_14, x_29, k_15, xk_out_7, type_as_15, out_21, v_14, x_30, v_15], Original ATen: [aten.view, aten._to_copy, aten.transpose, aten.slice, aten.copy, aten._scaled_dot_product_flash_attention]
        buf377 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf375, buf376, reinterpret_tensor(buf374, (1, 1, s98, 256), (256*s98, 256, 256, 1), 0), scale=0.0625)
        del buf374
        del buf376
        buf378 = buf377[0]
        assert_size_stride(buf378, (1, 1, s14, 256), (256*s14, 256*s14, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf378, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf377
        buf383 = buf346; del buf346  # reuse
        # Topologically Sorted Source Nodes: [out_23], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(arg111_1, buf383, 65536, stream=stream0)
        del arg111_1
        buf384 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [out_23], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg112_1, buf384, 256, stream=stream0)
        del arg112_1
        buf385 = reinterpret_tensor(buf375, (s14, 256), (256, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [out_23, x_31, out_22], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_5.run(buf384, buf378, buf383, buf385, s14, 4*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf383
        buf386 = buf243; del buf243  # reuse
        buf390 = reinterpret_tensor(buf378, (1, s14, 256), (256*s14, 256, 1), 0); del buf378  # reuse
        # Topologically Sorted Source Nodes: [out_17, tgt_7, tgt2_11, tgt_8, out_20, tgt_9, out_23, tgt_10, tgt2_14, linear_38], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_view_23.run(buf386, buf287, buf298, buf341, buf385, arg113_1, arg114_1, buf390, s14, 256, stream=stream0)
        del arg113_1
        del arg114_1
        del buf287
        del buf298
        del buf341
        del buf385
        buf391 = reinterpret_tensor(buf296, (2048, 256), (256, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(arg115_1, buf391, 524288, stream=stream0)
        del arg115_1
        buf392 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_15.run(arg116_1, buf392, 2048, stream=stream0)
        del arg116_1
        buf394 = empty_strided_cuda((1, s14, 2048), (2048*s14, 2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [tgt2_14, linear_38, relu_3], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_native_layer_norm_relu_t_view_26.run(buf392, buf390, buf391, buf394, s14, 32*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf392
        buf395 = reinterpret_tensor(buf391, (256, 2048), (2048, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [tgt2_15], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(arg117_1, buf395, 524288, stream=stream0)
        del arg117_1
        buf396 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [tgt2_15], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg118_1, buf396, 256, stream=stream0)
        del arg118_1
        buf397 = reinterpret_tensor(buf390, (s14, 256), (256, 1), 0); del buf390  # reuse
        # Topologically Sorted Source Nodes: [tgt2_15, linear_38, relu_3], Original ATen: [aten._to_copy, aten.view, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_view_18.run(buf396, buf394, buf395, buf397, s14, 2*((127 + s14) // 128), 1, 1, stream=stream0)
        del buf394
        del buf395
        del buf396
        buf401 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [tgt2_15, tgt_11, normed_output], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_27.run(buf401, buf397, arg119_1, arg120_1, s14, 256, stream=stream0)
        del arg119_1
        del arg120_1
        del buf397
    return (buf401, arg14_1, arg26_1, arg43_1, arg54_1, arg71_1, arg82_1, arg99_1, arg110_1, )


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1 = args
        args.clear()
        s14 = arg0_1
        s98 = arg2_1
        s46 = arg17_1
        partition0_args = [arg1_1, arg4_1, arg6_1, arg7_1, arg8_1, arg9_1, arg14_1, arg10_1, arg11_1, arg12_1, arg13_1, arg15_1, arg16_1, arg18_1, arg19_1, arg20_1, arg21_1, arg26_1, arg3_1, arg5_1, arg22_1, arg23_1, arg24_1, arg25_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg43_1, arg39_1, arg40_1, arg41_1, arg42_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg54_1, arg50_1, arg51_1, arg52_1, arg53_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg71_1, arg67_1, arg68_1, arg69_1, arg70_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg82_1, arg78_1, arg79_1, arg80_1, arg81_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg99_1, arg95_1, arg96_1, arg97_1, arg98_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg110_1, arg106_1, arg107_1, arg108_1, arg109_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, s14, s46, s98]
        del arg1_1, arg4_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg15_1, arg16_1, arg18_1, arg19_1, arg20_1, arg21_1, arg3_1, arg5_1, arg22_1, arg23_1, arg24_1, arg25_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1
        (buf401, arg14_1, arg26_1, arg43_1, arg54_1, arg71_1, arg82_1, arg99_1, arg110_1) = self.partitions[0](partition0_args)
        del partition0_args
        return (reinterpret_tensor(buf401, (s14, 1, 256), (256, 256*s14, 1), 0), arg26_1, arg14_1, arg54_1, arg43_1, arg82_1, arg71_1, arg110_1, arg99_1, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 4096
    arg1_1 = rand_strided((4096, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = 4100
    arg3_1 = rand_strided((4100, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((4096, 1, 256), (1, 1048576, 4096), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((4100, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.complex64)
    arg15_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = 4
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.complex64)
    arg27_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.complex64)
    arg44_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.complex64)
    arg55_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.complex64)
    arg72_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.complex64)
    arg83_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.complex64)
    arg100_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((4096, 128), (128, 1), device='cuda:0', dtype=torch.complex64)
    arg111_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
