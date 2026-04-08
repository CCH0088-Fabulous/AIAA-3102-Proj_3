# AOT ID: ['1_inference']
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



# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/4h/c4hhhoiyr3buv3ikiwjfa6rzrjv26myjvdmypjfahatxf2jzbdtu.py
# Topologically Sorted Source Nodes: [points, padding_point, points_1, setitem, getitem, truediv, setitem_1, truediv_1, mul, coords_1, coords_2], Original ATen: [aten.add, aten.zeros, aten.cat, aten.select, aten.div, aten.copy, aten.mul, aten.sub, aten._to_copy]
# Source node to ATen node mapping:
#   coords_1 => sub
#   coords_2 => convert_element_type_1
#   getitem => select
#   mul => mul
#   padding_point => full_default
#   points => add
#   points_1 => cat
#   setitem => copy, select_1
#   setitem_1 => copy_1, select_6
#   truediv => div
#   truediv_1 => div_1, select_4
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 1, 2][2, 2, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %add : Tensor "f32[1, 1, 2][2, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 0.5), kwargs = {})
#   %full_default : Tensor "f32[1, 1, 2][2, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 2], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cat : Tensor "f32[1, 2, 2][4, 2, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add, %full_default], 1), kwargs = {})
#   %select_1 : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat, 2, 0), kwargs = {})
#   %select : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat, 2, 0), kwargs = {})
#   %div : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%select, 1024), kwargs = {})
#   %copy : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_1, %div), kwargs = {})
#   %select_scatter_default : Tensor "f32[1, 2, 2][4, 2, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%cat, %copy, 2, 0), kwargs = {})
#   %select_6 : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_scatter_default, 2, 1), kwargs = {})
#   %select_4 : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_scatter_default, 2, 1), kwargs = {})
#   %div_1 : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%select_4, 1024), kwargs = {})
#   %copy_1 : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_6, %div_1), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[1, 2, 2][4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %copy_1, 2, 1), kwargs = {})
#   %mul : Tensor "f32[1, 2, 2][4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_scatter_default_1, 2), kwargs = {})
#   %sub : Tensor "f32[1, 2, 2][4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 2, 2][4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_add_cat_copy_div_mul_select_sub_zeros_0 = async_compile.triton('triton_poi_fused__to_copy_add_cat_copy_div_mul_select_sub_zeros_0', '''
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
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/6t/c6tf6lww5nil3eur3uot65dalc3zb4au3oemtmwokhqyqiic4hui.py
# Topologically Sorted Source Nodes: [coords_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   coords_2 => convert_element_type
# Graph fragment:
#   %arg3_1 : Tensor "f32[2, 128][128, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type : Tensor "bf16[2, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/y7/cy7gcp2bvzjdvver65aoyqwfnygyhfd6vjuslhoefxdclqrtz4tm.py
# Topologically Sorted Source Nodes: [points, padding_point, points_1, setitem, getitem, truediv, setitem_1, truediv_1, mul, coords_1, coords_2], Original ATen: [aten.add, aten.zeros, aten.cat, aten.select, aten.div, aten.copy, aten.mul, aten.sub, aten._to_copy, aten.view, aten.mm]
# Source node to ATen node mapping:
#   coords_1 => sub
#   coords_2 => convert_element_type, convert_element_type_1, mm, view
#   getitem => select
#   mul => mul
#   padding_point => full_default
#   points => add
#   points_1 => cat
#   setitem => copy, select_1
#   setitem_1 => copy_1, select_6
#   truediv => div
#   truediv_1 => div_1, select_4
# Graph fragment:
#   %convert_element_type_1 : Tensor "bf16[1, 2, 2][4, 2, 1]cuda:0" = PlaceHolder[target=convert_element_type_1]
#   %convert_element_type : Tensor "bf16[2, 128][128, 1]cuda:0" = PlaceHolder[target=convert_element_type]
#   %add : Tensor "f32[1, 1, 2][2, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 0.5), kwargs = {})
#   %full_default : Tensor "f32[1, 1, 2][2, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 2], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cat : Tensor "f32[1, 2, 2][4, 2, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add, %full_default], 1), kwargs = {})
#   %select_1 : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat, 2, 0), kwargs = {})
#   %select : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat, 2, 0), kwargs = {})
#   %div : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%select, 1024), kwargs = {})
#   %copy : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_1, %div), kwargs = {})
#   %select_scatter_default : Tensor "f32[1, 2, 2][4, 2, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%cat, %copy, 2, 0), kwargs = {})
#   %select_6 : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_scatter_default, 2, 1), kwargs = {})
#   %select_4 : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_scatter_default, 2, 1), kwargs = {})
#   %div_1 : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%select_4, 1024), kwargs = {})
#   %copy_1 : Tensor "f32[1, 2][4, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_6, %div_1), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[1, 2, 2][4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %copy_1, 2, 1), kwargs = {})
#   %mul : Tensor "f32[1, 2, 2][4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_scatter_default_1, 2), kwargs = {})
#   %sub : Tensor "f32[1, 2, 2][4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 2, 2][4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.bfloat16), kwargs = {})
#   %view : Tensor "bf16[2, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_1, [2, 2]), kwargs = {})
#   %convert_element_type : Tensor "bf16[2, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.bfloat16), kwargs = {})
#   %mm : Tensor "bf16[2, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view, %convert_element_type), kwargs = {})
#   return %mm
triton_tem_fused__to_copy_add_cat_copy_div_mm_mul_select_sub_view_zeros_2 = async_compile.triton('triton_tem_fused__to_copy_add_cat_copy_div_mm_mul_select_sub_view_zeros_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=8,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_cat_copy_div_mm_mul_select_sub_view_zeros_2', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_add_cat_copy_div_mm_mul_select_sub_view_zeros_2(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 2
    N = 128
    K = 2
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2
    stride_ak = 1
    stride_bk = 128
    stride_bn = 1

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

        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 2*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 128*idx_m
        b = tl.load(B + (xindex), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 128*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/dq/cdq7o3jciwybj6xvf7r677vm4but7pqa37dwckzmsljb76ehmazh.py
# Topologically Sorted Source Nodes: [padding_label, labels, eq_4, unsqueeze_4, eq_3, unsqueeze_3, eq_2, unsqueeze_2, eq_1, unsqueeze_1, eq, unsqueeze, zeros_like, add_1, coords_2, coords_3, sin, cos, point_embedding, point_embedding_1, add_2, point_embedding_2, add_3, point_embedding_3, add_4, point_embedding_4, add_5, point_embedding_5], Original ATen: [aten.neg, aten.cat, aten.eq, aten.unsqueeze, aten.zeros_like, aten.add, aten._unsafe_view, aten.mul, aten.sin, aten.cos, aten.where]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   coords_2 => view_1
#   coords_3 => mul_1
#   cos => cos
#   eq => eq
#   eq_1 => eq_1
#   eq_2 => eq_2
#   eq_3 => eq_3
#   eq_4 => eq_4
#   labels => cat_1
#   padding_label => full_default_1
#   point_embedding => cat_2
#   point_embedding_1 => where
#   point_embedding_2 => where_1
#   point_embedding_3 => where_2
#   point_embedding_4 => where_3
#   point_embedding_5 => where_4
#   sin => sin
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
#   unsqueeze_4 => unsqueeze_4
#   zeros_like => full_default_2
# Graph fragment:
#   %arg2_1 : Tensor "i32[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg4_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %mm : Tensor "bf16[2, 128][128, 1]cuda:0" = PlaceHolder[target=mm]
#   %where : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0" = PlaceHolder[target=where]
#   %arg1_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg5_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %where_3 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0" = PlaceHolder[target=where_3]
#   %arg7_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %full_default_1 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1], -1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cat_1 : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.cat.default](args = ([%arg2_1, %full_default_1], 1), kwargs = {})
#   %eq_4 : Tensor "b8[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%cat_1, 3), kwargs = {})
#   %unsqueeze_4 : Tensor "b8[1, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq_4, -1), kwargs = {})
#   %eq_3 : Tensor "b8[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%cat_1, 2), kwargs = {})
#   %unsqueeze_3 : Tensor "b8[1, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq_3, -1), kwargs = {})
#   %eq_2 : Tensor "b8[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%cat_1, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "b8[1, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq_2, -1), kwargs = {})
#   %eq_1 : Tensor "b8[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%cat_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "b8[1, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq_1, -1), kwargs = {})
#   %eq : Tensor "b8[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%cat_1, -1), kwargs = {})
#   %unsqueeze : Tensor "b8[1, 2, 1][2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %full_default_2 : Tensor "bf16[1, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 2, 256], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_1 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_2, %arg4_1), kwargs = {})
#   %view_1 : Tensor "bf16[1, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 2, 128]), kwargs = {})
#   %mul_1 : Tensor "bf16[1, 2, 128][256, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, 6.283185307179586), kwargs = {})
#   %sin : Tensor "bf16[1, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_1,), kwargs = {})
#   %cos : Tensor "bf16[1, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_1,), kwargs = {})
#   %cat_2 : Tensor "bf16[1, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sin, %cos], -1), kwargs = {})
#   %where : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze, %add_1, %cat_2), kwargs = {})
#   %add_2 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %arg1_1), kwargs = {})
#   %where_1 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_1, %add_2, %where), kwargs = {})
#   %add_3 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_1, %arg5_1), kwargs = {})
#   %where_2 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_2, %add_3, %where_1), kwargs = {})
#   %add_4 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_2, %arg6_1), kwargs = {})
#   %where_3 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_3, %add_4, %where_2), kwargs = {})
#   %add_5 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_3, %arg7_1), kwargs = {})
#   %where_4 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_4, %add_5, %where_3), kwargs = {})
#   return %where,%where_3,%where_4
triton_poi_fused__unsafe_view_add_cat_cos_eq_mul_neg_sin_unsqueeze_where_zeros_like_3 = async_compile.triton('triton_poi_fused__unsafe_view_add_cat_cos_eq_mul_neg_sin_unsqueeze_where_zeros_like_3', '''
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
''', device_str='cuda')

def partition_0(args):
    arg0_1, arg3_1, arg2_1, arg4_1, arg1_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 2), (2, 2, 1))
    assert_size_stride(arg3_1, (2, 128), (128, 1))
    assert_size_stride(arg2_1, (1, 1), (1, 1))
    assert_size_stride(arg4_1, (1, 256), (256, 1))
    assert_size_stride(arg1_1, (1, 256), (256, 1))
    assert_size_stride(arg5_1, (1, 256), (256, 1))
    assert_size_stride(arg6_1, (1, 256), (256, 1))
    assert_size_stride(arg7_1, (1, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 2, 2), (4, 2, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [points, padding_point, points_1, setitem, getitem, truediv, setitem_1, truediv_1, mul, coords_1, coords_2], Original ATen: [aten.add, aten.zeros, aten.cat, aten.select, aten.div, aten.copy, aten.mul, aten.sub, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_copy_div_mul_select_sub_zeros_0.run(arg0_1, buf0, 4, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((2, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [coords_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg3_1, buf1, 256, stream=stream0)
        del arg3_1
        buf2 = empty_strided_cuda((2, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [points, padding_point, points_1, setitem, getitem, truediv, setitem_1, truediv_1, mul, coords_1, coords_2], Original ATen: [aten.add, aten.zeros, aten.cat, aten.select, aten.div, aten.copy, aten.mul, aten.sub, aten._to_copy, aten.view, aten.mm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_cat_copy_div_mm_mul_select_sub_view_zeros_2.run(buf0, buf1, buf2, 1, 1, 1, stream=stream0)
        del buf0
        del buf1
        buf3 = empty_strided_cuda((1, 2, 256), (512, 256, 1), torch.float32)
        buf4 = buf3; del buf3  # reuse
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [padding_label, labels, eq_4, unsqueeze_4, eq_3, unsqueeze_3, eq_2, unsqueeze_2, eq_1, unsqueeze_1, eq, unsqueeze, zeros_like, add_1, coords_2, coords_3, sin, cos, point_embedding, point_embedding_1, add_2, point_embedding_2, add_3, point_embedding_3, add_4, point_embedding_4, add_5, point_embedding_5], Original ATen: [aten.neg, aten.cat, aten.eq, aten.unsqueeze, aten.zeros_like, aten.add, aten._unsafe_view, aten.mul, aten.sin, aten.cos, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_cat_cos_eq_mul_neg_sin_unsqueeze_where_zeros_like_3.run(buf5, arg2_1, arg4_1, buf2, arg1_1, arg5_1, arg6_1, arg7_1, 512, stream=stream0)
        del arg1_1
        del arg2_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del buf2
    return (buf5, )


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
        args.clear()
        partition0_args = [arg0_1, arg3_1, arg2_1, arg4_1, arg1_1, arg5_1, arg6_1, arg7_1]
        del arg0_1, arg3_1, arg2_1, arg4_1, arg1_1, arg5_1, arg6_1, arg7_1
        (buf5,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf5, reinterpret_tensor(arg8_1, (1, 256, 64, 64), (256, 1, 0, 0), 0), )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 1, 2), (2, 2, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int32)
    arg3_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
