# AOT ID: ['2_inference']
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



# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/mo/cmogdkyftoh6ous5o2szff6lbgrrhzkjmqed544eixqiah4ehnw6.py
# Topologically Sorted Source Nodes: [output_tokens, unsqueeze, output_tokens_1, tokens, q, k, v], Original ATen: [aten.cat, aten.unsqueeze, aten.expand, aten._to_copy]
# Source node to ATen node mapping:
#   k => convert_element_type_8
#   output_tokens => cat
#   output_tokens_1 => expand
#   q => convert_element_type_2
#   tokens => cat_1
#   unsqueeze => unsqueeze
#   v => convert_element_type_14
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %cat : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg0_1, %arg1_1, %arg2_1],), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 6, 256][1536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 0), kwargs = {})
#   %expand : Tensor "f32[1, 6, 256][1536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1]), kwargs = {})
#   %cat_1 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.cat.default](args = ([%expand, %arg3_1], 1), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2,%convert_element_type_8,%convert_element_type_14
triton_poi_fused__to_copy_cat_expand_unsqueeze_0 = async_compile.triton('triton_poi_fused__to_copy_cat_expand_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_expand_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 3, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_expand_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 256
    x0 = (xindex % 256)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x0), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp4
    tmp17 = tl.load(in_ptr1 + (x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp5 >= tmp13
    tmp19 = tl.full([1], 6, tl.int64)
    tmp20 = tmp5 < tmp19
    tmp21 = tmp18 & tmp4
    tmp22 = tl.load(in_ptr2 + (x0 + 256*((-2) + (x1))), tmp21 & xmask, other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.where(tmp9, tmp11, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 8, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tl.load(in_ptr3 + (x0 + 256*((-6) + x1)), tmp27 & xmask, other=0.0)
    tmp31 = tl.where(tmp4, tmp26, tmp30)
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp32, xmask)
    tl.store(out_ptr1 + (x2), tmp32, xmask)
    tl.store(out_ptr2 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/hc/chc5nsxvriksl5tjtobevbnild3baurm4lljhctnswzuyovvizdb.py
# Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   q => convert_element_type_1
# Graph fragment:
#   %arg7_1 : Tensor "f32[256, 256][256, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_1 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/aq/caqql2izgdha36wdggcdweem3z3txvly7d4mg7uelzdvj4gyky7l.py
# Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   q => convert_element_type
# Graph fragment:
#   %arg8_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convert_element_type : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/u4/cu4uryvanp5l7atcflotariyjsknbomk2arhcyrbuepbbte2epi2.py
# Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   q => addmm, convert_element_type, convert_element_type_1, permute_2, view_3
# Graph fragment:
#   %convert_element_type : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type]
#   %convert_element_type_2 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=convert_element_type_2]
#   %convert_element_type_1 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_1]
#   %convert_element_type : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg8_1, torch.bfloat16), kwargs = {})
#   %view_3 : Tensor "bf16[8, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_2, [8, 256]), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.bfloat16), kwargs = {})
#   %permute_2 : Tensor "bf16[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1, [1, 0]), kwargs = {})
#   %addmm : Tensor "bf16[8, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type, %view_3, %permute_2), kwargs = {})
#   return %addmm
triton_tem_fused__to_copy_addmm_t_view_3 = async_compile.triton('triton_tem_fused__to_copy_addmm_t_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_t_view_3', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_t_view_3(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8
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


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/i3/ci3u3ujmr3hexztw3a742xqlxducx5pjfnpr4d34fz5b2agb35by.py
# Topologically Sorted Source Nodes: [output_tokens, unsqueeze, output_tokens_1, tokens, out_2, queries, q_2, q_3], Original ATen: [aten.cat, aten.unsqueeze, aten.expand, aten.view, aten._to_copy, aten.native_layer_norm, aten.add]
# Source node to ATen node mapping:
#   out_2 => view_14
#   output_tokens => cat
#   output_tokens_1 => expand
#   q_2 => add_3
#   q_3 => convert_element_type_26
#   queries => add_1, add_2, convert_element_type_23, mul, mul_1, rsqrt, sub, var_mean
#   tokens => cat_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %addmm_3 : Tensor "bf16[8, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %getitem_10 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=getitem_10]
#   %buf22 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=buf22]
#   %arg15_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %arg16_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %add_2 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_2]
#   %arg0_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %add_3 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_3]
#   %cat : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg0_1, %arg1_1, %arg2_1],), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 6, 256][1536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 0), kwargs = {})
#   %expand : Tensor "f32[1, 6, 256][1536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1]), kwargs = {})
#   %cat_1 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.cat.default](args = ([%expand, %arg3_1], 1), kwargs = {})
#   %view_14 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 8, 256]), kwargs = {})
#   %convert_element_type_23 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_14, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_23, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_23, %getitem_10), kwargs = {})
#   %add_1 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg15_1), kwargs = {})
#   %add_2 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg16_1), kwargs = {})
#   %add_3 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %cat_1), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   return %getitem_10,%buf22,%add_2,%add_3,%convert_element_type_26
triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_4 = async_compile.triton('triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_4', '''
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
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/i7/ci766f4mz4klmzipd5ukd3vwe3m624r6jijzmoj2a5l6lefzfpuv.py
# Topologically Sorted Source Nodes: [q_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   q_3 => convert_element_type_25
# Graph fragment:
#   %arg17_1 : Tensor "f32[128, 256][256, 1]cuda:0" = PlaceHolder[target=arg17_1]
#   %convert_element_type_25 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_25
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/km/ckmrxdfm57vsy7lvx5flqolxcalycc2xop3qodr76kmqamedyhlp.py
# Topologically Sorted Source Nodes: [q_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   q_3 => convert_element_type_24
# Graph fragment:
#   %arg18_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %convert_element_type_24 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg18_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_24
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/6i/c6idl4peqo27rcomd2cfqgwau7eorqgaauwb3o42fby72vl57rjd.py
# Topologically Sorted Source Nodes: [q_3], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   q_3 => addmm_4, convert_element_type_24, convert_element_type_25, convert_element_type_26, permute_10, view_15
# Graph fragment:
#   %convert_element_type_24 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=convert_element_type_24]
#   %convert_element_type_26 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=convert_element_type_26]
#   %convert_element_type_25 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_25]
#   %convert_element_type_24 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg18_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   %view_15 : Tensor "bf16[8, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_26, [8, 256]), kwargs = {})
#   %convert_element_type_25 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.bfloat16), kwargs = {})
#   %permute_10 : Tensor "bf16[256, 128][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_25, [1, 0]), kwargs = {})
#   %addmm_4 : Tensor "bf16[8, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_24, %view_15, %permute_10), kwargs = {})
#   return %addmm_4
triton_tem_fused__to_copy_addmm_t_view_7 = async_compile.triton('triton_tem_fused__to_copy_addmm_t_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=2,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_t_view_7', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_t_view_7(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8
    N = 128
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
        xindex = idx_n + 128*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 128*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/3t/c3tcswii25iwzrjbhkx7kubl3zqojczb6msivcfclxhbhnxfezna.py
# Topologically Sorted Source Nodes: [src, flatten, image_embedding, pos_src, flatten_1, image_pe, k_2, k_3, v_2, k_5, q_6], Original ATen: [aten.add, aten.view, aten.permute, aten.unsqueeze, aten.clone, aten._to_copy]
# Source node to ATen node mapping:
#   flatten => view_1
#   flatten_1 => view_2
#   image_embedding => permute
#   image_pe => permute_1
#   k_2 => add_4
#   k_3 => convert_element_type_32
#   k_5 => add_12
#   pos_src => clone, unsqueeze_1, view
#   q_6 => convert_element_type_60
#   src => add
#   v_2 => convert_element_type_38
# Graph fragment:
#   %arg4_1 : Tensor "f32[1, 256, 64, 64][256, 1, 16384, 256]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "bf16[1, 256, 64, 64][256, 1, 16384, 256]cuda:0" = PlaceHolder[target=arg6_1]
#   %add : Tensor "f32[1, 256, 64, 64][256, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %arg5_1), kwargs = {})
#   %view_1 : Tensor "f32[1, 256, 4096][256, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [1, 256, 4096]), kwargs = {})
#   %permute : Tensor "f32[1, 4096, 256][256, 256, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 2, 1]), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[1, 1, 256, 64, 64][256, 256, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, 1), kwargs = {})
#   %clone : Tensor "bf16[1, 1, 256, 64, 64][1048576, 1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 256, 64, 64]), kwargs = {})
#   %view_2 : Tensor "bf16[1, 256, 4096][1048576, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view, [1, 256, 4096]), kwargs = {})
#   %permute_1 : Tensor "bf16[1, 4096, 256][1048576, 1, 4096]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1]), kwargs = {})
#   %add_4 : Tensor "f32[1, 4096, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %permute_1), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[1, 4096, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   %convert_element_type_38 : Tensor "bf16[1, 4096, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute, torch.bfloat16), kwargs = {})
#   %add_12 : Tensor "f32[1, 4096, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %permute_1), kwargs = {})
#   %convert_element_type_60 : Tensor "bf16[1, 4096, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.bfloat16), kwargs = {})
#   return %convert_element_type_32,%convert_element_type_60,%convert_element_type_38
triton_poi_fused__to_copy_add_clone_permute_unsqueeze_view_8 = async_compile.triton('triton_poi_fused__to_copy_add_clone_permute_unsqueeze_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_permute_unsqueeze_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4194304, 'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_permute_unsqueeze_view_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 4096*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1 + 256*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp2.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 256*y0), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + 256*y0), tmp6, xmask)
    tl.store(out_ptr2 + (x1 + 256*y0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/qp/cqpfvschkwznkposy2qqldm6dtmuat6zlzgwzjrvtmvzo7oycy5g.py
# Topologically Sorted Source Nodes: [k_3, src, flatten, image_embedding, pos_src, flatten_1, image_pe, k_2], Original ATen: [aten._to_copy, aten.add, aten.view, aten.permute, aten.unsqueeze, aten.clone, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   flatten => view_1
#   flatten_1 => view_2
#   image_embedding => permute
#   image_pe => permute_1
#   k_2 => add_4
#   k_3 => addmm_5, convert_element_type_30, convert_element_type_31, convert_element_type_32, permute_11, view_17
#   pos_src => clone, unsqueeze_1, view
#   src => add
# Graph fragment:
#   %convert_element_type_30 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=convert_element_type_30]
#   %convert_element_type_32 : Tensor "bf16[1, 4096, 256][1048576, 256, 1]cuda:0" = PlaceHolder[target=convert_element_type_32]
#   %convert_element_type_31 : Tensor "bf16[128, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_31]
#   %convert_element_type_30 : Tensor "bf16[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg20_1, torch.bfloat16), kwargs = {})
#   %add : Tensor "f32[1, 256, 64, 64][256, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %arg5_1), kwargs = {})
#   %view_1 : Tensor "f32[1, 256, 4096][256, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [1, 256, 4096]), kwargs = {})
#   %permute : Tensor "f32[1, 4096, 256][256, 256, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 2, 1]), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[1, 1, 256, 64, 64][256, 256, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, 1), kwargs = {})
#   %clone : Tensor "bf16[1, 1, 256, 64, 64][1048576, 1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 256, 64, 64]), kwargs = {})
#   %view_2 : Tensor "bf16[1, 256, 4096][1048576, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view, [1, 256, 4096]), kwargs = {})
#   %permute_1 : Tensor "bf16[1, 4096, 256][1048576, 1, 4096]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1]), kwargs = {})
#   %add_4 : Tensor "f32[1, 4096, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %permute_1), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[1, 4096, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   %view_17 : Tensor "bf16[4096, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_32, [4096, 256]), kwargs = {})
#   %convert_element_type_31 : Tensor "bf16[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg19_1, torch.bfloat16), kwargs = {})
#   %permute_11 : Tensor "bf16[256, 128][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_31, [1, 0]), kwargs = {})
#   %addmm_5 : Tensor "bf16[4096, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_30, %view_17, %permute_11), kwargs = {})
#   return %addmm_5
triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9 = async_compile.triton('triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=8,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4096
    N = 128
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
        xindex = idx_n + 128*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 128*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/wi/cwiuk7gyfpbzefmibdacbwwyk6gavomjiqorqjnhqnsz5ksqxh33.py
# Topologically Sorted Source Nodes: [out_5, x_7, out_4], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   out_4 => view_24
#   out_5 => addmm_7, convert_element_type_42, convert_element_type_43, permute_17, view_25
#   x_7 => permute_16
# Graph fragment:
#   %convert_element_type_42 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type_42]
#   %getitem_11 : Tensor "bf16[1, 8, 8, 16][1024, 16, 128, 1]cuda:0" = PlaceHolder[target=getitem_11]
#   %convert_element_type_43 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=convert_element_type_43]
#   %convert_element_type_42 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg24_1, torch.bfloat16), kwargs = {})
#   %permute_16 : Tensor "bf16[1, 8, 8, 16][1024, 128, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_11, [0, 2, 1, 3]), kwargs = {})
#   %view_24 : Tensor "bf16[1, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_16, [1, 8, 128]), kwargs = {})
#   %view_25 : Tensor "bf16[8, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_24, [8, 128]), kwargs = {})
#   %convert_element_type_43 : Tensor "bf16[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg23_1, torch.bfloat16), kwargs = {})
#   %permute_17 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_43, [1, 0]), kwargs = {})
#   %addmm_7 : Tensor "bf16[8, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_42, %view_25, %permute_17), kwargs = {})
#   return %addmm_7
triton_tem_fused__to_copy_addmm_t_transpose_view_10 = async_compile.triton('triton_tem_fused__to_copy_addmm_t_transpose_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_t_transpose_view_10', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_t_transpose_view_10(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8
    N = 256
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 128
    stride_ak = 1
    stride_bk = 1
    stride_bn = 128

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
        xindex = idx_n + 128*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/yc/cyci6tcfwsxaf4xqfbinxk5o2y5i3v4itramgrw3fkyb7nqmpclk.py
# Topologically Sorted Source Nodes: [out_5, queries_1, queries_2, linear_8], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   linear_8 => convert_element_type_49
#   out_5 => view_26
#   queries_1 => add_5
#   queries_2 => add_6, add_7, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %add_2 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_2]
#   %addmm_7 : Tensor "bf16[8, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_7]
#   %getitem_21 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=getitem_21]
#   %buf48 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=buf48]
#   %arg25_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg25_1]
#   %arg26_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg26_1]
#   %add_7 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_7]
#   %view_26 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_7, [1, 8, 256]), kwargs = {})
#   %add_5 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_26), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_21), kwargs = {})
#   %add_6 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg25_1), kwargs = {})
#   %add_7 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg26_1), kwargs = {})
#   %convert_element_type_49 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.bfloat16), kwargs = {})
#   return %getitem_21,%buf48,%add_7,%convert_element_type_49
triton_per_fused__to_copy_add_native_layer_norm_view_11 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_view_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 38912}}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp31 = tmp30.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp30, xmask)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp31, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/qv/cqv5fdprputtfwutnq6rcaknzytxmawhp5qkow6eivy7roxzbval.py
# Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_8 => convert_element_type_48
# Graph fragment:
#   %arg27_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=arg27_1]
#   %convert_element_type_48 : Tensor "bf16[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_48
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/ft/cftjymom7acprgepkzeyjcqimzvrnvbad6gnrl6z2k4f435aksqn.py
# Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_8 => convert_element_type_47
# Graph fragment:
#   %arg28_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %convert_element_type_47 : Tensor "bf16[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg28_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_47
triton_poi_fused__to_copy_13 = async_compile.triton('triton_poi_fused__to_copy_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/2p/c2pirg6pumgyajuv2sonn5ysofdnkd442bdahqtkqzlocrwbiolz.py
# Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   linear_8 => addmm_8, convert_element_type_47, convert_element_type_48, convert_element_type_49, permute_18, view_27
# Graph fragment:
#   %convert_element_type_47 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=convert_element_type_47]
#   %convert_element_type_49 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=convert_element_type_49]
#   %convert_element_type_48 : Tensor "bf16[2048, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_48]
#   %convert_element_type_47 : Tensor "bf16[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg28_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_49 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.bfloat16), kwargs = {})
#   %view_27 : Tensor "bf16[8, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_49, [8, 256]), kwargs = {})
#   %convert_element_type_48 : Tensor "bf16[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   %permute_18 : Tensor "bf16[256, 2048][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_48, [1, 0]), kwargs = {})
#   %addmm_8 : Tensor "bf16[8, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_47, %view_27, %permute_18), kwargs = {})
#   return %addmm_8
triton_tem_fused__to_copy_addmm_t_view_14 = async_compile.triton('triton_tem_fused__to_copy_addmm_t_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=1,
num_warps=2,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_t_view_14', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 16, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_t_view_14(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8
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


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/z2/cz2wfy6hi7akvjzhgjhbwzyklkvxc4p3aju4sx6daifaz7n6znwn.py
# Topologically Sorted Source Nodes: [linear_8, x_8], Original ATen: [aten.view, aten.relu]
# Source node to ATen node mapping:
#   linear_8 => view_28
#   x_8 => relu
# Graph fragment:
#   %addmm_8 : Tensor "bf16[8, 2048][2048, 1]cuda:0" = PlaceHolder[target=addmm_8]
#   %view_28 : Tensor "bf16[1, 8, 2048][16384, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_8, [1, 8, 2048]), kwargs = {})
#   %relu : Tensor "bf16[1, 8, 2048][16384, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_28,), kwargs = {})
#   return %relu
triton_poi_fused_relu_view_15 = async_compile.triton('triton_poi_fused_relu_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_view_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 98304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_view_15(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/jt/cjtf5wnp5657qdhwvdgubpjvzw3yviar5ozauqe4evmqajr2nkvv.py
# Topologically Sorted Source Nodes: [x_9, linear_8, x_8], Original ATen: [aten._to_copy, aten.view, aten.relu, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   linear_8 => view_28
#   x_8 => relu
#   x_9 => addmm_9, convert_element_type_53, convert_element_type_54, permute_19, view_29
# Graph fragment:
#   %convert_element_type_53 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type_53]
#   %relu : Tensor "bf16[1, 8, 2048][16384, 2048, 1]cuda:0" = PlaceHolder[target=relu]
#   %convert_element_type_54 : Tensor "bf16[256, 2048][2048, 1]cuda:0" = PlaceHolder[target=convert_element_type_54]
#   %convert_element_type_53 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg30_1, torch.bfloat16), kwargs = {})
#   %view_28 : Tensor "bf16[1, 8, 2048][16384, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_8, [1, 8, 2048]), kwargs = {})
#   %relu : Tensor "bf16[1, 8, 2048][16384, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_28,), kwargs = {})
#   %view_29 : Tensor "bf16[8, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%relu, [8, 2048]), kwargs = {})
#   %convert_element_type_54 : Tensor "bf16[256, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg29_1, torch.bfloat16), kwargs = {})
#   %permute_19 : Tensor "bf16[2048, 256][1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_54, [1, 0]), kwargs = {})
#   %addmm_9 : Tensor "bf16[8, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_53, %view_29, %permute_19), kwargs = {})
#   return %addmm_9
triton_tem_fused__to_copy_addmm_relu_t_view_16 = async_compile.triton('triton_tem_fused__to_copy_addmm_relu_t_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=2,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_relu_t_view_16', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_relu_t_view_16(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8
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


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/ki/ckim76uq34jh3433kov56bhh7my5lda3bl4tgby64fxaiiz23kvj.py
# Topologically Sorted Source Nodes: [output_tokens, unsqueeze, output_tokens_1, tokens, x_9, queries_3, queries_4, q_5, k_6, v_4, q_8, q_9, k_8, v_6], Original ATen: [aten.cat, aten.unsqueeze, aten.expand, aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   k_6 => convert_element_type_66
#   k_8 => convert_element_type_89
#   output_tokens => cat
#   output_tokens_1 => expand
#   q_5 => add_11
#   q_8 => add_16
#   q_9 => convert_element_type_83
#   queries_3 => add_8
#   queries_4 => add_10, add_9, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
#   tokens => cat_1
#   unsqueeze => unsqueeze
#   v_4 => convert_element_type_72
#   v_6 => convert_element_type_95
#   x_9 => view_30
# Graph fragment:
#   %add_7 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_7]
#   %addmm_9 : Tensor "bf16[8, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %getitem_23 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=getitem_23]
#   %buf60 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=buf60]
#   %arg31_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg31_1]
#   %arg32_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %add_10 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_10]
#   %arg0_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %add_11 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_11]
#   %add_16 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_16]
#   %cat : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg0_1, %arg1_1, %arg2_1],), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 6, 256][1536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 0), kwargs = {})
#   %expand : Tensor "f32[1, 6, 256][1536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1]), kwargs = {})
#   %cat_1 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.cat.default](args = ([%expand, %arg3_1], 1), kwargs = {})
#   %view_30 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_9, [1, 8, 256]), kwargs = {})
#   %add_8 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_30), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_23), kwargs = {})
#   %add_9 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_4 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_5 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg31_1), kwargs = {})
#   %add_10 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg32_1), kwargs = {})
#   %add_11 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %cat_1), kwargs = {})
#   %convert_element_type_66 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.bfloat16), kwargs = {})
#   %convert_element_type_72 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
#   %add_16 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %cat_1), kwargs = {})
#   %convert_element_type_83 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_16, torch.bfloat16), kwargs = {})
#   %convert_element_type_89 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_16, torch.bfloat16), kwargs = {})
#   %convert_element_type_95 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
#   return %getitem_23,%buf60,%add_10,%add_11,%add_16,%convert_element_type_66,%convert_element_type_83,%convert_element_type_89,%convert_element_type_72,%convert_element_type_95
triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_17 = async_compile.triton('triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'out_ptr6': '*bf16', 'out_ptr7': '*bf16', 'out_ptr8': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 6, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 79872}}
)
@triton.jit
def triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp65 = tmp30.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp30, xmask)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp64, xmask)
    tl.store(out_ptr5 + (r0_1 + 256*x0), tmp64, xmask)
    tl.store(out_ptr6 + (r0_1 + 256*x0), tmp64, xmask)
    tl.store(out_ptr7 + (r0_1 + 256*x0), tmp65, xmask)
    tl.store(out_ptr8 + (r0_1 + 256*x0), tmp65, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/md/cmdodjp4ilnwiejqazlqui5puqseprx5xe2pjzdegsgmk6ricwyq.py
# Topologically Sorted Source Nodes: [out_8, x_13, out_7], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   out_7 => view_40
#   out_8 => addmm_13, convert_element_type_76, convert_element_type_77, permute_27, view_41
#   x_13 => permute_26
# Graph fragment:
#   %convert_element_type_76 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type_76]
#   %getitem_24 : Tensor "bf16[1, 8, 4096, 16][524288, 16, 128, 1]cuda:0" = PlaceHolder[target=getitem_24]
#   %convert_element_type_77 : Tensor "bf16[256, 128][128, 1]cuda:0" = PlaceHolder[target=convert_element_type_77]
#   %convert_element_type_76 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg40_1, torch.bfloat16), kwargs = {})
#   %permute_26 : Tensor "bf16[1, 4096, 8, 16][524288, 128, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_24, [0, 2, 1, 3]), kwargs = {})
#   %view_40 : Tensor "bf16[1, 4096, 128][524288, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_26, [1, 4096, 128]), kwargs = {})
#   %view_41 : Tensor "bf16[4096, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_40, [4096, 128]), kwargs = {})
#   %convert_element_type_77 : Tensor "bf16[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg39_1, torch.bfloat16), kwargs = {})
#   %permute_27 : Tensor "bf16[128, 256][1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_77, [1, 0]), kwargs = {})
#   %addmm_13 : Tensor "bf16[4096, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_76, %view_41, %permute_27), kwargs = {})
#   return %addmm_13
triton_tem_fused__to_copy_addmm_t_transpose_view_18 = async_compile.triton('triton_tem_fused__to_copy_addmm_t_transpose_view_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_t_transpose_view_18', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_t_transpose_view_18(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4096
    N = 256
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 128
    stride_ak = 1
    stride_bk = 1
    stride_bn = 128

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
        xindex = idx_n + 128*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 128*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/e5/ce572zzy3jnexgzs762ll3bh2mfkkxnl2dmdwfljqnfnzedq6zhn.py
# Topologically Sorted Source Nodes: [src, flatten, image_embedding, pos_src, flatten_1, image_pe, out_8, keys, keys_1, k_10, k_11, v_8, k_13, q_15], Original ATen: [aten.add, aten.view, aten.permute, aten.unsqueeze, aten.clone, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   flatten => view_1
#   flatten_1 => view_2
#   image_embedding => permute
#   image_pe => permute_1
#   k_10 => add_21
#   k_11 => convert_element_type_112
#   k_13 => add_29
#   keys => add_13
#   keys_1 => add_14, add_15, mul_6, mul_7, rsqrt_3, sub_3, var_mean_3
#   out_8 => view_42
#   pos_src => clone, unsqueeze_1, view
#   q_15 => convert_element_type_140
#   src => add
#   v_8 => convert_element_type_118
# Graph fragment:
#   %arg4_1 : Tensor "f32[1, 256, 64, 64][256, 1, 16384, 256]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %addmm_13 : Tensor "bf16[4096, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_13]
#   %getitem_34 : Tensor "f32[1, 4096, 1][4096, 1, 4096]cuda:0" = PlaceHolder[target=getitem_34]
#   %buf86 : Tensor "f32[1, 4096, 1][4096, 1, 4096]cuda:0" = PlaceHolder[target=buf86]
#   %arg41_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %arg42_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg42_1]
#   %add_15 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0" = PlaceHolder[target=add_15]
#   %arg6_1 : Tensor "bf16[1, 256, 64, 64][256, 1, 16384, 256]cuda:0" = PlaceHolder[target=arg6_1]
#   %add : Tensor "f32[1, 256, 64, 64][256, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %arg5_1), kwargs = {})
#   %view_1 : Tensor "f32[1, 256, 4096][256, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [1, 256, 4096]), kwargs = {})
#   %permute : Tensor "f32[1, 4096, 256][256, 256, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 2, 1]), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[1, 1, 256, 64, 64][256, 256, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, 1), kwargs = {})
#   %clone : Tensor "bf16[1, 1, 256, 64, 64][1048576, 1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 256, 64, 64]), kwargs = {})
#   %view_2 : Tensor "bf16[1, 256, 4096][1048576, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view, [1, 256, 4096]), kwargs = {})
#   %permute_1 : Tensor "bf16[1, 4096, 256][1048576, 1, 4096]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1]), kwargs = {})
#   %view_42 : Tensor "bf16[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_13, [1, 4096, 256]), kwargs = {})
#   %add_13 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %view_42), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_13, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %getitem_34), kwargs = {})
#   %add_14 : Tensor "f32[1, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_33, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_6 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_7 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %arg41_1), kwargs = {})
#   %add_15 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %arg42_1), kwargs = {})
#   %add_21 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %permute_1), kwargs = {})
#   %convert_element_type_112 : Tensor "bf16[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_21, torch.bfloat16), kwargs = {})
#   %convert_element_type_118 : Tensor "bf16[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.bfloat16), kwargs = {})
#   %add_29 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %permute_1), kwargs = {})
#   %convert_element_type_140 : Tensor "bf16[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_29, torch.bfloat16), kwargs = {})
#   return %getitem_34,%buf86,%add_15,%convert_element_type_112,%convert_element_type_140,%convert_element_type_118
triton_red_fused__to_copy_add_clone_native_layer_norm_permute_unsqueeze_view_19 = async_compile.triton('triton_red_fused__to_copy_add_clone_native_layer_norm_permute_unsqueeze_view_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_native_layer_norm_permute_unsqueeze_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 4, 'num_reduction': 2, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4194304, 'r0_': 29362176}}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_native_layer_norm_permute_unsqueeze_view_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp7_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 + tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight, roffset == 0
        )
        tmp7_mean = tl.where(r0_mask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(r0_mask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(r0_mask, tmp7_weight_next, tmp7_weight)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp7_mean, tmp7_m2, tmp7_weight, 1)
    tmp7 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (x0 + 4096*r0_1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr5 + (r0_1 + 256*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 + tmp17
        tmp19 = tmp18 - tmp7
        tmp20 = tl.full([1, 1], 256.0, tl.float32)
        tmp21 = (tmp11 / tmp20)
        tmp22 = tl.full([1, 1], 1e-05, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp29.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 256*x0), tmp29, r0_mask)
        tl.store(out_ptr3 + (r0_1 + 256*x0), tmp33, r0_mask)
        tl.store(out_ptr4 + (r0_1 + 256*x0), tmp33, r0_mask)
        tl.store(out_ptr5 + (r0_1 + 256*x0), tmp34, r0_mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/5v/c5vm4jofhnroujcgyzso5xepanplv6xps46mzxm5dhfpjrqvsepr.py
# Topologically Sorted Source Nodes: [output_tokens, unsqueeze, output_tokens_1, tokens, out_11, queries_5, queries_6, q_11, q_12], Original ATen: [aten.cat, aten.unsqueeze, aten.expand, aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   out_11 => view_54
#   output_tokens => cat
#   output_tokens_1 => expand
#   q_11 => add_20
#   q_12 => convert_element_type_106
#   queries_5 => add_17
#   queries_6 => add_18, add_19, mul_8, mul_9, rsqrt_4, sub_4, var_mean_4
#   tokens => cat_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %add_10 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_10]
#   %addmm_17 : Tensor "bf16[8, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %getitem_45 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=getitem_45]
#   %buf111 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=buf111]
#   %arg51_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg51_1]
#   %arg52_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %add_19 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_19]
#   %arg0_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %add_20 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_20]
#   %cat : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg0_1, %arg1_1, %arg2_1],), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 6, 256][1536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 0), kwargs = {})
#   %expand : Tensor "f32[1, 6, 256][1536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1]), kwargs = {})
#   %cat_1 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.cat.default](args = ([%expand, %arg3_1], 1), kwargs = {})
#   %view_54 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_17, [1, 8, 256]), kwargs = {})
#   %add_17 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_54), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_17, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_17, %getitem_45), kwargs = {})
#   %add_18 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_8 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_9 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg51_1), kwargs = {})
#   %add_19 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg52_1), kwargs = {})
#   %add_20 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %cat_1), kwargs = {})
#   %convert_element_type_106 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_20, torch.bfloat16), kwargs = {})
#   return %getitem_45,%buf111,%add_19,%add_20,%convert_element_type_106
triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_20 = async_compile.triton('triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_20', '''
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
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/zr/czrahjgpozltlt2hnlbmiowu6rcvy46tf37xjhvfvjdhrqkfv5ic.py
# Topologically Sorted Source Nodes: [linear_22, x_22], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear_22 => addmm_22, convert_element_type_127, convert_element_type_128, convert_element_type_129, permute_44, view_67, view_68
#   x_22 => relu_1
# Graph fragment:
#   %convert_element_type_127 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=convert_element_type_127]
#   %convert_element_type_129 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=convert_element_type_129]
#   %convert_element_type_128 : Tensor "bf16[2048, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_128]
#   %addmm_22 : Tensor "bf16[8, 2048][2048, 1]cuda:0" = PlaceHolder[target=addmm_22]
#   %convert_element_type_127 : Tensor "bf16[2048][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg64_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_129 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_24, torch.bfloat16), kwargs = {})
#   %view_67 : Tensor "bf16[8, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_129, [8, 256]), kwargs = {})
#   %convert_element_type_128 : Tensor "bf16[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg63_1, torch.bfloat16), kwargs = {})
#   %permute_44 : Tensor "bf16[256, 2048][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_128, [1, 0]), kwargs = {})
#   %addmm_22 : Tensor "bf16[8, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_127, %view_67, %permute_44), kwargs = {})
#   %view_68 : Tensor "bf16[1, 8, 2048][16384, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_22, [1, 8, 2048]), kwargs = {})
#   %relu_1 : Tensor "bf16[1, 8, 2048][16384, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_68,), kwargs = {})
#   return %addmm_22,%relu_1
triton_tem_fused__to_copy_addmm_relu_t_view_21 = async_compile.triton('triton_tem_fused__to_copy_addmm_relu_t_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=1,
num_warps=2,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr1': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_relu_t_view_21', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 16, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_relu_t_view_21(in_ptr0, arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8
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


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/io/ciofh4ft6jy6oltjivfsxskwpua2lm6pmqbbuqlznb3rrcbfzl2z.py
# Topologically Sorted Source Nodes: [output_tokens, unsqueeze, output_tokens_1, tokens, x_23, queries_9, queries_10, q_14, k_14, v_10, q_17, q_18], Original ATen: [aten.cat, aten.unsqueeze, aten.expand, aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   k_14 => convert_element_type_146
#   output_tokens => cat
#   output_tokens_1 => expand
#   q_14 => add_28
#   q_17 => add_33
#   q_18 => convert_element_type_163
#   queries_10 => add_26, add_27, mul_12, mul_13, rsqrt_6, sub_6, var_mean_6
#   queries_9 => add_25
#   tokens => cat_1
#   unsqueeze => unsqueeze
#   v_10 => convert_element_type_152
#   x_23 => view_70
# Graph fragment:
#   %add_24 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_24]
#   %addmm_23 : Tensor "bf16[8, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_23]
#   %getitem_58 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=getitem_58]
#   %buf150 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=buf150]
#   %arg67_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg67_1]
#   %arg68_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg68_1]
#   %add_27 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_27]
#   %arg0_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1, 2, 256][512, 256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %add_28 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_28]
#   %add_33 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_33]
#   %cat : Tensor "f32[6, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg0_1, %arg1_1, %arg2_1],), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 6, 256][1536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 0), kwargs = {})
#   %expand : Tensor "f32[1, 6, 256][1536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, -1, -1]), kwargs = {})
#   %cat_1 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=9] = call_function[target=torch.ops.aten.cat.default](args = ([%expand, %arg3_1], 1), kwargs = {})
#   %view_70 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_23, [1, 8, 256]), kwargs = {})
#   %add_25 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_70), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_25, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_25, %getitem_58), kwargs = {})
#   %add_26 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_57, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_26,), kwargs = {})
#   %mul_12 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %mul_13 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %arg67_1), kwargs = {})
#   %add_27 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %arg68_1), kwargs = {})
#   %add_28 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %cat_1), kwargs = {})
#   %convert_element_type_146 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_28, torch.bfloat16), kwargs = {})
#   %convert_element_type_152 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_27, torch.bfloat16), kwargs = {})
#   %add_33 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %cat_1), kwargs = {})
#   %convert_element_type_163 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.bfloat16), kwargs = {})
#   return %getitem_58,%buf150,%add_27,%add_28,%add_33,%convert_element_type_146,%convert_element_type_163,%convert_element_type_152
triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_22 = async_compile.triton('triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'out_ptr6': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 4, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 63488}}
)
@triton.jit
def triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr4, out_ptr5, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp65 = tmp30.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp30, xmask)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp64, xmask)
    tl.store(out_ptr5 + (r0_1 + 256*x0), tmp64, xmask)
    tl.store(out_ptr6 + (r0_1 + 256*x0), tmp65, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/tf/ctfzqmjwi6xr2xm53cza4sgqlxxed7ec6uaidj3sb246thvjz3np.py
# Topologically Sorted Source Nodes: [pos_src, flatten_1, image_pe, out_17, keys_2, keys_3, k_16, k_17, v_12, transpose_28, src_1, conv_transpose2d], Original ATen: [aten.unsqueeze, aten.clone, aten.view, aten.permute, aten.add, aten.native_layer_norm, aten._to_copy, aten.transpose, aten.convolution]
# Source node to ATen node mapping:
#   conv_transpose2d => convert_element_type_184, convert_element_type_185, convert_element_type_186, convolution
#   flatten_1 => view_2
#   image_pe => permute_1
#   k_16 => add_34
#   k_17 => convert_element_type_169
#   keys_2 => add_30
#   keys_3 => add_31, add_32, mul_14, mul_15, rsqrt_7, sub_7, var_mean_7
#   out_17 => view_82
#   pos_src => clone, unsqueeze_1, view
#   src_1 => view_95
#   transpose_28 => permute_62
#   v_12 => convert_element_type_175
# Graph fragment:
#   %add_15 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0" = PlaceHolder[target=add_15]
#   %addmm_27 : Tensor "bf16[4096, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_27]
#   %getitem_69 : Tensor "f32[1, 4096, 1][4096, 1, 4096]cuda:0" = PlaceHolder[target=getitem_69]
#   %buf176 : Tensor "f32[1, 4096, 1][4096, 1, 4096]cuda:0" = PlaceHolder[target=buf176]
#   %arg77_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg77_1]
#   %arg78_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg78_1]
#   %add_32 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0" = PlaceHolder[target=add_32]
#   %arg6_1 : Tensor "bf16[1, 256, 64, 64][256, 1, 16384, 256]cuda:0" = PlaceHolder[target=arg6_1]
#   %unsqueeze_1 : Tensor "bf16[1, 1, 256, 64, 64][256, 256, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, 1), kwargs = {})
#   %clone : Tensor "bf16[1, 1, 256, 64, 64][1048576, 1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 256, 64, 64]), kwargs = {})
#   %view_2 : Tensor "bf16[1, 256, 4096][1048576, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view, [1, 256, 4096]), kwargs = {})
#   %permute_1 : Tensor "bf16[1, 4096, 256][1048576, 1, 4096]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1]), kwargs = {})
#   %view_82 : Tensor "bf16[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_27, [1, 4096, 256]), kwargs = {})
#   %add_30 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_82), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_30, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_30, %getitem_69), kwargs = {})
#   %add_31 : Tensor "f32[1, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_68, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, 4096, 1][4096, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_31,), kwargs = {})
#   %mul_14 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %mul_15 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %arg77_1), kwargs = {})
#   %add_32 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %arg78_1), kwargs = {})
#   %add_34 : Tensor "f32[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %permute_1), kwargs = {})
#   %convert_element_type_169 : Tensor "bf16[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_34, torch.bfloat16), kwargs = {})
#   %convert_element_type_175 : Tensor "bf16[1, 4096, 256][1048576, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_32, torch.bfloat16), kwargs = {})
#   %permute_62 : Tensor "f32[1, 256, 4096][1048576, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_95 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [1, 256, 64, 64]), kwargs = {})
#   %convert_element_type_186 : Tensor "bf16[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_95, torch.bfloat16), kwargs = {})
#   %convert_element_type_185 : Tensor "bf16[256, 64, 2, 2][256, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg89_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_184 : Tensor "bf16[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg90_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_186, %convert_element_type_185, %convert_element_type_184, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %getitem_69,%buf176,%add_32,%convert_element_type_169,%convert_element_type_175,%buf258
triton_per_fused__to_copy_add_clone_convolution_native_layer_norm_permute_transpose_unsqueeze_view_23 = async_compile.triton('triton_per_fused__to_copy_add_clone_convolution_native_layer_norm_permute_transpose_unsqueeze_view_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_convolution_native_layer_norm_permute_transpose_unsqueeze_view_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 20973568}}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_convolution_native_layer_norm_permute_transpose_unsqueeze_view_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), None)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 256*x0), None).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r0_1 + 256*x0), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 256, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp17 = tmp3 - tmp11
    tmp18 = tl.full([1, 1], 256.0, tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tl.full([1, 1], 1e-05, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp31, None)
    tl.store(out_ptr3 + (r0_1 + 256*x0), tmp32, None)
    tl.store(out_ptr4 + (r0_1 + 256*x0), tmp32, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/d2/cd2sprguz7p5bgbgn7xivooccbj5lbrsitqhswo4rd77cdgb4sie.py
# Topologically Sorted Source Nodes: [out_20, queries_11, queries_12], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   out_20 => view_94
#   queries_11 => add_35
#   queries_12 => add_36, add_37, mul_16, mul_17, rsqrt_8, sub_8, var_mean_8
# Graph fragment:
#   %add_27 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_27]
#   %addmm_31 : Tensor "bf16[8, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_31]
#   %getitem_80 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=getitem_80]
#   %buf202 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=buf202]
#   %arg87_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg87_1]
#   %arg88_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg88_1]
#   %view_94 : Tensor "bf16[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_31, [1, 8, 256]), kwargs = {})
#   %add_35 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_94), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_35, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_35, %getitem_80), kwargs = {})
#   %add_36 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_79, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_36,), kwargs = {})
#   %mul_16 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_17 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %arg87_1), kwargs = {})
#   %add_37 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %arg88_1), kwargs = {})
#   return %getitem_80,%buf202,%add_37
triton_per_fused_add_native_layer_norm_view_24 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 30720}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp30, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/qh/cqhq4lbydzxt73v3hfuf4zllnmi45tl3bljeljqe7d6kpdxjdpr2.py
# Topologically Sorted Source Nodes: [mask_tokens_out, getitem_4, linear_32], Original ATen: [aten.slice, aten.select, aten._to_copy]
# Source node to ATen node mapping:
#   getitem_4 => select_1
#   linear_32 => convert_element_type_195
#   mask_tokens_out => slice_1
# Graph fragment:
#   %add_37 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_37]
#   %slice_1 : Tensor "f32[1, 4, 256][2048, 256, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_37, 1, 2, 6), kwargs = {})
#   %select_1 : Tensor "f32[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%slice_1, 1, 0), kwargs = {})
#   %convert_element_type_195 : Tensor "bf16[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_195
triton_poi_fused__to_copy_select_slice_25 = async_compile.triton('triton_poi_fused__to_copy_select_slice_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_select_slice_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_select_slice_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/6s/c6sovyz7ighpz5bpijhvupaq6yota7xoyk2qafk4hutaesywa2hx.py
# Topologically Sorted Source Nodes: [linear_32, mask_tokens_out, getitem_4], Original ATen: [aten._to_copy, aten.slice, aten.select, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   getitem_4 => select_1
#   linear_32 => addmm_32, convert_element_type_193, convert_element_type_194, convert_element_type_195, permute_63
#   mask_tokens_out => slice_1
# Graph fragment:
#   %convert_element_type_193 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type_193]
#   %convert_element_type_195 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_195]
#   %convert_element_type_194 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_194]
#   %convert_element_type_193 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg98_1, torch.bfloat16), kwargs = {})
#   %slice_1 : Tensor "f32[1, 4, 256][2048, 256, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_37, 1, 2, 6), kwargs = {})
#   %select_1 : Tensor "f32[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%slice_1, 1, 0), kwargs = {})
#   %convert_element_type_195 : Tensor "bf16[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_194 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg97_1, torch.bfloat16), kwargs = {})
#   %permute_63 : Tensor "bf16[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_194, [1, 0]), kwargs = {})
#   %addmm_32 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_193, %convert_element_type_195, %permute_63), kwargs = {})
#   return %addmm_32
triton_tem_fused__to_copy_addmm_select_slice_t_26 = async_compile.triton('triton_tem_fused__to_copy_addmm_select_slice_t_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=2,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_select_slice_t_26', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_select_slice_t_26(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1
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
        a = tl.load(A + ((tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))

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
    tl.store(out_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), tmp1, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/in/cinmdn2msfslwrufywoi6f46ifmiveuvosqcon62edo5sahmxqvr.py
# Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   x_34 => relu_2
# Graph fragment:
#   %addmm_32 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_32]
#   %relu_2 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_32,), kwargs = {})
#   return %relu_2
triton_poi_fused_relu_27 = async_compile.triton('triton_poi_fused_relu_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_27(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/vr/cvrhx6nzctkxy7t5no2s7k76j6aqqejibu6rtk62tqtmfs2gi7qa.py
# Topologically Sorted Source Nodes: [x_36], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_36 => convert_element_type_205
# Graph fragment:
#   %arg101_1 : Tensor "f32[32, 256][256, 1]cuda:0" = PlaceHolder[target=arg101_1]
#   %convert_element_type_205 : Tensor "bf16[32, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg101_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_205
triton_poi_fused__to_copy_28 = async_compile.triton('triton_poi_fused__to_copy_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 65536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/yo/cyod6cep5jbiymriknkzkze73unhefiqjv6sk4mq5duaw6qktkwc.py
# Topologically Sorted Source Nodes: [x_36], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_36 => convert_element_type_204
# Graph fragment:
#   %arg102_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg102_1]
#   %convert_element_type_204 : Tensor "bf16[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg102_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_204
triton_poi_fused__to_copy_29 = async_compile.triton('triton_poi_fused__to_copy_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/ze/czedihncnjuqhfnqqoorlj2ioio7kqpxdqhgs3omm3rsiz2eouzu.py
# Topologically Sorted Source Nodes: [x_36, x_35], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   x_35 => relu_3
#   x_36 => addmm_34, convert_element_type_204, convert_element_type_205, permute_65
# Graph fragment:
#   %convert_element_type_204 : Tensor "bf16[32][1]cuda:0" = PlaceHolder[target=convert_element_type_204]
#   %relu_3 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=relu_3]
#   %convert_element_type_205 : Tensor "bf16[32, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_205]
#   %convert_element_type_204 : Tensor "bf16[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg102_1, torch.bfloat16), kwargs = {})
#   %relu_3 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_33,), kwargs = {})
#   %convert_element_type_205 : Tensor "bf16[32, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg101_1, torch.bfloat16), kwargs = {})
#   %permute_65 : Tensor "bf16[256, 32][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_205, [1, 0]), kwargs = {})
#   %addmm_34 : Tensor "bf16[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_204, %relu_3, %permute_65), kwargs = {})
#   return %addmm_34
triton_tem_fused__to_copy_addmm_relu_t_30 = async_compile.triton('triton_tem_fused__to_copy_addmm_relu_t_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=2,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_relu_t_30', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_relu_t_30(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1
    N = 32
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
        a = tl.load(A + ((tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 32*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 32*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), tmp1, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/qu/cqucooakyajuxeqpuzsaaad7cu5dkvwdjjeiyaouyh44p4dpmspb.py
# Topologically Sorted Source Nodes: [mask_tokens_out, getitem_5, linear_35], Original ATen: [aten.slice, aten.select, aten._to_copy]
# Source node to ATen node mapping:
#   getitem_5 => select_2
#   linear_35 => convert_element_type_211
#   mask_tokens_out => slice_1
# Graph fragment:
#   %add_37 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_37]
#   %slice_1 : Tensor "f32[1, 4, 256][2048, 256, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_37, 1, 2, 6), kwargs = {})
#   %select_2 : Tensor "f32[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%slice_1, 1, 1), kwargs = {})
#   %convert_element_type_211 : Tensor "bf16[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_2, torch.bfloat16), kwargs = {})
#   return %convert_element_type_211
triton_poi_fused__to_copy_select_slice_31 = async_compile.triton('triton_poi_fused__to_copy_select_slice_31', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_select_slice_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_select_slice_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/ai/cai4d7v4nqxm3o7q64g6jtyw6rdlzjfc6f5ezyqlaleitheeqzui.py
# Topologically Sorted Source Nodes: [mask_tokens_out, getitem_6, linear_38], Original ATen: [aten.slice, aten.select, aten._to_copy]
# Source node to ATen node mapping:
#   getitem_6 => select_3
#   linear_38 => convert_element_type_227
#   mask_tokens_out => slice_1
# Graph fragment:
#   %add_37 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_37]
#   %slice_1 : Tensor "f32[1, 4, 256][2048, 256, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_37, 1, 2, 6), kwargs = {})
#   %select_3 : Tensor "f32[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%slice_1, 1, 2), kwargs = {})
#   %convert_element_type_227 : Tensor "bf16[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_227
triton_poi_fused__to_copy_select_slice_32 = async_compile.triton('triton_poi_fused__to_copy_select_slice_32', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_select_slice_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_select_slice_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/pz/cpzgiajsufkwabqmz3j5wtev3zvg3yw7mgazvka4n4me7wiw5bx2.py
# Topologically Sorted Source Nodes: [mask_tokens_out, getitem_7, linear_41], Original ATen: [aten.slice, aten.select, aten._to_copy]
# Source node to ATen node mapping:
#   getitem_7 => select_4
#   linear_41 => convert_element_type_243
#   mask_tokens_out => slice_1
# Graph fragment:
#   %add_37 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_37]
#   %slice_1 : Tensor "f32[1, 4, 256][2048, 256, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_37, 1, 2, 6), kwargs = {})
#   %select_4 : Tensor "f32[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%slice_1, 1, 3), kwargs = {})
#   %convert_element_type_243 : Tensor "bf16[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_4, torch.bfloat16), kwargs = {})
#   return %convert_element_type_243
triton_poi_fused__to_copy_select_slice_33 = async_compile.triton('triton_poi_fused__to_copy_select_slice_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_select_slice_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_select_slice_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1280 + x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/y3/cy3t6g5waz4ijzi43gsswtxebnu3wp37sx5lon4grzq2vt3nw5z6.py
# Topologically Sorted Source Nodes: [hyper_in], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   hyper_in => cat_2
# Graph fragment:
#   %addmm_34 : Tensor "bf16[1, 32][32, 1]cuda:0" = PlaceHolder[target=addmm_34]
#   %cat_2 : Tensor "bf16[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%addmm_34, %addmm_37, %addmm_40, %addmm_43], 1), kwargs = {})
#   return %buf253
triton_poi_fused_stack_34 = async_compile.triton('triton_poi_fused_stack_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/w4/cw46qfhclqscwmq5zlbtfojx6ul3qy2whlgklimu42dlpmk3ku2i.py
# Topologically Sorted Source Nodes: [transpose_28, src_1, conv_transpose2d], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   conv_transpose2d => convert_element_type_184, convert_element_type_185, convert_element_type_186, convolution
#   src_1 => view_95
#   transpose_28 => permute_62
# Graph fragment:
#   %arg89_1 : Tensor "f32[256, 64, 2, 2][256, 4, 2, 1]cuda:0" = PlaceHolder[target=arg89_1]
#   %buf259 : Tensor "bf16[256, 64, 2, 2][256, 4, 2, 1]cuda:0" = PlaceHolder[target=buf259]
#   %permute_62 : Tensor "f32[1, 256, 4096][1048576, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_95 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [1, 256, 64, 64]), kwargs = {})
#   %convert_element_type_186 : Tensor "bf16[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_95, torch.bfloat16), kwargs = {})
#   %convert_element_type_185 : Tensor "bf16[256, 64, 2, 2][256, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg89_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_184 : Tensor "bf16[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg90_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_186, %convert_element_type_185, %convert_element_type_184, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf259,%buf261
triton_poi_fused__to_copy_convolution_transpose_view_35 = async_compile.triton('triton_poi_fused__to_copy_convolution_transpose_view_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_transpose_view_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_transpose_view_35(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 4)
    x2 = ((xindex // 4) % 64)
    x3 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr1 + (x2 + 64*x1 + 256*x3), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/k4/ck4ae6brlduvjeeuidim5nag34zytjxpdenlwtdzluqkfzm4x5vj.py
# Topologically Sorted Source Nodes: [transpose_28, src_1, conv_transpose2d], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   conv_transpose2d => convert_element_type_184, convert_element_type_185, convert_element_type_186, convolution
#   src_1 => view_95
#   transpose_28 => permute_62
# Graph fragment:
#   %arg90_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %permute_62 : Tensor "f32[1, 256, 4096][1048576, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_95 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [1, 256, 64, 64]), kwargs = {})
#   %convert_element_type_186 : Tensor "bf16[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_95, torch.bfloat16), kwargs = {})
#   %convert_element_type_185 : Tensor "bf16[256, 64, 2, 2][256, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg89_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_184 : Tensor "bf16[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg90_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_186, %convert_element_type_185, %convert_element_type_184, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf260
triton_poi_fused__to_copy_convolution_transpose_view_36 = async_compile.triton('triton_poi_fused__to_copy_convolution_transpose_view_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_transpose_view_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_transpose_view_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/en/cengcxx7uqgnm6t4zj54cow54nh3sc6itw4ld533ylpl3swgnjq7.py
# Topologically Sorted Source Nodes: [getitem_2, transpose_28, src_1, conv_transpose2d, add_20, u, sub_1, sub, pow_1, s, add_21, sqrt, x_32, mul, getitem_3, x_33, upscaled_embedding, conv_transpose2d_1], Original ATen: [aten.unsqueeze, aten.transpose, aten.view, aten._to_copy, aten.convolution, aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul, aten.gelu]
# Source node to ATen node mapping:
#   add_20 => add_38
#   add_21 => add_39
#   conv_transpose2d => convert_element_type_184, convert_element_type_185, convert_element_type_186, convolution
#   conv_transpose2d_1 => convert_element_type_188, convert_element_type_189, convert_element_type_190, convolution_1
#   getitem_2 => unsqueeze_2, unsqueeze_3
#   getitem_3 => unsqueeze_4, unsqueeze_5
#   mul => mul_18
#   pow_1 => convert_element_type_187, pow_1
#   s => mean_1
#   sqrt => sqrt
#   src_1 => view_95
#   sub => sub_9
#   sub_1 => sub_10
#   transpose_28 => permute_62
#   u => mean
#   upscaled_embedding => add_41, erf, mul_19, mul_20, mul_21
#   x_32 => div
#   x_33 => add_40
# Graph fragment:
#   %buf262 : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = PlaceHolder[target=buf262]
#   %buf260 : Tensor "bf16[64][1]cuda:0" = PlaceHolder[target=buf260]
#   %arg91_1 : Tensor "bf16[1, 64, 128, 128][64, 1, 8192, 64]cuda:0" = PlaceHolder[target=arg91_1]
#   %buf263 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = PlaceHolder[target=buf263]
#   %arg92_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg92_1]
#   %buf264 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = PlaceHolder[target=buf264]
#   %arg93_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg93_1]
#   %add_40 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = PlaceHolder[target=add_40]
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg92_1, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 2), kwargs = {})
#   %permute_62 : Tensor "f32[1, 256, 4096][1048576, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_32, [0, 2, 1]), kwargs = {})
#   %view_95 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_62, [1, 256, 64, 64]), kwargs = {})
#   %convert_element_type_186 : Tensor "bf16[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_95, torch.bfloat16), kwargs = {})
#   %convert_element_type_185 : Tensor "bf16[256, 64, 2, 2][256, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg89_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_184 : Tensor "bf16[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg90_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_186, %convert_element_type_185, %convert_element_type_184, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   %add_38 : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %arg91_1), kwargs = {})
#   %mean : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_38, [1], True), kwargs = {})
#   %sub_10 : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_38, %mean), kwargs = {})
#   %sub_9 : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_38, %mean), kwargs = {})
#   %convert_element_type_187 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_9, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_187, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   %add_39 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %sqrt : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_39,), kwargs = {})
#   %div : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_10, %sqrt), kwargs = {})
#   %mul_18 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %div), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg93_1, 1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 2), kwargs = {})
#   %add_40 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_5), kwargs = {})
#   %mul_19 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, 0.5), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_20,), kwargs = {})
#   %add_41 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_21 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %add_41), kwargs = {})
#   %convert_element_type_190 : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_21, torch.bfloat16), kwargs = {})
#   %convert_element_type_189 : Tensor "bf16[64, 32, 2, 2][128, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg94_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_188 : Tensor "bf16[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg95_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_190, %convert_element_type_189, %convert_element_type_188, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf263,%buf264,%add_40,%buf266
triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_transpose_unsqueeze_view_37 = async_compile.triton('triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_transpose_unsqueeze_view_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_transpose_unsqueeze_view_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 8389248}}
)
@triton.jit
def triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_transpose_unsqueeze_view_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 64*x0), None).to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 64.0, tl.float32)
    tmp10 = (tmp8 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp19 = (tmp17 / tmp9)
    tmp20 = tl.full([1, 1], 1e-06, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt_rn(tmp21)
    tmp23 = (tmp13 / tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full([1, 1], 0.5, tl.float32)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp30 = tmp26 * tmp29
    tmp31 = libdevice.erf(tmp30)
    tmp32 = tl.full([1, 1], 1.0, tl.float32)
    tmp33 = tmp31 + tmp32
    tmp34 = tmp28 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 64*x0), tmp35, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/kh/ckhl4af2gof5xsxungu7r7am7dg5n7nyolpm7doofm2u2ssvxcrd.py
# Topologically Sorted Source Nodes: [upscaled_embedding, conv_transpose2d_1], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   conv_transpose2d_1 => convert_element_type_188, convert_element_type_189, convert_element_type_190, convolution_1
#   upscaled_embedding => add_41, erf, mul_19, mul_20, mul_21
# Graph fragment:
#   %arg94_1 : Tensor "f32[64, 32, 2, 2][128, 4, 2, 1]cuda:0" = PlaceHolder[target=arg94_1]
#   %buf267 : Tensor "bf16[64, 32, 2, 2][128, 4, 2, 1]cuda:0" = PlaceHolder[target=buf267]
#   %mul_19 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, 0.5), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_20,), kwargs = {})
#   %add_41 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_21 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %add_41), kwargs = {})
#   %convert_element_type_190 : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_21, torch.bfloat16), kwargs = {})
#   %convert_element_type_189 : Tensor "bf16[64, 32, 2, 2][128, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg94_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_188 : Tensor "bf16[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg95_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_190, %convert_element_type_189, %convert_element_type_188, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   return %buf267,%buf269
triton_poi_fused__to_copy_convolution_gelu_38 = async_compile.triton('triton_poi_fused__to_copy_convolution_gelu_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_gelu_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_gelu_38(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 4)
    x2 = ((xindex // 4) % 32)
    x3 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr1 + (x2 + 32*x1 + 128*x3), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/sv/csvm4eh4s3qcqy56ujxrpopy3ovx4zusho4behrbpf3o55dqdtln.py
# Topologically Sorted Source Nodes: [upscaled_embedding, conv_transpose2d_1, add_23, upscaled_embedding_1], Original ATen: [aten.gelu, aten._to_copy, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_23 => add_42
#   conv_transpose2d_1 => convert_element_type_188, convert_element_type_189, convert_element_type_190, convolution_1
#   upscaled_embedding => add_41, erf, mul_19, mul_20, mul_21
#   upscaled_embedding_1 => add_43, convert_element_type_191, convert_element_type_192, erf_1, mul_22, mul_23, mul_24
# Graph fragment:
#   %buf270 : Tensor "bf16[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0" = PlaceHolder[target=buf270]
#   %buf268 : Tensor "bf16[32][1]cuda:0" = PlaceHolder[target=buf268]
#   %arg96_1 : Tensor "bf16[1, 32, 256, 256][2097152, 65536, 256, 1]cuda:0" = PlaceHolder[target=arg96_1]
#   %mul_19 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, 0.5), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_20,), kwargs = {})
#   %add_41 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_21 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %add_41), kwargs = {})
#   %convert_element_type_190 : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_21, torch.bfloat16), kwargs = {})
#   %convert_element_type_189 : Tensor "bf16[64, 32, 2, 2][128, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg94_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_188 : Tensor "bf16[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg95_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_190, %convert_element_type_189, %convert_element_type_188, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   %add_42 : Tensor "bf16[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %arg96_1), kwargs = {})
#   %convert_element_type_191 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_42, torch.float32), kwargs = {})
#   %mul_22 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_191, 0.5), kwargs = {})
#   %mul_23 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_191, 0.7071067811865476), kwargs = {})
#   %erf_1 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_23,), kwargs = {})
#   %add_43 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_24 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %add_43), kwargs = {})
#   %convert_element_type_192 : Tensor "bf16[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.bfloat16), kwargs = {})
#   return %convert_element_type_192
triton_poi_fused__to_copy_add_convolution_gelu_39 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_gelu_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4194304, 'x': 12582976}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_gelu_39(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 32
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_out_ptr0 + (x1 + 32*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0 + 65536*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.full([1, 1], 0.5, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp9 = tmp5 * tmp8
    tmp10 = libdevice.erf(tmp9)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp7 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 32*y0), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/4r/c4rv2tmodcqlwlxmrch2kwcdelosigidigcuzrefklw2pks2n2uu.py
# Topologically Sorted Source Nodes: [hyper_in, matmul, upscaled_embedding, conv_transpose2d_1, add_23, upscaled_embedding_1, view_1], Original ATen: [aten.stack, aten.bmm, aten.gelu, aten._to_copy, aten.convolution, aten.add, aten.view]
# Source node to ATen node mapping:
#   add_23 => add_42
#   conv_transpose2d_1 => convert_element_type_188, convert_element_type_189, convert_element_type_190, convolution_1
#   hyper_in => view_96
#   matmul => mm_default, squeeze_dim, squeeze_dim_1
#   upscaled_embedding => add_41, erf, mul_19, mul_20, mul_21
#   upscaled_embedding_1 => add_43, convert_element_type_191, convert_element_type_192, erf_1, mul_22, mul_23, mul_24
#   view_1 => view_97
# Graph fragment:
#   %cat_2 : Tensor "bf16[1, 128][128, 1]cuda:0" = PlaceHolder[target=cat_2]
#   %convert_element_type_192 : Tensor "bf16[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0" = PlaceHolder[target=convert_element_type_192]
#   %view_96 : Tensor "bf16[1, 4, 32][128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_2, [1, 4, 32]), kwargs = {})
#   %squeeze_dim : Tensor "bf16[4, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%expand_2, 0), kwargs = {})
#   %mul_19 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, 0.5), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_20,), kwargs = {})
#   %add_41 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_21 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %add_41), kwargs = {})
#   %convert_element_type_190 : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_21, torch.bfloat16), kwargs = {})
#   %convert_element_type_189 : Tensor "bf16[64, 32, 2, 2][128, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg94_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_188 : Tensor "bf16[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg95_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_190, %convert_element_type_189, %convert_element_type_188, [2, 2], [0, 0], [1, 1], True, [0, 0], 1), kwargs = {})
#   %add_42 : Tensor "bf16[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %arg96_1), kwargs = {})
#   %convert_element_type_191 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_42, torch.float32), kwargs = {})
#   %mul_22 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_191, 0.5), kwargs = {})
#   %mul_23 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_191, 0.7071067811865476), kwargs = {})
#   %erf_1 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_23,), kwargs = {})
#   %add_43 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_24 : Tensor "f32[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %add_43), kwargs = {})
#   %convert_element_type_192 : Tensor "bf16[1, 32, 256, 256][2097152, 1, 8192, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.bfloat16), kwargs = {})
#   %view_97 : Tensor "bf16[1, 32, 65536][2097152, 1, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_192, [1, 32, 65536]), kwargs = {})
#   %squeeze_dim_1 : Tensor "bf16[32, 65536][1, 32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%expand_3, 0), kwargs = {})
#   %mm_default : Tensor "bf16[4, 65536][65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%squeeze_dim, %squeeze_dim_1), kwargs = {})
#   return %mm_default
triton_tem_fused__to_copy_add_bmm_convolution_gelu_stack_view_40 = async_compile.triton('triton_tem_fused__to_copy_add_bmm_convolution_gelu_stack_view_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_bmm_convolution_gelu_stack_view_40', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_add_bmm_convolution_gelu_stack_view_40(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4
    N = 65536
    K = 32
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 32
    stride_ak = 1
    stride_bk = 1
    stride_bn = 32

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
        xindex = idx_n + 32*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 65536*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 32*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 65536*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/xk/cxkmotys4q45riok725esohjd6fneiffpvjum65t4btinnc3jesv.py
# Topologically Sorted Source Nodes: [iou_token_out, linear_44], Original ATen: [aten.select, aten._to_copy]
# Source node to ATen node mapping:
#   iou_token_out => select
#   linear_44 => convert_element_type_261
# Graph fragment:
#   %add_37 : Tensor "f32[1, 8, 256][2048, 256, 1]cuda:0" = PlaceHolder[target=add_37]
#   %select : Tensor "f32[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%add_37, 1, 1), kwargs = {})
#   %convert_element_type_261 : Tensor "bf16[1, 256][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select, torch.bfloat16), kwargs = {})
#   return %convert_element_type_261
triton_poi_fused__to_copy_select_41 = async_compile.triton('triton_poi_fused__to_copy_select_41', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_select_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_select_41(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/pi/cpi7fxdupkeiwp2c5o34bbu7ymvkcsh2quo4jtveqoaveh3nattn.py
# Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_48 => convert_element_type_271
# Graph fragment:
#   %arg125_1 : Tensor "f32[4, 256][256, 1]cuda:0" = PlaceHolder[target=arg125_1]
#   %convert_element_type_271 : Tensor "bf16[4, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg125_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_271
triton_poi_fused__to_copy_42 = async_compile.triton('triton_poi_fused__to_copy_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/qx/cqxpxv7c4xwpz6jee6ti6ry5bz2ozfhc37wtg6xymyei2aabqbwv.py
# Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_48 => convert_element_type_270
# Graph fragment:
#   %arg126_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg126_1]
#   %convert_element_type_270 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg126_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_270
triton_poi_fused__to_copy_43 = async_compile.triton('triton_poi_fused__to_copy_43', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 16}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/bk/cbk6pet7gawzznlauyh7qdluhhqwjzy5th2lkvazak5wpnykn5zs.py
# Topologically Sorted Source Nodes: [x_48, x_47, x_49], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm, aten.sigmoid]
# Source node to ATen node mapping:
#   x_47 => relu_11
#   x_48 => addmm_46, convert_element_type_270, convert_element_type_271, permute_77
#   x_49 => sigmoid
# Graph fragment:
#   %convert_element_type_270 : Tensor "bf16[4][1]cuda:0" = PlaceHolder[target=convert_element_type_270]
#   %relu_11 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=relu_11]
#   %convert_element_type_271 : Tensor "bf16[4, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_271]
#   %addmm_46 : Tensor "bf16[1, 4][4, 1]cuda:0" = PlaceHolder[target=addmm_46]
#   %convert_element_type_270 : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg126_1, torch.bfloat16), kwargs = {})
#   %relu_11 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_45,), kwargs = {})
#   %convert_element_type_271 : Tensor "bf16[4, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg125_1, torch.bfloat16), kwargs = {})
#   %permute_77 : Tensor "bf16[256, 4][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_271, [1, 0]), kwargs = {})
#   %addmm_46 : Tensor "bf16[1, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_270, %relu_11, %permute_77), kwargs = {})
#   %sigmoid : Tensor "bf16[1, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_46,), kwargs = {})
#   return %addmm_46,%sigmoid
triton_tem_fused__to_copy_addmm_relu_sigmoid_t_44 = async_compile.triton('triton_tem_fused__to_copy_addmm_relu_sigmoid_t_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=1,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr1': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_relu_sigmoid_t_44', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_relu_sigmoid_t_44(in_ptr0, arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1
    N = 4
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
        a = tl.load(A + ((tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 4*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 4*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tmp2 = tl.sigmoid(tmp1)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp2, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/vl/cvlj2j7ana7j2aiardkwpog43umbslkzeydyth3u2yycdxrfyqo2.py
# Topologically Sorted Source Nodes: [linear_48, x_50, x_51], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   linear_48 => addmm_48, convert_element_type_281, convert_element_type_282, permute_79
#   x_50 => relu_12
#   x_51 => relu_13
# Graph fragment:
#   %convert_element_type_281 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type_281]
#   %relu_12 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=relu_12]
#   %convert_element_type_282 : Tensor "bf16[256, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_282]
#   %addmm_48 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_48]
#   %convert_element_type_281 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg130_1, torch.bfloat16), kwargs = {})
#   %relu_12 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_47,), kwargs = {})
#   %convert_element_type_282 : Tensor "bf16[256, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg129_1, torch.bfloat16), kwargs = {})
#   %permute_79 : Tensor "bf16[256, 256][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_282, [1, 0]), kwargs = {})
#   %addmm_48 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_281, %relu_12, %permute_79), kwargs = {})
#   %relu_13 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_48,), kwargs = {})
#   return %addmm_48,%relu_13
triton_tem_fused__to_copy_addmm_relu_t_45 = async_compile.triton('triton_tem_fused__to_copy_addmm_relu_t_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=2,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr1': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_relu_t_45', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_relu_t_45(in_ptr0, arg_A, arg_B, out_ptr1):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1
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
        a = tl.load(A + ((tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))

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
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp3, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/is/cisg6jxrcdkeyyfnbcrbpclxm4c7z2wg4idfhczvarwczcohkm6i.py
# Topologically Sorted Source Nodes: [x_52, x_51], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   x_51 => relu_13
#   x_52 => addmm_49, convert_element_type_286, convert_element_type_287, permute_80
# Graph fragment:
#   %arg132_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg132_1]
#   %convert_element_type_286 : Tensor "bf16[1][1]cuda:0" = PlaceHolder[target=convert_element_type_286]
#   %convert_element_type_286 : Tensor "bf16[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg132_1, torch.bfloat16), kwargs = {})
#   %relu_13 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_48,), kwargs = {})
#   %convert_element_type_287 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg131_1, torch.bfloat16), kwargs = {})
#   %permute_80 : Tensor "bf16[256, 1][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_287, [1, 0]), kwargs = {})
#   %addmm_49 : Tensor "bf16[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_286, %relu_13, %permute_80), kwargs = {})
#   return %convert_element_type_286,%buf297
triton_poi_fused__to_copy_addmm_relu_t_46 = async_compile.triton('triton_poi_fused__to_copy_addmm_relu_t_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_relu_t_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_relu_t_46(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tmp1.to(tl.float32)
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp2, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/36/c36mml7o74gpfcx63ebzbrnm4vchu7hl3swde76stopkzivrmx5g.py
# Topologically Sorted Source Nodes: [x_52, x_51], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   x_51 => relu_13
#   x_52 => addmm_49, convert_element_type_286, convert_element_type_287, permute_80
# Graph fragment:
#   %buf297 : Tensor "bf16[1, 1][1, 1]cuda:0" = PlaceHolder[target=buf297]
#   %relu_13 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=relu_13]
#   %convert_element_type_287 : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_287]
#   %convert_element_type_286 : Tensor "bf16[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg132_1, torch.bfloat16), kwargs = {})
#   %relu_13 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_48,), kwargs = {})
#   %convert_element_type_287 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg131_1, torch.bfloat16), kwargs = {})
#   %permute_80 : Tensor "bf16[256, 1][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_287, [1, 0]), kwargs = {})
#   %addmm_49 : Tensor "bf16[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_286, %relu_13, %permute_80), kwargs = {})
#   return %addmm_49
triton_tem_fused__to_copy_addmm_relu_t_47 = async_compile.triton('triton_tem_fused__to_copy_addmm_relu_t_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=1,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_addmm_relu_t_47', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_addmm_relu_t_47(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 1
    N = 1
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 0

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
        a = tl.load(A + ((tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_K])).broadcast_to(xindex.shape)))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_m + idx_n
        b = tl.load(B + ((tl.broadcast_to(idx_m, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_m + idx_n
    tmp0 = tl.load(in_ptr0 + (0)).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [BLOCK_M, BLOCK_N])
    tmp2 = acc + tmp1
    tl.store(out_ptr0 + (tl.full([BLOCK_M, BLOCK_N], 0, tl.int32).broadcast_to(BLOCK_M, BLOCK_N)), tmp2, None)
''', device_str='cuda')

def partition_0(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg4_1, arg5_1, arg6_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 256), (256, 1))
    assert_size_stride(arg1_1, (1, 256), (256, 1))
    assert_size_stride(arg2_1, (4, 256), (256, 1))
    assert_size_stride(arg3_1, (1, 2, 256), (512, 256, 1))
    assert_size_stride(arg7_1, (256, 256), (256, 1))
    assert_size_stride(arg8_1, (256, ), (1, ))
    assert_size_stride(arg9_1, (256, 256), (256, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, 256), (256, 1))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, 256), (256, 1))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (128, 256), (256, 1))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (1, 256, 64, 64), (256, 1, 16384, 256))
    assert_size_stride(arg5_1, (1, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(arg6_1, (1, 256, 64, 64), (256, 1, 16384, 256))
    assert_size_stride(arg19_1, (128, 256), (256, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, 256), (256, 1))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (256, 128), (128, 1))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (2048, 256), (256, 1))
    assert_size_stride(arg28_1, (2048, ), (1, ))
    assert_size_stride(arg29_1, (256, 2048), (2048, 1))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (128, 256), (256, 1))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, 256), (256, 1))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, 256), (256, 1))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (256, 128), (128, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, 256), (256, 1))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, 256), (256, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, 256), (256, 1))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (256, 256), (256, 1))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (128, 256), (256, 1))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, 256), (256, 1))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, 256), (256, 1))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (256, 128), (128, 1))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (2048, 256), (256, 1))
    assert_size_stride(arg64_1, (2048, ), (1, ))
    assert_size_stride(arg65_1, (256, 2048), (2048, 1))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (128, 256), (256, 1))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, 256), (256, 1))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, 256), (256, 1))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (256, 128), (128, 1))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg77_1, (256, ), (1, ))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg79_1, (128, 256), (256, 1))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, 256), (256, 1))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (128, 256), (256, 1))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (256, 128), (128, 1))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (256, 256), (256, 1))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (256, 256), (256, 1))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (32, 256), (256, 1))
    assert_size_stride(arg102_1, (32, ), (1, ))
    assert_size_stride(arg103_1, (256, 256), (256, 1))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, 256), (256, 1))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (32, 256), (256, 1))
    assert_size_stride(arg108_1, (32, ), (1, ))
    assert_size_stride(arg109_1, (256, 256), (256, 1))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, 256), (256, 1))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (32, 256), (256, 1))
    assert_size_stride(arg114_1, (32, ), (1, ))
    assert_size_stride(arg115_1, (256, 256), (256, 1))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (256, 256), (256, 1))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (32, 256), (256, 1))
    assert_size_stride(arg120_1, (32, ), (1, ))
    assert_size_stride(arg89_1, (256, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(arg90_1, (64, ), (1, ))
    assert_size_stride(arg91_1, (1, 64, 128, 128), (64, 1, 8192, 64))
    assert_size_stride(arg92_1, (64, ), (1, ))
    assert_size_stride(arg93_1, (64, ), (1, ))
    assert_size_stride(arg94_1, (64, 32, 2, 2), (128, 4, 2, 1))
    assert_size_stride(arg95_1, (32, ), (1, ))
    assert_size_stride(arg96_1, (1, 32, 256, 256), (2097152, 65536, 256, 1))
    assert_size_stride(arg121_1, (256, 256), (256, 1))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, 256), (256, 1))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (4, 256), (256, 1))
    assert_size_stride(arg126_1, (4, ), (1, ))
    assert_size_stride(arg127_1, (256, 256), (256, 1))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (256, 256), (256, 1))
    assert_size_stride(arg130_1, (256, ), (1, ))
    assert_size_stride(arg131_1, (1, 256), (256, 1))
    assert_size_stride(arg132_1, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 8, 256), (2048, 256, 1), torch.bfloat16)
        buf4 = empty_strided_cuda((1, 8, 256), (2048, 256, 1), torch.bfloat16)
        buf8 = empty_strided_cuda((1, 8, 256), (2048, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [output_tokens, unsqueeze, output_tokens_1, tokens, q, k, v], Original ATen: [aten.cat, aten.unsqueeze, aten.expand, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_expand_unsqueeze_0.run(arg0_1, arg1_1, arg2_1, arg3_1, buf0, buf4, buf8, 2048, stream=stream0)
        buf1 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg7_1, buf1, 65536, stream=stream0)
        del arg7_1
        buf2 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg8_1, buf2, 256, stream=stream0)
        del arg8_1
        buf3 = empty_strided_cuda((8, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_3.run(buf2, buf0, buf1, buf3, 4, 1, 1, stream=stream0)
        buf5 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [k], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg9_1, buf5, 65536, stream=stream0)
        del arg9_1
        buf6 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [k], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg10_1, buf6, 256, stream=stream0)
        del arg10_1
        buf7 = reinterpret_tensor(buf0, (8, 256), (256, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [k], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_3.run(buf6, buf4, buf5, buf7, 4, 1, 1, stream=stream0)
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg11_1, buf9, 65536, stream=stream0)
        del arg11_1
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg12_1, buf10, 256, stream=stream0)
        del arg12_1
        buf11 = reinterpret_tensor(buf4, (8, 256), (256, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_3.run(buf10, buf8, buf9, buf11, 4, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [q, x, q_1, k, x_1, k_1, v, x_2, v_1, out], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf12 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf3, (1, 8, 8, 32), (2048, 32, 256, 1), 0), reinterpret_tensor(buf7, (1, 8, 8, 32), (2048, 32, 256, 1), 0), reinterpret_tensor(buf11, (1, 8, 8, 32), (2048, 32, 256, 1), 0), scale=0.17677669529663687)
        buf13 = buf12[0]
        assert_size_stride(buf13, (1, 8, 8, 32), (2048, 32, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf13, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf12
        buf18 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg13_1, buf18, 65536, stream=stream0)
        del arg13_1
        buf19 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg14_1, buf19, 256, stream=stream0)
        del arg14_1
        buf20 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [out_2, x_3, out_1], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_3.run(buf19, buf13, buf18, buf20, 4, 1, 1, stream=stream0)
        buf24 = empty_strided_cuda((1, 8, 256), (2048, 256, 1), torch.float32)
        buf26 = reinterpret_tensor(buf13, (1, 8, 256), (2048, 256, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [output_tokens, unsqueeze, output_tokens_1, tokens, out_2, queries, q_2, q_3], Original ATen: [aten.cat, aten.unsqueeze, aten.expand, aten.view, aten._to_copy, aten.native_layer_norm, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_4.run(buf20, arg15_1, arg16_1, arg0_1, arg1_1, arg2_1, arg3_1, buf24, buf26, 8, 256, stream=stream0)
        del arg15_1
        del arg16_1
        buf27 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg17_1, buf27, 32768, stream=stream0)
        del arg17_1
        buf28 = empty_strided_cuda((128, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg18_1, buf28, 128, stream=stream0)
        del arg18_1
        buf29 = empty_strided_cuda((8, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_3], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_7.run(buf28, buf26, buf27, buf29, 4, 1, 1, stream=stream0)
        buf30 = empty_strided_cuda((1, 4096, 256), (1048576, 256, 1), torch.bfloat16)
        buf62 = empty_strided_cuda((1, 4096, 256), (1048576, 256, 1), torch.bfloat16)
        buf34 = empty_strided_cuda((1, 4096, 256), (1048576, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [src, flatten, image_embedding, pos_src, flatten_1, image_pe, k_2, k_3, v_2, k_5, q_6], Original ATen: [aten.add, aten.view, aten.permute, aten.unsqueeze, aten.clone, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_permute_unsqueeze_view_8.run(arg4_1, arg5_1, arg6_1, buf30, buf62, buf34, 4096, 256, stream=stream0)
        buf31 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [k_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg19_1, buf31, 32768, stream=stream0)
        del arg19_1
        buf32 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [k_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg20_1, buf32, 128, stream=stream0)
        del arg20_1
        buf33 = empty_strided_cuda((4096, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [k_3, src, flatten, image_embedding, pos_src, flatten_1, image_pe, k_2], Original ATen: [aten._to_copy, aten.add, aten.view, aten.permute, aten.unsqueeze, aten.clone, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9.run(buf32, buf30, buf31, buf33, 64, 1, 1, stream=stream0)
        buf35 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [v_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg21_1, buf35, 32768, stream=stream0)
        del arg21_1
        buf36 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [v_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg22_1, buf36, 128, stream=stream0)
        del arg22_1
        buf37 = empty_strided_cuda((4096, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [src, flatten, image_embedding, v_2], Original ATen: [aten.add, aten.view, aten.permute, aten._to_copy, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9.run(buf36, buf34, buf35, buf37, 64, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [q_3, x_4, q_4, k_3, x_5, k_4, v_2, x_6, v_3, out_3], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf38 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf29, (1, 8, 8, 16), (1024, 16, 128, 1), 0), reinterpret_tensor(buf33, (1, 8, 4096, 16), (524288, 16, 128, 1), 0), reinterpret_tensor(buf37, (1, 8, 4096, 16), (524288, 16, 128, 1), 0), scale=0.25)
        del buf33
        buf39 = buf38[0]
        assert_size_stride(buf39, (1, 8, 8, 16), (1024, 16, 128, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf39, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf38
        buf44 = reinterpret_tensor(buf35, (256, 128), (128, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg23_1, buf44, 32768, stream=stream0)
        del arg23_1
        buf45 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg24_1, buf45, 256, stream=stream0)
        del arg24_1
        buf46 = reinterpret_tensor(buf26, (8, 256), (256, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [out_5, x_7, out_4], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_transpose_view_10.run(buf45, buf39, buf44, buf46, 4, 1, 1, stream=stream0)
        buf50 = buf24; del buf24  # reuse
        buf51 = reinterpret_tensor(buf20, (1, 8, 256), (2048, 256, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [out_5, queries_1, queries_2, linear_8], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_view_11.run(buf50, buf46, arg25_1, arg26_1, buf51, 8, 256, stream=stream0)
        del arg25_1
        del arg26_1
        buf52 = reinterpret_tensor(buf37, (2048, 256), (256, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(arg27_1, buf52, 524288, stream=stream0)
        del arg27_1
        buf53 = reinterpret_tensor(buf46, (2048, ), (1, ), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(arg28_1, buf53, 2048, stream=stream0)
        del arg28_1
        buf54 = empty_strided_cuda((8, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_14.run(buf53, buf51, buf52, buf54, 64, 1, 1, stream=stream0)
        buf55 = reinterpret_tensor(buf54, (1, 8, 2048), (16384, 2048, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [linear_8, x_8], Original ATen: [aten.view, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_view_15.run(buf55, 16384, stream=stream0)
        buf56 = reinterpret_tensor(buf52, (256, 2048), (2048, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(arg29_1, buf56, 524288, stream=stream0)
        del arg29_1
        buf57 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg30_1, buf57, 256, stream=stream0)
        del arg30_1
        buf58 = reinterpret_tensor(buf53, (8, 256), (256, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [x_9, linear_8, x_8], Original ATen: [aten._to_copy, aten.view, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_view_16.run(buf57, buf55, buf56, buf58, 8, 1, 1, stream=stream0)
        buf66 = buf50; del buf50  # reuse
        buf68 = buf51; del buf51  # reuse
        buf89 = reinterpret_tensor(buf3, (1, 8, 256), (2048, 256, 1), 0); del buf3  # reuse
        buf93 = reinterpret_tensor(buf11, (1, 8, 256), (2048, 256, 1), 0); del buf11  # reuse
        buf72 = buf8; del buf8  # reuse
        buf97 = empty_strided_cuda((1, 8, 256), (2048, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [output_tokens, unsqueeze, output_tokens_1, tokens, x_9, queries_3, queries_4, q_5, k_6, v_4, q_8, q_9, k_8, v_6], Original ATen: [aten.cat, aten.unsqueeze, aten.expand, aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_17.run(buf66, buf58, arg31_1, arg32_1, arg0_1, arg1_1, arg2_1, arg3_1, buf68, buf89, buf93, buf72, buf97, 8, 256, stream=stream0)
        del arg31_1
        del arg32_1
        del buf58
        buf63 = reinterpret_tensor(buf44, (128, 256), (256, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [q_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg33_1, buf63, 32768, stream=stream0)
        del arg33_1
        buf64 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [q_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg34_1, buf64, 128, stream=stream0)
        del arg34_1
        buf65 = reinterpret_tensor(buf56, (4096, 128), (128, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [src, flatten, image_embedding, pos_src, flatten_1, image_pe, q_6, k_5], Original ATen: [aten.add, aten.view, aten.permute, aten.unsqueeze, aten.clone, aten._to_copy, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9.run(buf64, buf62, buf63, buf65, 64, 1, 1, stream=stream0)
        buf69 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [k_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg35_1, buf69, 32768, stream=stream0)
        del arg35_1
        buf70 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [k_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg36_1, buf70, 128, stream=stream0)
        del arg36_1
        buf71 = reinterpret_tensor(buf39, (8, 128), (128, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [k_6], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_7.run(buf70, buf68, buf69, buf71, 4, 1, 1, stream=stream0)
        del buf68
        buf73 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg37_1, buf73, 32768, stream=stream0)
        del arg37_1
        buf74 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg38_1, buf74, 128, stream=stream0)
        del arg38_1
        buf75 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_7.run(buf74, buf72, buf73, buf75, 4, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [q_6, x_10, q_7, k_6, x_11, k_7, v_4, x_12, v_5, out_6], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf76 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf65, (1, 8, 4096, 16), (524288, 16, 128, 1), 0), reinterpret_tensor(buf71, (1, 8, 8, 16), (1024, 16, 128, 1), 0), reinterpret_tensor(buf75, (1, 8, 8, 16), (1024, 16, 128, 1), 0), scale=0.25)
        del buf71
        buf77 = buf76[0]
        assert_size_stride(buf77, (1, 8, 4096, 16), (524288, 16, 128, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf77, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf76
        buf82 = reinterpret_tensor(buf73, (256, 128), (128, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg39_1, buf82, 32768, stream=stream0)
        del arg39_1
        buf83 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg40_1, buf83, 256, stream=stream0)
        del arg40_1
        buf84 = reinterpret_tensor(buf62, (4096, 256), (256, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [out_8, x_13, out_7], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_transpose_view_18.run(buf83, buf77, buf82, buf84, 256, 1, 1, stream=stream0)
        buf119 = empty_strided_cuda((1, 4096, 256), (1048576, 256, 1), torch.float32)
        buf120 = buf34; del buf34  # reuse
        buf152 = buf30; del buf30  # reuse
        buf124 = empty_strided_cuda((1, 4096, 256), (1048576, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [src, flatten, image_embedding, pos_src, flatten_1, image_pe, out_8, keys, keys_1, k_10, k_11, v_8, k_13, q_15], Original ATen: [aten.add, aten.view, aten.permute, aten.unsqueeze, aten.clone, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_clone_native_layer_norm_permute_unsqueeze_view_19.run(arg4_1, arg5_1, buf84, arg41_1, arg42_1, arg6_1, buf119, buf120, buf152, buf124, 4096, 256, stream=stream0)
        del arg41_1
        del arg42_1
        del arg4_1
        del arg5_1
        buf90 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [q_9], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg43_1, buf90, 65536, stream=stream0)
        del arg43_1
        buf91 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [q_9], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg44_1, buf91, 256, stream=stream0)
        del arg44_1
        buf92 = reinterpret_tensor(buf72, (8, 256), (256, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [q_9], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_3.run(buf91, buf89, buf90, buf92, 4, 1, 1, stream=stream0)
        buf94 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [k_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg45_1, buf94, 65536, stream=stream0)
        del arg45_1
        buf95 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [k_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg46_1, buf95, 256, stream=stream0)
        del arg46_1
        buf96 = reinterpret_tensor(buf89, (8, 256), (256, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [k_8], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_3.run(buf95, buf93, buf94, buf96, 4, 1, 1, stream=stream0)
        buf98 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [v_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg47_1, buf98, 65536, stream=stream0)
        del arg47_1
        buf99 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [v_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg48_1, buf99, 256, stream=stream0)
        del arg48_1
        buf100 = reinterpret_tensor(buf93, (8, 256), (256, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [v_6], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_3.run(buf99, buf97, buf98, buf100, 4, 1, 1, stream=stream0)
        del buf97
        # Topologically Sorted Source Nodes: [q_9, x_14, q_10, k_8, x_15, k_9, v_6, x_16, v_7, out_9], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf101 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf92, (1, 8, 8, 32), (2048, 32, 256, 1), 0), reinterpret_tensor(buf96, (1, 8, 8, 32), (2048, 32, 256, 1), 0), reinterpret_tensor(buf100, (1, 8, 8, 32), (2048, 32, 256, 1), 0), scale=0.17677669529663687)
        buf102 = buf101[0]
        assert_size_stride(buf102, (1, 8, 8, 32), (2048, 32, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf102, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf101
        buf107 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg49_1, buf107, 65536, stream=stream0)
        del arg49_1
        buf108 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg50_1, buf108, 256, stream=stream0)
        del arg50_1
        buf109 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [out_11, x_17, out_10], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_3.run(buf108, buf102, buf107, buf109, 4, 1, 1, stream=stream0)
        del buf107
        buf113 = buf66; del buf66  # reuse
        buf115 = reinterpret_tensor(buf102, (1, 8, 256), (2048, 256, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [output_tokens, unsqueeze, output_tokens_1, tokens, out_11, queries_5, queries_6, q_11, q_12], Original ATen: [aten.cat, aten.unsqueeze, aten.expand, aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_20.run(buf113, buf109, arg51_1, arg52_1, arg0_1, arg1_1, arg2_1, arg3_1, buf115, 8, 256, stream=stream0)
        del arg51_1
        del arg52_1
        buf116 = reinterpret_tensor(buf82, (128, 256), (256, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [q_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg53_1, buf116, 32768, stream=stream0)
        del arg53_1
        buf117 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [q_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg54_1, buf117, 128, stream=stream0)
        del arg54_1
        buf118 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [q_12], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_7.run(buf117, buf115, buf116, buf118, 4, 1, 1, stream=stream0)
        buf121 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [k_11], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg55_1, buf121, 32768, stream=stream0)
        del arg55_1
        buf122 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [k_11], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg56_1, buf122, 128, stream=stream0)
        del arg56_1
        buf123 = reinterpret_tensor(buf77, (4096, 128), (128, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [pos_src, flatten_1, image_pe, k_11, k_10], Original ATen: [aten.unsqueeze, aten.clone, aten.view, aten.permute, aten._to_copy, aten.add, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9.run(buf122, buf120, buf121, buf123, 64, 1, 1, stream=stream0)
        buf125 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [v_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg57_1, buf125, 32768, stream=stream0)
        del arg57_1
        buf126 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [v_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg58_1, buf126, 128, stream=stream0)
        del arg58_1
        buf127 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [v_8], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9.run(buf126, buf124, buf125, buf127, 64, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [q_12, x_18, q_13, k_11, x_19, k_12, v_8, x_20, v_9, out_12], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf128 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf118, (1, 8, 8, 16), (1024, 16, 128, 1), 0), reinterpret_tensor(buf123, (1, 8, 4096, 16), (524288, 16, 128, 1), 0), reinterpret_tensor(buf127, (1, 8, 4096, 16), (524288, 16, 128, 1), 0), scale=0.25)
        del buf123
        buf129 = buf128[0]
        assert_size_stride(buf129, (1, 8, 8, 16), (1024, 16, 128, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf129, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf128
        buf134 = reinterpret_tensor(buf125, (256, 128), (128, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg59_1, buf134, 32768, stream=stream0)
        del arg59_1
        buf135 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg60_1, buf135, 256, stream=stream0)
        del arg60_1
        buf136 = reinterpret_tensor(buf115, (8, 256), (256, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [out_14, x_21, out_13], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_transpose_view_10.run(buf135, buf129, buf134, buf136, 4, 1, 1, stream=stream0)
        buf140 = buf113; del buf113  # reuse
        buf141 = reinterpret_tensor(buf109, (1, 8, 256), (2048, 256, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [out_14, queries_7, queries_8, linear_22], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_view_11.run(buf140, buf136, arg61_1, arg62_1, buf141, 8, 256, stream=stream0)
        del arg61_1
        del arg62_1
        buf142 = reinterpret_tensor(buf127, (2048, 256), (256, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(arg63_1, buf142, 524288, stream=stream0)
        del arg63_1
        buf143 = reinterpret_tensor(buf136, (2048, ), (1, ), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(arg64_1, buf143, 2048, stream=stream0)
        del arg64_1
        buf145 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [linear_22, x_22], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_view_21.run(buf143, buf141, buf142, buf145, 64, 1, 1, stream=stream0)
        buf146 = reinterpret_tensor(buf142, (256, 2048), (2048, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(arg65_1, buf146, 524288, stream=stream0)
        del arg65_1
        buf147 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg66_1, buf147, 256, stream=stream0)
        del arg66_1
        buf148 = reinterpret_tensor(buf143, (8, 256), (256, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_23, linear_22, x_22], Original ATen: [aten._to_copy, aten.view, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_view_16.run(buf147, buf145, buf146, buf148, 8, 1, 1, stream=stream0)
        del buf145
        buf156 = buf140; del buf140  # reuse
        buf158 = buf141; del buf141  # reuse
        buf179 = reinterpret_tensor(buf92, (1, 8, 256), (2048, 256, 1), 0); del buf92  # reuse
        buf162 = reinterpret_tensor(buf100, (1, 8, 256), (2048, 256, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [output_tokens, unsqueeze, output_tokens_1, tokens, x_23, queries_9, queries_10, q_14, k_14, v_10, q_17, q_18], Original ATen: [aten.cat, aten.unsqueeze, aten.expand, aten.view, aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_cat_expand_native_layer_norm_unsqueeze_view_22.run(buf156, buf148, arg67_1, arg68_1, arg0_1, arg1_1, arg2_1, arg3_1, buf158, buf179, buf162, 8, 256, stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg67_1
        del arg68_1
        del buf148
        buf153 = reinterpret_tensor(buf134, (128, 256), (256, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [q_15], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg69_1, buf153, 32768, stream=stream0)
        del arg69_1
        buf154 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [q_15], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg70_1, buf154, 128, stream=stream0)
        del arg70_1
        buf155 = reinterpret_tensor(buf146, (4096, 128), (128, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [pos_src, flatten_1, image_pe, q_15, k_13], Original ATen: [aten.unsqueeze, aten.clone, aten.view, aten.permute, aten._to_copy, aten.add, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9.run(buf154, buf152, buf153, buf155, 64, 1, 1, stream=stream0)
        buf159 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [k_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg71_1, buf159, 32768, stream=stream0)
        del arg71_1
        buf160 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [k_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg72_1, buf160, 128, stream=stream0)
        del arg72_1
        buf161 = reinterpret_tensor(buf129, (8, 128), (128, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [k_14], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_7.run(buf160, buf158, buf159, buf161, 4, 1, 1, stream=stream0)
        del buf158
        buf163 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [v_10], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg73_1, buf163, 32768, stream=stream0)
        del arg73_1
        buf164 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [v_10], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg74_1, buf164, 128, stream=stream0)
        del arg74_1
        buf165 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [v_10], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_7.run(buf164, buf162, buf163, buf165, 4, 1, 1, stream=stream0)
        del buf162
        del buf164
        # Topologically Sorted Source Nodes: [q_15, x_24, q_16, k_14, x_25, k_15, v_10, x_26, v_11, out_15], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf166 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf155, (1, 8, 4096, 16), (524288, 16, 128, 1), 0), reinterpret_tensor(buf161, (1, 8, 8, 16), (1024, 16, 128, 1), 0), reinterpret_tensor(buf165, (1, 8, 8, 16), (1024, 16, 128, 1), 0), scale=0.25)
        del buf155
        del buf161
        del buf165
        buf167 = buf166[0]
        assert_size_stride(buf167, (1, 8, 4096, 16), (524288, 16, 128, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf167, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf166
        buf172 = reinterpret_tensor(buf163, (256, 128), (128, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg75_1, buf172, 32768, stream=stream0)
        del arg75_1
        buf173 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg76_1, buf173, 256, stream=stream0)
        del arg76_1
        buf174 = reinterpret_tensor(buf152, (4096, 256), (256, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [out_17, x_27, out_16], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_transpose_view_18.run(buf173, buf167, buf172, buf174, 256, 1, 1, stream=stream0)
        del buf167
        del buf172
        del buf173
        buf183 = buf119; del buf119  # reuse
        buf184 = buf124; del buf124  # reuse
        buf188 = buf120; del buf120  # reuse
        buf258 = reinterpret_tensor(buf84, (1, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [pos_src, flatten_1, image_pe, out_17, keys_2, keys_3, k_16, k_17, v_12, transpose_28, src_1, conv_transpose2d], Original ATen: [aten.unsqueeze, aten.clone, aten.view, aten.permute, aten.add, aten.native_layer_norm, aten._to_copy, aten.transpose, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_clone_convolution_native_layer_norm_permute_transpose_unsqueeze_view_23.run(buf183, buf174, arg77_1, arg78_1, arg6_1, buf184, buf188, buf258, 4096, 256, stream=stream0)
        del arg6_1
        del arg77_1
        del arg78_1
        del buf174
        del buf183
        buf180 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_18], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg79_1, buf180, 32768, stream=stream0)
        del arg79_1
        buf181 = empty_strided_cuda((128, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_18], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg80_1, buf181, 128, stream=stream0)
        del arg80_1
        buf182 = empty_strided_cuda((8, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q_18], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_view_7.run(buf181, buf179, buf180, buf182, 4, 1, 1, stream=stream0)
        buf185 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [k_17], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg81_1, buf185, 32768, stream=stream0)
        del arg81_1
        buf186 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [k_17], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg82_1, buf186, 128, stream=stream0)
        del arg82_1
        buf187 = empty_strided_cuda((4096, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [pos_src, flatten_1, image_pe, k_17, k_16], Original ATen: [aten.unsqueeze, aten.clone, aten.view, aten.permute, aten._to_copy, aten.add, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9.run(buf186, buf184, buf185, buf187, 64, 1, 1, stream=stream0)
        del buf184
        buf189 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [v_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg83_1, buf189, 32768, stream=stream0)
        del arg83_1
        buf190 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [v_12], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(arg84_1, buf190, 128, stream=stream0)
        del arg84_1
        buf191 = empty_strided_cuda((4096, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [v_12], Original ATen: [aten._to_copy, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_clone_permute_t_unsqueeze_view_9.run(buf190, buf188, buf189, buf191, 64, 1, 1, stream=stream0)
        del buf188
        # Topologically Sorted Source Nodes: [q_18, x_28, q_19, k_17, x_29, k_18, v_12, x_30, v_13, out_18], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf192 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf182, (1, 8, 8, 16), (1024, 16, 128, 1), 0), reinterpret_tensor(buf187, (1, 8, 4096, 16), (524288, 16, 128, 1), 0), reinterpret_tensor(buf191, (1, 8, 4096, 16), (524288, 16, 128, 1), 0), scale=0.25)
        del buf182
        del buf187
        del buf191
        buf193 = buf192[0]
        assert_size_stride(buf193, (1, 8, 8, 16), (1024, 16, 128, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf193, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf192
        buf198 = reinterpret_tensor(buf189, (256, 128), (128, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(arg85_1, buf198, 32768, stream=stream0)
        del arg85_1
        buf199 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg86_1, buf199, 256, stream=stream0)
        del arg86_1
        buf200 = reinterpret_tensor(buf179, (8, 256), (256, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [out_20, x_31, out_19], Original ATen: [aten._to_copy, aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_t_transpose_view_10.run(buf199, buf193, buf198, buf200, 4, 1, 1, stream=stream0)
        del buf198
        buf204 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [out_20, queries_11, queries_12], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_view_24.run(buf204, buf200, arg87_1, arg88_1, 8, 256, stream=stream0)
        del arg87_1
        del arg88_1
        del buf200
        buf205 = reinterpret_tensor(buf199, (1, 256), (256, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [mask_tokens_out, getitem_4, linear_32], Original ATen: [aten.slice, aten.select, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_select_slice_25.run(buf204, buf205, 256, stream=stream0)
        buf206 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg97_1, buf206, 65536, stream=stream0)
        del arg97_1
        buf207 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg98_1, buf207, 256, stream=stream0)
        del arg98_1
        buf208 = empty_strided_cuda((1, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_32, mask_tokens_out, getitem_4], Original ATen: [aten._to_copy, aten.slice, aten.select, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf207, buf205, buf206, buf208, 8, 1, 1, stream=stream0)
        buf209 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf209, 256, stream=stream0)
        buf210 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg99_1, buf210, 65536, stream=stream0)
        del arg99_1
        buf211 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg100_1, buf211, 256, stream=stream0)
        del arg100_1
        buf212 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [linear_33, x_34], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf211, buf209, buf210, buf212, 8, 1, 1, stream=stream0)
        buf213 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf213, 256, stream=stream0)
        buf214 = empty_strided_cuda((32, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_28.run(arg101_1, buf214, 8192, stream=stream0)
        del arg101_1
        buf215 = empty_strided_cuda((32, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(arg102_1, buf215, 32, stream=stream0)
        del arg102_1
        buf216 = empty_strided_cuda((1, 32), (32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_36, x_35], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_30.run(buf215, buf213, buf214, buf216, 1, 1, 1, stream=stream0)
        buf217 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [mask_tokens_out, getitem_5, linear_35], Original ATen: [aten.slice, aten.select, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_select_slice_31.run(buf204, buf217, 256, stream=stream0)
        buf218 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg103_1, buf218, 65536, stream=stream0)
        del arg103_1
        buf219 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg104_1, buf219, 256, stream=stream0)
        del arg104_1
        buf220 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [mask_tokens_out, linear_35, getitem_5], Original ATen: [aten.slice, aten._to_copy, aten.select, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf219, buf217, buf218, buf220, 8, 1, 1, stream=stream0)
        buf221 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf221, 256, stream=stream0)
        buf222 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg105_1, buf222, 65536, stream=stream0)
        del arg105_1
        buf223 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg106_1, buf223, 256, stream=stream0)
        del arg106_1
        buf224 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [linear_36, x_37], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf223, buf221, buf222, buf224, 8, 1, 1, stream=stream0)
        buf225 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf225, 256, stream=stream0)
        buf226 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_28.run(arg107_1, buf226, 8192, stream=stream0)
        del arg107_1
        buf227 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(arg108_1, buf227, 32, stream=stream0)
        del arg108_1
        buf228 = empty_strided_cuda((1, 32), (32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_39, x_38], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_30.run(buf227, buf225, buf226, buf228, 1, 1, 1, stream=stream0)
        buf229 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [mask_tokens_out, getitem_6, linear_38], Original ATen: [aten.slice, aten.select, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_select_slice_32.run(buf204, buf229, 256, stream=stream0)
        buf230 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg109_1, buf230, 65536, stream=stream0)
        del arg109_1
        buf231 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg110_1, buf231, 256, stream=stream0)
        del arg110_1
        buf232 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [mask_tokens_out, linear_38, getitem_6], Original ATen: [aten.slice, aten._to_copy, aten.select, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf231, buf229, buf230, buf232, 8, 1, 1, stream=stream0)
        buf233 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf233, 256, stream=stream0)
        buf234 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg111_1, buf234, 65536, stream=stream0)
        del arg111_1
        buf235 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg112_1, buf235, 256, stream=stream0)
        del arg112_1
        buf236 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [linear_39, x_40], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf235, buf233, buf234, buf236, 8, 1, 1, stream=stream0)
        buf237 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf237, 256, stream=stream0)
        buf238 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_28.run(arg113_1, buf238, 8192, stream=stream0)
        del arg113_1
        buf239 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(arg114_1, buf239, 32, stream=stream0)
        del arg114_1
        buf240 = empty_strided_cuda((1, 32), (32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_42, x_41], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_30.run(buf239, buf237, buf238, buf240, 1, 1, 1, stream=stream0)
        buf241 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [mask_tokens_out, getitem_7, linear_41], Original ATen: [aten.slice, aten.select, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_select_slice_33.run(buf204, buf241, 256, stream=stream0)
        buf242 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg115_1, buf242, 65536, stream=stream0)
        del arg115_1
        buf243 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg116_1, buf243, 256, stream=stream0)
        del arg116_1
        buf244 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [mask_tokens_out, linear_41, getitem_7], Original ATen: [aten.slice, aten._to_copy, aten.select, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf243, buf241, buf242, buf244, 8, 1, 1, stream=stream0)
        buf245 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf245, 256, stream=stream0)
        buf246 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg117_1, buf246, 65536, stream=stream0)
        del arg117_1
        buf247 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg118_1, buf247, 256, stream=stream0)
        del arg118_1
        buf248 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [linear_42, x_43], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf247, buf245, buf246, buf248, 8, 1, 1, stream=stream0)
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf249, 256, stream=stream0)
        buf250 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_28.run(arg119_1, buf250, 8192, stream=stream0)
        del arg119_1
        buf251 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(arg120_1, buf251, 32, stream=stream0)
        del arg120_1
        buf252 = empty_strided_cuda((1, 32), (32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_45, x_44], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_30.run(buf251, buf249, buf250, buf252, 1, 1, 1, stream=stream0)
        del buf251
        buf257 = reinterpret_tensor(buf190, (1, 128), (128, 1), 0); del buf190  # reuse
        buf253 = reinterpret_tensor(buf257, (1, 32), (128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [hyper_in], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_34.run(buf216, buf253, 32, stream=stream0)
        del buf216
        buf254 = reinterpret_tensor(buf257, (1, 32), (128, 1), 32)  # alias
        # Topologically Sorted Source Nodes: [hyper_in], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_34.run(buf228, buf254, 32, stream=stream0)
        del buf228
        buf255 = reinterpret_tensor(buf257, (1, 32), (128, 1), 64)  # alias
        # Topologically Sorted Source Nodes: [hyper_in], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_34.run(buf240, buf255, 32, stream=stream0)
        del buf240
        buf256 = reinterpret_tensor(buf257, (1, 32), (128, 1), 96)  # alias
        # Topologically Sorted Source Nodes: [hyper_in], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_34.run(buf252, buf256, 32, stream=stream0)
        buf261 = reinterpret_tensor(buf246, (256, 64, 2, 2), (256, 1, 128, 64), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [transpose_28, src_1, conv_transpose2d], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_transpose_view_35.run(arg89_1, buf261, 65536, stream=stream0)
        del arg89_1
        del buf253
        del buf254
        del buf255
        del buf256
        buf260 = empty_strided_cuda((64, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [transpose_28, src_1, conv_transpose2d], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_transpose_view_36.run(arg90_1, buf260, 64, stream=stream0)
        del arg90_1
        # Topologically Sorted Source Nodes: [transpose_28, src_1, conv_transpose2d], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.convolution]
        buf262 = extern_kernels.convolution(buf258, buf261, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (1, 64, 128, 128), (1048576, 1, 8192, 64), 'torch.ops.aten.convolution.default')
        buf266 = reinterpret_tensor(buf258, (1, 64, 128, 128), (1048576, 1, 8192, 64), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [getitem_2, transpose_28, src_1, conv_transpose2d, add_20, u, sub_1, sub, pow_1, s, add_21, sqrt, x_32, mul, getitem_3, x_33, upscaled_embedding, conv_transpose2d_1], Original ATen: [aten.unsqueeze, aten.transpose, aten.view, aten._to_copy, aten.convolution, aten.add, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_transpose_unsqueeze_view_37.run(buf262, buf260, arg91_1, arg92_1, arg93_1, buf266, 16384, 64, stream=stream0)
        del arg91_1
        del arg92_1
        del arg93_1
        del buf260
        del buf262
        buf269 = reinterpret_tensor(buf250, (64, 32, 2, 2), (128, 1, 64, 32), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [upscaled_embedding, conv_transpose2d_1], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_gelu_38.run(arg94_1, buf269, 8192, stream=stream0)
        del arg94_1
        buf268 = reinterpret_tensor(buf252, (32, ), (1, ), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [upscaled_embedding, conv_transpose2d_1], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(arg95_1, buf268, 32, stream=stream0)
        del arg95_1
        # Topologically Sorted Source Nodes: [upscaled_embedding, conv_transpose2d_1], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
        buf270 = extern_kernels.convolution(buf266, buf269, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (1, 32, 256, 256), (2097152, 1, 8192, 32), 'torch.ops.aten.convolution.default')
        del buf266
        del buf269
        buf271 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [upscaled_embedding, conv_transpose2d_1, add_23, upscaled_embedding_1], Original ATen: [aten.gelu, aten._to_copy, aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_convolution_gelu_39.run(buf271, buf268, arg96_1, 65536, 32, stream=stream0)
        del arg96_1
        del buf268
        buf272 = empty_strided_cuda((4, 65536), (65536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hyper_in, matmul, upscaled_embedding, conv_transpose2d_1, add_23, upscaled_embedding_1, view_1], Original ATen: [aten.stack, aten.bmm, aten.gelu, aten._to_copy, aten.convolution, aten.add, aten.view]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_bmm_convolution_gelu_stack_view_40.run(buf257, buf271, buf272, 1024, 1, 1, stream=stream0)
        del buf257
        del buf271
        buf273 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [iou_token_out, linear_44], Original ATen: [aten.select, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_select_41.run(buf204, buf273, 256, stream=stream0)
        buf274 = reinterpret_tensor(buf261, (256, 256), (256, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg121_1, buf274, 65536, stream=stream0)
        del arg121_1
        buf275 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg122_1, buf275, 256, stream=stream0)
        del arg122_1
        buf276 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [linear_44, iou_token_out], Original ATen: [aten._to_copy, aten.select, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf275, buf273, buf274, buf276, 8, 1, 1, stream=stream0)
        buf277 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf277, 256, stream=stream0)
        buf278 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg123_1, buf278, 65536, stream=stream0)
        del arg123_1
        buf279 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg124_1, buf279, 256, stream=stream0)
        del arg124_1
        buf280 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [linear_45, x_46], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf279, buf277, buf278, buf280, 8, 1, 1, stream=stream0)
        buf281 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf281, 256, stream=stream0)
        buf282 = reinterpret_tensor(buf193, (4, 256), (256, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_42.run(arg125_1, buf282, 1024, stream=stream0)
        del arg125_1
        buf283 = empty_strided_cuda((4, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_43.run(arg126_1, buf283, 4, stream=stream0)
        del arg126_1
        buf285 = empty_strided_cuda((1, 4), (4, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_48, x_47, x_49], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_sigmoid_t_44.run(buf283, buf281, buf282, buf285, 1, 1, 1, stream=stream0)
        del buf282
        del buf283
        buf286 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [getitem_8, linear_47], Original ATen: [aten.select, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(buf204, buf286, 256, stream=stream0)
        buf287 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg127_1, buf287, 65536, stream=stream0)
        del arg127_1
        buf288 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg128_1, buf288, 256, stream=stream0)
        del arg128_1
        buf289 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [linear_47, getitem_8], Original ATen: [aten._to_copy, aten.select, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_select_slice_t_26.run(buf288, buf286, buf287, buf289, 8, 1, 1, stream=stream0)
        buf290 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_27.run(buf290, 256, stream=stream0)
        buf291 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg129_1, buf291, 65536, stream=stream0)
        del arg129_1
        buf292 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg130_1, buf292, 256, stream=stream0)
        del arg130_1
        buf294 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [linear_48, x_50, x_51], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_45.run(buf292, buf290, buf291, buf294, 8, 1, 1, stream=stream0)
        del buf290
        del buf291
        buf295 = reinterpret_tensor(buf292, (1, 256), (256, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg131_1, buf295, 256, stream=stream0)
        del arg131_1
        buf296 = empty_strided_cuda((1, ), (1, ), torch.bfloat16)
        buf297 = reinterpret_tensor(buf296, (1, 1), (1, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [x_52, x_51], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_relu_t_46.run(buf297, arg132_1, 1, stream=stream0)
        del arg132_1
        buf298 = empty_strided_cuda((1, 1), (1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_52, x_51], Original ATen: [aten._to_copy, aten.relu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_addmm_relu_t_47.run(buf297, buf294, buf295, buf298, 1, 1, 1, stream=stream0)
        del buf294
        del buf295
        del buf297
    return (buf204, buf272, buf285, buf298, )


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1 = args
        args.clear()
        partition0_args = [arg0_1, arg1_1, arg2_1, arg3_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg4_1, arg5_1, arg6_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1]
        del arg0_1, arg1_1, arg2_1, arg3_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg4_1, arg5_1, arg6_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1
        (buf204, buf272, buf285, buf298) = self.partitions[0](partition0_args)
        del partition0_args
        return (reinterpret_tensor(buf272, (1, 3, 256, 256), (262144, 65536, 256, 1), 65536), reinterpret_tensor(buf285, (1, 3), (4, 1), 1), reinterpret_tensor(buf204, (1, 3, 256), (2048, 256, 1), 768), buf298, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 2, 256), (512, 256, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 256, 64, 64), (256, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1, 256, 64, 64), (256, 1, 16384, 256), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1, 64, 128, 128), (64, 1, 8192, 64), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((64, 32, 2, 2), (128, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1, 32, 256, 256), (2097152, 65536, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((32, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((32, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((32, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((32, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((4, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
