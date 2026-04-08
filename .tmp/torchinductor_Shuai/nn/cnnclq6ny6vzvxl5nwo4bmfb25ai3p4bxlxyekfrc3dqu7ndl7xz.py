# AOT ID: ['3_inference']
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



# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/y5/cy5kjsz5b57zijouhsge6zrte55grofqlunxlkzv534yaqfhomln.py
# Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   x_8 => convert_element_type_19, convert_element_type_20, convert_element_type_21, convolution_5
# Graph fragment:
#   %arg19_1 : Tensor "f32[1, 256, 64, 64][256, 1, 16384, 256]cuda:0" = PlaceHolder[target=arg19_1]
#   %convert_element_type_21 : Tensor "bf16[1, 256, 64, 64][256, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg19_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[256, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg20_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg21_1, torch.bfloat16), kwargs = {})
#   %convolution_5 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_21, %convert_element_type_20, %convert_element_type_19, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused__to_copy_convolution_0 = async_compile.triton('triton_poi_fused__to_copy_convolution_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/hc/chc5nsxvriksl5tjtobevbnild3baurm4lljhctnswzuyovvizdb.py
# Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_8 => convert_element_type_20
# Graph fragment:
#   %arg20_1 : Tensor "f32[256, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=arg20_1]
#   %convert_element_type_20 : Tensor "bf16[256, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg20_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_20
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
# Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_8 => convert_element_type_19
# Graph fragment:
#   %arg21_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg21_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_19
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


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/d7/cd7lvh6tfxjxjouhulsawyhqgz4gmug3rxlrbe4urirpzkpexkex.py
# Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   x_8 => convert_element_type_19, convert_element_type_20, convert_element_type_21, convolution_5
# Graph fragment:
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type_19]
#   %buf0 : Tensor "bf16[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = PlaceHolder[target=buf0]
#   %convert_element_type_20 : Tensor "bf16[256, 256, 1, 1][256, 1, 65536, 65536]cuda:0" = PlaceHolder[target=convert_element_type_20]
#   %convert_element_type_21 : Tensor "bf16[1, 256, 64, 64][256, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg19_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[256, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg20_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg21_1, torch.bfloat16), kwargs = {})
#   %convolution_5 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_21, %convert_element_type_20, %convert_element_type_19, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf3
triton_tem_fused__to_copy_convolution_3 = async_compile.triton('triton_tem_fused__to_copy_convolution_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_3', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_3(in_ptr0, arg_A, arg_B, out_ptr0):
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

    M = 4096
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


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/fe/cfe2wusgalm5sbnpihw5gublune4jjr4fladpyg5naatgy2x73af.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[4, 1, 3, 3][9, 9, 3, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type_1 : Tensor "bf16[4, 1, 3, 3][9, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/ah/cahrsrussfayovigy6fahwi7tgtbfkipdg4ypbc6rki4srfhjz3t.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convert_element_type, convert_element_type_1, convert_element_type_2, convolution
# Graph fragment:
#   %convert_element_type_2 : Tensor "bf16[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=convert_element_type_2]
#   %convert_element_type_1 : Tensor "bf16[4, 1, 3, 3][9, 9, 3, 1]cuda:0" = PlaceHolder[target=convert_element_type_1]
#   %convert_element_type_2 : Tensor "bf16[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[4, 1, 3, 3][9, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convert_element_type : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_2, %convert_element_type_1, %convert_element_type, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf6
triton_tem_fused__to_copy_convolution_5 = async_compile.triton('triton_tem_fused__to_copy_convolution_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=4,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_5', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_5(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 256
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 1
    IN_C = 1
    IN_H = 1024
    IN_W = 1024
    OUT_C = 4
    OUT_H = 512
    OUT_W = 512

    # Strides:
    stride_xn = 1048576
    stride_xc = 1048576
    stride_xh = 1024
    stride_xw = 1
    stride_wc_out = 9
    stride_wc_in = 9
    stride_wh = 3
    stride_ww = 1

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 512*idx_h + 262144*idx_c + 1048576*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_w + 512*idx_h + 262144*idx_c, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/y2/cy236hwgvgsnwi2kkps36ku2xex7ikyyje2cgkdvru6cb5ae2fdp.py
# Topologically Sorted Source Nodes: [input_1, u, sub, pow_1, s], Original ATen: [aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow]
# Source node to ATen node mapping:
#   input_1 => convert_element_type, convert_element_type_1, convert_element_type_2, convolution
#   pow_1 => convert_element_type_3, pow_1
#   s => mean_1
#   sub => sub
#   u => mean
# Graph fragment:
#   %buf6 : Tensor "bf16[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = PlaceHolder[target=buf6]
#   %arg1_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mean : Tensor "bf16[1, 1, 512, 512][262144, 512, 512, 1]cuda:0" = PlaceHolder[target=mean]
#   %convert_element_type_2 : Tensor "bf16[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[4, 1, 3, 3][9, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convert_element_type : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_2, %convert_element_type_1, %convert_element_type, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean : Tensor "bf16[1, 1, 512, 512][262144, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution, [1], True), kwargs = {})
#   %sub : Tensor "bf16[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %mean), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_3, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, 1, 512, 512][262144, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   return %mean,%mean_1
triton_poi_fused__to_copy_convolution_mean_pow_sub_6 = async_compile.triton('triton_poi_fused__to_copy_convolution_mean_pow_sub_6', '''
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
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/fr/cfrlr635tmiuaqpzxxw6xxthownjqhgtibx2a25ecodh4dymt2cl.py
# Topologically Sorted Source Nodes: [getitem, input_1, sub_1, add, sqrt, x, mul, getitem_1, x_1, input_2, input_3], Original ATen: [aten.unsqueeze, aten._to_copy, aten.convolution, aten.sub, aten.add, aten.sqrt, aten.div, aten.mul, aten.gelu]
# Source node to ATen node mapping:
#   add => add
#   getitem => unsqueeze, unsqueeze_1
#   getitem_1 => unsqueeze_2, unsqueeze_3
#   input_1 => convert_element_type, convert_element_type_1, convert_element_type_2, convolution
#   input_2 => add_2, erf, mul_1, mul_2, mul_3
#   input_3 => convert_element_type_6
#   mul => mul
#   sqrt => sqrt
#   sub_1 => sub_1
#   x => div
#   x_1 => add_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %buf6 : Tensor "bf16[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = PlaceHolder[target=buf6]
#   %arg1_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mean : Tensor "bf16[1, 1, 512, 512][262144, 512, 512, 1]cuda:0" = PlaceHolder[target=mean]
#   %mean_1 : Tensor "f32[1, 1, 512, 512][262144, 512, 512, 1]cuda:0" = PlaceHolder[target=mean_1]
#   %arg4_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %add_1 : Tensor "f32[1, 4, 512, 512][1048576, 1, 2048, 4]cuda:0" = PlaceHolder[target=add_1]
#   %unsqueeze : Tensor "f32[4, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[4, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[4, 1, 3, 3][9, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convert_element_type : Tensor "bf16[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_2, %convert_element_type_1, %convert_element_type, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : Tensor "bf16[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %mean), kwargs = {})
#   %add : Tensor "f32[1, 1, 512, 512][262144, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %sqrt : Tensor "f32[1, 1, 512, 512][262144, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %div : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %sqrt), kwargs = {})
#   %mul : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %div), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[4, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[4, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 2), kwargs = {})
#   %add_1 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze_3), kwargs = {})
#   %mul_1 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %mul_2 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_2,), kwargs = {})
#   %add_2 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %add_2), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   return %add_1,%convert_element_type_6
triton_poi_fused__to_copy_add_convolution_div_gelu_mul_sqrt_sub_unsqueeze_7 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_div_gelu_mul_sqrt_sub_unsqueeze_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_div_gelu_mul_sqrt_sub_unsqueeze_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 3670016, 'x': 2097176}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_div_gelu_mul_sqrt_sub_unsqueeze_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 262144*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tl.full([1, 1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt_rn(tmp10)
    tmp12 = (tmp7 / tmp11)
    tmp13 = tmp0 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0.5, tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp19 = tmp15 * tmp18
    tmp20 = libdevice.erf(tmp19)
    tmp21 = tl.full([1, 1], 1.0, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp17 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (x1 + 4*y0), tmp24, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/vo/cvo276r6757qi3j3276gd5wzyqc62idvvml254g4mh3e5nupj7ho.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_3 => convert_element_type_5
# Graph fragment:
#   %arg5_1 : Tensor "f32[16, 4, 3, 3][36, 9, 3, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convert_element_type_5 : Tensor "bf16[16, 4, 3, 3][36, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_5
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 2304, 'x': 2304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4*x2 + 36*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/al/cal7id7ikovyi7ytczbtoaov7hzqu7757twgwekrpfhinidpihzk.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_2 => add_2, erf, mul_1, mul_2, mul_3
#   input_3 => convert_element_type_4, convert_element_type_5, convert_element_type_6, convolution_1
# Graph fragment:
#   %convert_element_type_6 : Tensor "bf16[1, 4, 512, 512][1048576, 1, 2048, 4]cuda:0" = PlaceHolder[target=convert_element_type_6]
#   %convert_element_type_5 : Tensor "bf16[16, 4, 3, 3][36, 1, 12, 4]cuda:0" = PlaceHolder[target=convert_element_type_5]
#   %mul_1 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %mul_2 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_2,), kwargs = {})
#   %add_2 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %add_2), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[16, 4, 3, 3][36, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_6, %convert_element_type_5, %convert_element_type_4, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf12
triton_tem_fused__to_copy_convolution_gelu_9 = async_compile.triton('triton_tem_fused__to_copy_convolution_gelu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=4,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_gelu_9', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_gelu_9(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 256
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 1
    IN_C = 4
    IN_H = 512
    IN_W = 512
    OUT_C = 16
    OUT_H = 256
    OUT_W = 256

    # Strides:
    stride_xn = 1048576
    stride_xc = 1
    stride_xh = 2048
    stride_xw = 4
    stride_wc_out = 36
    stride_wc_in = 1
    stride_wh = 12
    stride_ww = 4

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 256*idx_h + 65536*idx_c + 1048576*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 16*idx_w + 4096*idx_h, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/wj/cwjir5did63oskdfu5qhffyxod73vfl3fptytscdi7pflraup5zr.py
# Topologically Sorted Source Nodes: [getitem_2, input_2, input_3, u_1, sub_3, sub_2, pow_2, s_1, add_2, sqrt_1, x_2, mul_1, getitem_3, x_3, input_4, input_5], Original ATen: [aten.unsqueeze, aten.gelu, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add_2 => add_3
#   getitem_2 => unsqueeze_4, unsqueeze_5
#   getitem_3 => unsqueeze_6, unsqueeze_7
#   input_2 => add_2, erf, mul_1, mul_2, mul_3
#   input_3 => convert_element_type_4, convert_element_type_5, convert_element_type_6, convolution_1
#   input_4 => add_5, erf_1, mul_5, mul_6, mul_7
#   input_5 => convert_element_type_10
#   mul_1 => mul_4
#   pow_2 => convert_element_type_7, pow_2
#   s_1 => mean_3
#   sqrt_1 => sqrt_1
#   sub_2 => sub_2
#   sub_3 => sub_3
#   u_1 => mean_2
#   x_2 => div_1
#   x_3 => add_4
# Graph fragment:
#   %buf12 : Tensor "bf16[1, 16, 256, 256][1048576, 1, 4096, 16]cuda:0" = PlaceHolder[target=buf12]
#   %arg6_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %buf13 : Tensor "f32[1, 1, 256, 256][65536, 65536, 256, 1]cuda:0" = PlaceHolder[target=buf13]
#   %arg7_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %buf14 : Tensor "f32[1, 1, 256, 256][65536, 65536, 256, 1]cuda:0" = PlaceHolder[target=buf14]
#   %arg8_1 : Tensor "f32[16][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %add_4 : Tensor "f32[1, 16, 256, 256][1048576, 1, 4096, 16]cuda:0" = PlaceHolder[target=add_4]
#   %unsqueeze_4 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, 1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 2), kwargs = {})
#   %mul_1 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.5), kwargs = {})
#   %mul_2 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_2,), kwargs = {})
#   %add_2 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %add_2), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[1, 4, 512, 512][1048576, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[16, 4, 3, 3][36, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_6, %convert_element_type_5, %convert_element_type_4, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_2 : Tensor "bf16[1, 1, 256, 256][65536, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_1, [1], True), kwargs = {})
#   %sub_3 : Tensor "bf16[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %mean_2), kwargs = {})
#   %sub_2 : Tensor "bf16[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %mean_2), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_2, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_7, 2), kwargs = {})
#   %mean_3 : Tensor "f32[1, 1, 256, 256][65536, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [1], True), kwargs = {})
#   %add_3 : Tensor "f32[1, 1, 256, 256][65536, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %sqrt_1 : Tensor "f32[1, 1, 256, 256][65536, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_3,), kwargs = {})
#   %div_1 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_3, %sqrt_1), kwargs = {})
#   %mul_4 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %div_1), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg8_1, 1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[16, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, 2), kwargs = {})
#   %add_4 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %unsqueeze_7), kwargs = {})
#   %mul_5 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.5), kwargs = {})
#   %mul_6 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.7071067811865476), kwargs = {})
#   %erf_1 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_5 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_7 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_5), kwargs = {})
#   %convert_element_type_10 : Tensor "bf16[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
#   return %buf13,%buf14,%add_4,%convert_element_type_10
triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_10 = async_compile.triton('triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 6291648}}
)
@triton.jit
def triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 16.0, tl.float32)
    tmp9 = (tmp7 / tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp18 = (tmp16 / tmp8)
    tmp19 = tl.full([1, 1], 1e-06, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp12 / tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1, 1], 0.5, tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp29 = tmp25 * tmp28
    tmp30 = libdevice.erf(tmp29)
    tmp31 = tl.full([1, 1], 1.0, tl.float32)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp27 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 16*x0), tmp34, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/f7/cf7bvbkvxe3v4fy26prueihqd2uzyzzyw3rmh3idgmya6zzo4dhy.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_9
# Graph fragment:
#   %arg9_1 : Tensor "f32[64, 16, 3, 3][144, 9, 3, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %convert_element_type_9 : Tensor "bf16[64, 16, 3, 3][144, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_9
triton_poi_fused__to_copy_11 = async_compile.triton('triton_poi_fused__to_copy_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 36864, 'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 16*x2 + 144*y1), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/d4/cd47xxxnaz4g54qcze2ibemfctxkarbmmsjqcs6yzq44lifyl6nq.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_4 => add_5, erf_1, mul_5, mul_6, mul_7
#   input_5 => convert_element_type_10, convert_element_type_8, convert_element_type_9, convolution_2
# Graph fragment:
#   %convert_element_type_10 : Tensor "bf16[1, 16, 256, 256][1048576, 1, 4096, 16]cuda:0" = PlaceHolder[target=convert_element_type_10]
#   %convert_element_type_9 : Tensor "bf16[64, 16, 3, 3][144, 1, 48, 16]cuda:0" = PlaceHolder[target=convert_element_type_9]
#   %mul_5 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.5), kwargs = {})
#   %mul_6 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.7071067811865476), kwargs = {})
#   %erf_1 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_5 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_7 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_5), kwargs = {})
#   %convert_element_type_10 : Tensor "bf16[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[64, 16, 3, 3][144, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.bfloat16), kwargs = {})
#   %convolution_2 : Tensor "bf16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_10, %convert_element_type_9, %convert_element_type_8, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf18
triton_tem_fused__to_copy_convolution_gelu_12 = async_compile.triton('triton_tem_fused__to_copy_convolution_gelu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_gelu_12', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_gelu_12(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 1
    IN_C = 16
    IN_H = 256
    IN_W = 256
    OUT_C = 64
    OUT_H = 128
    OUT_W = 128

    # Strides:
    stride_xn = 1048576
    stride_xc = 1
    stride_xh = 4096
    stride_xw = 16
    stride_wc_out = 144
    stride_wc_in = 1
    stride_wh = 48
    stride_ww = 16

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 128*idx_h + 16384*idx_c + 1048576*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 64*idx_w + 8192*idx_h, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/xw/cxw5gbe52nl3oepsudreqt74woizuw2ymf5mgvpvkbmmifxxjbu3.py
# Topologically Sorted Source Nodes: [getitem_4, input_4, input_5, u_2, sub_5, sub_4, pow_3, s_2, add_4, sqrt_2, x_4, mul_2, getitem_5, x_5, input_6, input_7], Original ATen: [aten.unsqueeze, aten.gelu, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_6
#   getitem_4 => unsqueeze_8, unsqueeze_9
#   getitem_5 => unsqueeze_10, unsqueeze_11
#   input_4 => add_5, erf_1, mul_5, mul_6, mul_7
#   input_5 => convert_element_type_10, convert_element_type_8, convert_element_type_9, convolution_2
#   input_6 => add_8, erf_2, mul_10, mul_11, mul_9
#   input_7 => convert_element_type_14
#   mul_2 => mul_8
#   pow_3 => convert_element_type_11, pow_3
#   s_2 => mean_5
#   sqrt_2 => sqrt_2
#   sub_4 => sub_4
#   sub_5 => sub_5
#   u_2 => mean_4
#   x_4 => div_2
#   x_5 => add_7
# Graph fragment:
#   %buf18 : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = PlaceHolder[target=buf18]
#   %arg10_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %buf19 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = PlaceHolder[target=buf19]
#   %arg11_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %buf20 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = PlaceHolder[target=buf20]
#   %arg12_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %add_7 : Tensor "f32[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = PlaceHolder[target=add_7]
#   %unsqueeze_8 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg11_1, 1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, 2), kwargs = {})
#   %mul_5 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.5), kwargs = {})
#   %mul_6 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 0.7071067811865476), kwargs = {})
#   %erf_1 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_5 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_7 : Tensor "f32[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_5), kwargs = {})
#   %convert_element_type_10 : Tensor "bf16[1, 16, 256, 256][1048576, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[64, 16, 3, 3][144, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg10_1, torch.bfloat16), kwargs = {})
#   %convolution_2 : Tensor "bf16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_10, %convert_element_type_9, %convert_element_type_8, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_4 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_2, [1], True), kwargs = {})
#   %sub_5 : Tensor "bf16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %mean_4), kwargs = {})
#   %sub_4 : Tensor "bf16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %mean_4), kwargs = {})
#   %convert_element_type_11 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_4, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_11, 2), kwargs = {})
#   %mean_5 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [1], True), kwargs = {})
#   %add_6 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, 1e-06), kwargs = {})
#   %sqrt_2 : Tensor "f32[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %div_2 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_5, %sqrt_2), kwargs = {})
#   %mul_8 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %div_2), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, 1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 2), kwargs = {})
#   %add_7 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_11), kwargs = {})
#   %mul_9 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.5), kwargs = {})
#   %mul_10 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.7071067811865476), kwargs = {})
#   %erf_2 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_10,), kwargs = {})
#   %add_8 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_11 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %add_8), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
#   return %buf19,%buf20,%add_7,%convert_element_type_14
triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_13 = async_compile.triton('triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 6292224}}
)
@triton.jit
def triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 64.0, tl.float32)
    tmp9 = (tmp7 / tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp18 = (tmp16 / tmp8)
    tmp19 = tl.full([1, 1], 1e-06, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp12 / tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1, 1], 0.5, tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp29 = tmp25 * tmp28
    tmp30 = libdevice.erf(tmp29)
    tmp31 = tl.full([1, 1], 1.0, tl.float32)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp27 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 64*x0), tmp34, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/pn/cpn7jlokp2i4erjddp6rm2swb7wlnmx4m6paeo7dbw4tke4hqv6w.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_7 => convert_element_type_13
# Graph fragment:
#   %arg13_1 : Tensor "f32[256, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %convert_element_type_13 : Tensor "bf16[256, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_13
triton_poi_fused__to_copy_14 = async_compile.triton('triton_poi_fused__to_copy_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 589824, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/3z/c3zq6hrvpf3jkbn5rsmpkbien26id6yinpgrjlehmzjl4xstzan3.py
# Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_6 => add_8, erf_2, mul_10, mul_11, mul_9
#   input_7 => convert_element_type_12, convert_element_type_13, convert_element_type_14, convolution_3
# Graph fragment:
#   %convert_element_type_14 : Tensor "bf16[1, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = PlaceHolder[target=convert_element_type_14]
#   %convert_element_type_13 : Tensor "bf16[256, 64, 3, 3][576, 1, 192, 64]cuda:0" = PlaceHolder[target=convert_element_type_13]
#   %mul_9 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.5), kwargs = {})
#   %mul_10 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.7071067811865476), kwargs = {})
#   %erf_2 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_10,), kwargs = {})
#   %add_8 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_11 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %add_8), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[256, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg14_1, torch.bfloat16), kwargs = {})
#   %convolution_3 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_14, %convert_element_type_13, %convert_element_type_12, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf24
triton_tem_fused__to_copy_convolution_gelu_15 = async_compile.triton('triton_tem_fused__to_copy_convolution_gelu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_gelu_15', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_gelu_15(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 1
    IN_C = 64
    IN_H = 128
    IN_W = 128
    OUT_C = 256
    OUT_H = 64
    OUT_W = 64

    # Strides:
    stride_xn = 1048576
    stride_xc = 1
    stride_xh = 8192
    stride_xw = 64
    stride_wc_out = 576
    stride_wc_in = 1
    stride_wh = 192
    stride_ww = 64

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 64*idx_h + 4096*idx_c + 1048576*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 256*idx_w + 16384*idx_h, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/qw/cqwfz2q6c54bniem3dlbq6gwhpgfs5telmli35lesgkffst4gqgh.py
# Topologically Sorted Source Nodes: [getitem_6, input_6, input_7, u_3, sub_7, sub_6, pow_4, s_3, add_6, sqrt_3, x_6, mul_3, getitem_7, x_7, input_8, input_9], Original ATen: [aten.unsqueeze, aten.gelu, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add_6 => add_9
#   getitem_6 => unsqueeze_12, unsqueeze_13
#   getitem_7 => unsqueeze_14, unsqueeze_15
#   input_6 => add_8, erf_2, mul_10, mul_11, mul_9
#   input_7 => convert_element_type_12, convert_element_type_13, convert_element_type_14, convolution_3
#   input_8 => add_11, erf_3, mul_13, mul_14, mul_15
#   input_9 => convert_element_type_16, convert_element_type_17, convert_element_type_18, convolution_4
#   mul_3 => mul_12
#   pow_4 => convert_element_type_15, pow_4
#   s_3 => mean_7
#   sqrt_3 => sqrt_3
#   sub_6 => sub_6
#   sub_7 => sub_7
#   u_3 => mean_6
#   x_6 => div_3
#   x_7 => add_10
# Graph fragment:
#   %buf24 : Tensor "bf16[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = PlaceHolder[target=buf24]
#   %arg14_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %buf25 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0" = PlaceHolder[target=buf25]
#   %arg15_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %buf26 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0" = PlaceHolder[target=buf26]
#   %arg16_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %add_10 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = PlaceHolder[target=add_10]
#   %unsqueeze_12 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, 1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, 2), kwargs = {})
#   %mul_9 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.5), kwargs = {})
#   %mul_10 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.7071067811865476), kwargs = {})
#   %erf_2 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_10,), kwargs = {})
#   %add_8 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_11 : Tensor "f32[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %add_8), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[1, 64, 128, 128][1048576, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[256, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg13_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg14_1, torch.bfloat16), kwargs = {})
#   %convolution_3 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_14, %convert_element_type_13, %convert_element_type_12, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean_6 : Tensor "bf16[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_3, [1], True), kwargs = {})
#   %sub_7 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %mean_6), kwargs = {})
#   %sub_6 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %mean_6), kwargs = {})
#   %convert_element_type_15 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_6, torch.float32), kwargs = {})
#   %pow_4 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_15, 2), kwargs = {})
#   %mean_7 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [1], True), kwargs = {})
#   %add_9 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, 1e-06), kwargs = {})
#   %sqrt_3 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_9,), kwargs = {})
#   %div_3 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_7, %sqrt_3), kwargs = {})
#   %mul_12 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %div_3), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg16_1, 1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, 2), kwargs = {})
#   %add_10 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %unsqueeze_15), kwargs = {})
#   %mul_13 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.5), kwargs = {})
#   %mul_14 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.7071067811865476), kwargs = {})
#   %erf_3 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_14,), kwargs = {})
#   %add_11 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_15 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %add_11), kwargs = {})
#   %convert_element_type_18 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %convert_element_type_17 : Tensor "bf16[256, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg18_1, torch.bfloat16), kwargs = {})
#   %convolution_4 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_18, %convert_element_type_17, %convert_element_type_16, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf25,%buf26,%add_10,%buf28
triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_16 = async_compile.triton('triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 6294528}}
)
@triton.jit
def triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 256.0, tl.float32)
    tmp9 = (tmp7 / tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp18 = (tmp16 / tmp8)
    tmp19 = tl.full([1, 1], 1e-06, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp12 / tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1, 1], 0.5, tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp29 = tmp25 * tmp28
    tmp30 = libdevice.erf(tmp29)
    tmp31 = tl.full([1, 1], 1.0, tl.float32)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp27 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 256*x0), tmp34, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/xs/cxsaazjbbiomgo2ignn565wsamsqupodyj3rsp72zlyn335tv3al.py
# Topologically Sorted Source Nodes: [x_8, input_8, input_9, x_9], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.add]
# Source node to ATen node mapping:
#   input_8 => add_11, erf_3, mul_13, mul_14, mul_15
#   input_9 => convert_element_type_16, convert_element_type_17, convert_element_type_18, convolution_4
#   x_8 => convert_element_type_19, convert_element_type_20, convert_element_type_21, convolution_5
#   x_9 => add_12
# Graph fragment:
#   %buf3 : Tensor "bf16[4096, 256][256, 1]cuda:0" = PlaceHolder[target=buf3]
#   %buf31 : Tensor "bf16[4096, 256][256, 1]cuda:0" = PlaceHolder[target=buf31]
#   %convert_element_type_21 : Tensor "bf16[1, 256, 64, 64][256, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg19_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[256, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg20_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg21_1, torch.bfloat16), kwargs = {})
#   %convolution_5 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_21, %convert_element_type_20, %convert_element_type_19, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_13 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.5), kwargs = {})
#   %mul_14 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.7071067811865476), kwargs = {})
#   %erf_3 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_14,), kwargs = {})
#   %add_11 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_15 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %add_11), kwargs = {})
#   %convert_element_type_18 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %convert_element_type_17 : Tensor "bf16[256, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg18_1, torch.bfloat16), kwargs = {})
#   %convolution_4 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_18, %convert_element_type_17, %convert_element_type_16, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %convolution_4), kwargs = {})
#   return %add_12
triton_poi_fused__to_copy_add_convolution_gelu_17 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_gelu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_gelu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_gelu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/zw/czw43x472bwfv4gouwnqsylxrzp4q3kemsxmu7r37m6zqbwttvan.py
# Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_10 => convert_element_type_23
# Graph fragment:
#   %arg22_1 : Tensor "f32[256, 1, 7, 7][49, 49, 7, 1]cuda:0" = PlaceHolder[target=arg22_1]
#   %convert_element_type_23 : Tensor "bf16[256, 1, 7, 7][49, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg22_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_23
triton_poi_fused__to_copy_18 = async_compile.triton('triton_poi_fused__to_copy_18', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 100352}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/4d/c4dxrr6mduhnikm3ossiz6trqurpy4pf36gqyy56ndceg6fal6co.py
# Topologically Sorted Source Nodes: [getitem_8, x_10, u_4, sub_9, sub_8, pow_5, s_4, add_9, sqrt_4, x_11, mul_4, getitem_9, x_12, x_13, x_14], Original ATen: [aten.unsqueeze, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul, aten.permute]
# Source node to ATen node mapping:
#   add_9 => add_13
#   getitem_8 => unsqueeze_16, unsqueeze_17
#   getitem_9 => unsqueeze_18, unsqueeze_19
#   mul_4 => mul_16
#   pow_5 => convert_element_type_24, pow_5
#   s_4 => mean_9
#   sqrt_4 => sqrt_4
#   sub_8 => sub_8
#   sub_9 => sub_9
#   u_4 => mean_8
#   x_10 => convert_element_type_22, convert_element_type_23, convolution_6
#   x_11 => div_4
#   x_12 => add_14
#   x_13 => permute
#   x_14 => convert_element_type_27
# Graph fragment:
#   %buf34 : Tensor "bf16[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = PlaceHolder[target=buf34]
#   %arg23_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %buf35 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0" = PlaceHolder[target=buf35]
#   %arg24_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %buf36 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0" = PlaceHolder[target=buf36]
#   %arg25_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg25_1]
#   %unsqueeze_16 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, 1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 2), kwargs = {})
#   %convert_element_type_23 : Tensor "bf16[256, 1, 7, 7][49, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg22_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_22 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg23_1, torch.bfloat16), kwargs = {})
#   %convolution_6 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_12, %convert_element_type_23, %convert_element_type_22, [1, 1], [3, 3], [1, 1], False, [0, 0], 256), kwargs = {})
#   %mean_8 : Tensor "bf16[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_6, [1], True), kwargs = {})
#   %sub_9 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %mean_8), kwargs = {})
#   %sub_8 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %mean_8), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_8, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_24, 2), kwargs = {})
#   %mean_9 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [1], True), kwargs = {})
#   %add_13 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_9, 1e-06), kwargs = {})
#   %sqrt_4 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_13,), kwargs = {})
#   %div_4 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_9, %sqrt_4), kwargs = {})
#   %mul_16 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %div_4), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, 1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %add_14 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %unsqueeze_19), kwargs = {})
#   %permute : Tensor "f32[1, 64, 64, 256][1048576, 64, 1, 4096]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_14, [0, 2, 3, 1]), kwargs = {})
#   %convert_element_type_27 : Tensor "bf16[1, 64, 64, 256][1048576, 64, 1, 4096]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute, torch.bfloat16), kwargs = {})
#   return %buf35,%buf36,%expand
triton_per_fused__to_copy_add_convolution_div_mean_mul_permute_pow_sqrt_sub_unsqueeze_19 = async_compile.triton('triton_per_fused__to_copy_add_convolution_div_mean_mul_permute_pow_sqrt_sub_unsqueeze_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_convolution_div_mean_mul_permute_pow_sqrt_sub_unsqueeze_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 6294528}}
)
@triton.jit
def triton_per_fused__to_copy_add_convolution_div_mean_mul_permute_pow_sqrt_sub_unsqueeze_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 256.0, tl.float32)
    tmp9 = (tmp7 / tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp18 = (tmp16 / tmp8)
    tmp19 = tl.full([1, 1], 1e-06, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.sqrt_rn(tmp20)
    tmp22 = (tmp12 / tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp26, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/5l/c5lbnk6oxftp7gw3ziexz34pgqdef64iylxc3ufkicwvgrwsvroe.py
# Topologically Sorted Source Nodes: [x_14], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_14 => convert_element_type_26
# Graph fragment:
#   %arg26_1 : Tensor "f32[1024, 256][256, 1]cuda:0" = PlaceHolder[target=arg26_1]
#   %convert_element_type_26 : Tensor "bf16[1024, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_26
triton_poi_fused__to_copy_20 = async_compile.triton('triton_poi_fused__to_copy_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/qq/cqqwcgwuze5a3memgudk6wgd4zydam2p6bslxsibvqifn3y5eut6.py
# Topologically Sorted Source Nodes: [getitem_8, x_10, u_4, sub_9, sub_8, pow_5, s_4, add_9, sqrt_4, x_11, mul_4, getitem_9, x_12, x_13, x_14], Original ATen: [aten.unsqueeze, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul, aten.permute, aten.view, aten.t, aten.expand, aten.bmm]
# Source node to ATen node mapping:
#   add_9 => add_13
#   getitem_8 => unsqueeze_16, unsqueeze_17
#   getitem_9 => unsqueeze_18, unsqueeze_19
#   mul_4 => mul_16
#   pow_5 => convert_element_type_24, pow_5
#   s_4 => mean_9
#   sqrt_4 => sqrt_4
#   sub_8 => sub_8
#   sub_9 => sub_9
#   u_4 => mean_8
#   x_10 => convert_element_type_22, convert_element_type_23, convolution_6
#   x_11 => div_4
#   x_12 => add_14
#   x_13 => permute
#   x_14 => bmm, convert_element_type_26, convert_element_type_27, expand_1, permute_1, view, view_1
# Graph fragment:
#   %expand : Tensor "bf16[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0" = PlaceHolder[target=expand]
#   %convert_element_type_26 : Tensor "bf16[1024, 256][256, 1]cuda:0" = PlaceHolder[target=convert_element_type_26]
#   %unsqueeze_16 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, 1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 2), kwargs = {})
#   %convert_element_type_23 : Tensor "bf16[256, 1, 7, 7][49, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg22_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_22 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg23_1, torch.bfloat16), kwargs = {})
#   %convolution_6 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_12, %convert_element_type_23, %convert_element_type_22, [1, 1], [3, 3], [1, 1], False, [0, 0], 256), kwargs = {})
#   %mean_8 : Tensor "bf16[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_6, [1], True), kwargs = {})
#   %sub_9 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %mean_8), kwargs = {})
#   %sub_8 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %mean_8), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_8, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_24, 2), kwargs = {})
#   %mean_9 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [1], True), kwargs = {})
#   %add_13 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_9, 1e-06), kwargs = {})
#   %sqrt_4 : Tensor "f32[1, 1, 64, 64][4096, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_13,), kwargs = {})
#   %div_4 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_9, %sqrt_4), kwargs = {})
#   %mul_16 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %div_4), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, 1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %add_14 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %unsqueeze_19), kwargs = {})
#   %permute : Tensor "f32[1, 64, 64, 256][1048576, 64, 1, 4096]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_14, [0, 2, 3, 1]), kwargs = {})
#   %convert_element_type_27 : Tensor "bf16[1, 64, 64, 256][1048576, 64, 1, 4096]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute, torch.bfloat16), kwargs = {})
#   %view : Tensor "bf16[64, 64, 256][64, 1, 4096]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand, [64, 64, 256]), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[1024, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   %permute_1 : Tensor "bf16[256, 1024][1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_26, [1, 0]), kwargs = {})
#   %expand_1 : Tensor "bf16[1, 64, 256, 1024][256, 0, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%permute_1, [1, 64, 256, 1024]), kwargs = {})
#   %view_1 : Tensor "bf16[64, 256, 1024][0, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_1, [64, 256, 1024]), kwargs = {})
#   %bmm : Tensor "bf16[64, 64, 1024][65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %view_1), kwargs = {})
#   return %bmm
triton_tem_fused__to_copy_add_bmm_convolution_div_expand_mean_mul_permute_pow_sqrt_sub_t_unsqueeze_view_21 = async_compile.triton('triton_tem_fused__to_copy_add_bmm_convolution_div_expand_mean_mul_permute_pow_sqrt_sub_t_unsqueeze_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_bmm_convolution_div_expand_mean_mul_permute_pow_sqrt_sub_t_unsqueeze_view_21', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_add_bmm_convolution_div_expand_mean_mul_permute_pow_sqrt_sub_t_unsqueeze_view_21(arg_A, arg_B, out_ptr0):
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

    M = 64
    N = 1024
    K = 256

    stride_aq = 16384
    stride_am = 256
    stride_ak = 1

    stride_bq = 0
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
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1024*idx_m + 65536*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/4r/c4rg5gdc7p4nygceotcwcndnsxsa6mbycx3rytyverqthwmmsccv.py
# Topologically Sorted Source Nodes: [x_14, x_15], Original ATen: [aten.view, aten._to_copy, aten.add, aten.gelu]
# Source node to ATen node mapping:
#   x_14 => add_15, convert_element_type_25, view_2
#   x_15 => add_16, convert_element_type_30, convert_element_type_31, erf_4, mul_17, mul_18, mul_19
# Graph fragment:
#   %bmm : Tensor "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = PlaceHolder[target=bmm]
#   %arg27_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %view_2 : Tensor "bf16[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [1, 64, 64, 1024]), kwargs = {})
#   %convert_element_type_25 : Tensor "bf16[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   %add_15 : Tensor "bf16[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %convert_element_type_25), kwargs = {})
#   %convert_element_type_30 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.float32), kwargs = {})
#   %mul_17 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, 0.5), kwargs = {})
#   %mul_18 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, 0.7071067811865476), kwargs = {})
#   %erf_4 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_18,), kwargs = {})
#   %add_16 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_19 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %add_16), kwargs = {})
#   %convert_element_type_31 : Tensor "bf16[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_19, torch.bfloat16), kwargs = {})
#   return %convert_element_type_31
triton_poi_fused__to_copy_add_gelu_view_22 = async_compile.triton('triton_poi_fused__to_copy_add_gelu_view_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_gelu_view_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 25169920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_gelu_view_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.full([1], 0.5, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = tl.full([1], 1.0, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/hb/chbfwkwo6j3oe3jrrscjczryw3zitosvccuoysbngbrcjepwin62.py
# Topologically Sorted Source Nodes: [x_16, x_14, x_15], Original ATen: [aten._to_copy, aten.view, aten.add, aten.gelu, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   x_14 => add_15, convert_element_type_25, view_2
#   x_15 => add_16, convert_element_type_30, convert_element_type_31, erf_4, mul_17, mul_18, mul_19
#   x_16 => addmm, convert_element_type_32, convert_element_type_33, permute_2, view_3
# Graph fragment:
#   %convert_element_type_32 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=convert_element_type_32]
#   %convert_element_type_31 : Tensor "bf16[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0" = PlaceHolder[target=convert_element_type_31]
#   %convert_element_type_33 : Tensor "bf16[256, 1024][1024, 1]cuda:0" = PlaceHolder[target=convert_element_type_33]
#   %convert_element_type_32 : Tensor "bf16[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg29_1, torch.bfloat16), kwargs = {})
#   %view_2 : Tensor "bf16[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [1, 64, 64, 1024]), kwargs = {})
#   %convert_element_type_25 : Tensor "bf16[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg27_1, torch.bfloat16), kwargs = {})
#   %add_15 : Tensor "bf16[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %convert_element_type_25), kwargs = {})
#   %convert_element_type_30 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.float32), kwargs = {})
#   %mul_17 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, 0.5), kwargs = {})
#   %mul_18 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, 0.7071067811865476), kwargs = {})
#   %erf_4 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_18,), kwargs = {})
#   %add_16 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_4, 1), kwargs = {})
#   %mul_19 : Tensor "f32[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %add_16), kwargs = {})
#   %convert_element_type_31 : Tensor "bf16[1, 64, 64, 1024][4194304, 65536, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_19, torch.bfloat16), kwargs = {})
#   %view_3 : Tensor "bf16[4096, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_31, [4096, 1024]), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[256, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg28_1, torch.bfloat16), kwargs = {})
#   %permute_2 : Tensor "bf16[1024, 256][1, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_33, [1, 0]), kwargs = {})
#   %addmm : Tensor "bf16[4096, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_32, %view_3, %permute_2), kwargs = {})
#   return %addmm
triton_tem_fused__to_copy_add_addmm_gelu_t_view_23 = async_compile.triton('triton_tem_fused__to_copy_add_addmm_gelu_t_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=8,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_addmm_gelu_t_view_23', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_add_addmm_gelu_t_view_23(in_ptr0, arg_A, arg_B, out_ptr0):
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

    M = 4096
    N = 256
    K = 1024
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1024
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1024

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
        xindex = idx_n + 1024*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 256*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 1024*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


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


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/yq/cyqtp37tt5ru2uavwhdtv5e64zzj35ipxlv7n6jqcipksggaj2m5.py
# Topologically Sorted Source Nodes: [x_16, x_17, x_18, x_19, x_20], Original ATen: [aten.view, aten.mul, aten.permute, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   x_16 => view_4
#   x_17 => mul_20
#   x_18 => permute_3
#   x_19 => add_17
#   x_20 => convert_element_type_39
# Graph fragment:
#   %add_12 : Tensor "bf16[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = PlaceHolder[target=add_12]
#   %arg30_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %addmm : Tensor "bf16[4096, 256][256, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_4 : Tensor "bf16[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 64, 64, 256]), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg30_1, %view_4), kwargs = {})
#   %permute_3 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_20, [0, 3, 1, 2]), kwargs = {})
#   %add_17 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %permute_3), kwargs = {})
#   %convert_element_type_39 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.bfloat16), kwargs = {})
#   return %convert_element_type_39
triton_poi_fused__to_copy_add_mul_permute_view_24 = async_compile.triton('triton_poi_fused__to_copy_add_mul_permute_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_permute_view_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8389632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_permute_view_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp6 = tmp1 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/64/c64ec6kejsligotvegei7zj2zw4zt44a3kiujwgurgzle75ihfql.py
# Topologically Sorted Source Nodes: [x_16, x_17, x_18, x_19, x_26, x_27, x_28, x_29, x_30], Original ATen: [aten.view, aten.mul, aten.permute, aten.add, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   x_16 => view_4
#   x_17 => mul_20
#   x_18 => permute_3
#   x_19 => add_17
#   x_26 => view_9
#   x_27 => mul_25
#   x_28 => permute_7
#   x_29 => add_22
#   x_30 => convert_element_type_53, convert_element_type_54, convert_element_type_55, convolution_8
# Graph fragment:
#   %add_12 : Tensor "bf16[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = PlaceHolder[target=add_12]
#   %arg30_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %addmm : Tensor "bf16[4096, 256][256, 1]cuda:0" = PlaceHolder[target=addmm]
#   %arg39_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %addmm_1 : Tensor "bf16[4096, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %view_4 : Tensor "bf16[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 64, 64, 256]), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg30_1, %view_4), kwargs = {})
#   %permute_3 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_20, [0, 3, 1, 2]), kwargs = {})
#   %add_17 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %permute_3), kwargs = {})
#   %view_9 : Tensor "bf16[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, 64, 64, 256]), kwargs = {})
#   %mul_25 : Tensor "f32[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg39_1, %view_9), kwargs = {})
#   %permute_7 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_25, [0, 3, 1, 2]), kwargs = {})
#   %add_22 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %permute_7), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.bfloat16), kwargs = {})
#   %convert_element_type_54 : Tensor "bf16[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg40_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_53 : Tensor "bf16[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %convolution_8 : Tensor "bf16[1, 64, 64, 64][262144, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_55, %convert_element_type_54, %convert_element_type_53, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf56
triton_poi_fused__to_copy_add_convolution_mul_permute_view_25 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_mul_permute_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_mul_permute_view_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 10487808}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_mul_permute_view_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp6 = tmp1 + tmp5
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tmp6 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/qr/cqr4rudbhuxbgiuy2ckxhpgpnhlat42utukxiavulripbedoe5r2.py
# Topologically Sorted Source Nodes: [x_30], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_30 => convert_element_type_54
# Graph fragment:
#   %arg40_1 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=arg40_1]
#   %convert_element_type_54 : Tensor "bf16[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg40_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_54
triton_poi_fused__to_copy_26 = async_compile.triton('triton_poi_fused__to_copy_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/g5/cg5sgdrdeazdzecpm3j7ysz7mebwwswha5274b5zwrjbpqbfwa6v.py
# Topologically Sorted Source Nodes: [x_30], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_30 => convert_element_type_53
# Graph fragment:
#   %arg41_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %convert_element_type_53 : Tensor "bf16[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_53
triton_poi_fused__to_copy_27 = async_compile.triton('triton_poi_fused__to_copy_27', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/rk/crkib4wkdt3vhvveqeesdawvc5mvlrf3cydnu6oao2gmxpqlzicg.py
# Topologically Sorted Source Nodes: [x_16, x_17, x_18, x_19, x_26, x_27, x_28, x_29, x_30], Original ATen: [aten.view, aten.mul, aten.permute, aten.add, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   x_16 => view_4
#   x_17 => mul_20
#   x_18 => permute_3
#   x_19 => add_17
#   x_26 => view_9
#   x_27 => mul_25
#   x_28 => permute_7
#   x_29 => add_22
#   x_30 => convert_element_type_53, convert_element_type_54, convert_element_type_55, convolution_8
# Graph fragment:
#   %convert_element_type_53 : Tensor "bf16[64][1]cuda:0" = PlaceHolder[target=convert_element_type_53]
#   %buf56 : Tensor "bf16[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = PlaceHolder[target=buf56]
#   %convert_element_type_54 : Tensor "bf16[64, 256, 1, 1][256, 1, 16384, 16384]cuda:0" = PlaceHolder[target=convert_element_type_54]
#   %view_4 : Tensor "bf16[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 64, 64, 256]), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg30_1, %view_4), kwargs = {})
#   %permute_3 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_20, [0, 3, 1, 2]), kwargs = {})
#   %add_17 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %permute_3), kwargs = {})
#   %view_9 : Tensor "bf16[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, 64, 64, 256]), kwargs = {})
#   %mul_25 : Tensor "f32[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg39_1, %view_9), kwargs = {})
#   %permute_7 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_25, [0, 3, 1, 2]), kwargs = {})
#   %add_22 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %permute_7), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.bfloat16), kwargs = {})
#   %convert_element_type_54 : Tensor "bf16[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg40_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_53 : Tensor "bf16[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %convolution_8 : Tensor "bf16[1, 64, 64, 64][262144, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_55, %convert_element_type_54, %convert_element_type_53, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf59
triton_tem_fused__to_copy_add_convolution_mul_permute_view_28 = async_compile.triton('triton_tem_fused__to_copy_add_convolution_mul_permute_view_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_convolution_mul_permute_view_28', 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_add_convolution_mul_permute_view_28(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4096
    N = 64
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
        xindex = idx_n + 64*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 256*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 64*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/7r/c7rcbyuqf3qm7pgkrrhtfwxccanrksdcobbhaanetrm5vektgqez.py
# Topologically Sorted Source Nodes: [x_16, x_17, x_18, x_19, x_26, x_27, x_28, x_29, x_30], Original ATen: [aten.view, aten.mul, aten.permute, aten.add, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   x_16 => view_4
#   x_17 => mul_20
#   x_18 => permute_3
#   x_19 => add_17
#   x_26 => view_9
#   x_27 => mul_25
#   x_28 => permute_7
#   x_29 => add_22
#   x_30 => convert_element_type_53, convert_element_type_54, convert_element_type_55, convolution_8
# Graph fragment:
#   %buf59 : Tensor "bf16[4096, 64][64, 1]cuda:0" = PlaceHolder[target=buf59]
#   %view_4 : Tensor "bf16[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 64, 64, 256]), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg30_1, %view_4), kwargs = {})
#   %permute_3 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_20, [0, 3, 1, 2]), kwargs = {})
#   %add_17 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %permute_3), kwargs = {})
#   %view_9 : Tensor "bf16[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, 64, 64, 256]), kwargs = {})
#   %mul_25 : Tensor "f32[1, 64, 64, 256][1048576, 16384, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg39_1, %view_9), kwargs = {})
#   %permute_7 : Tensor "f32[1, 256, 64, 64][1048576, 1, 16384, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_25, [0, 3, 1, 2]), kwargs = {})
#   %add_22 : Tensor "f32[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %permute_7), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[1, 256, 64, 64][1048576, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.bfloat16), kwargs = {})
#   %convert_element_type_54 : Tensor "bf16[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg40_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_53 : Tensor "bf16[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %convolution_8 : Tensor "bf16[1, 64, 64, 64][262144, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_55, %convert_element_type_54, %convert_element_type_53, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_8
triton_poi_fused__to_copy_add_convolution_mul_permute_view_29 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_mul_permute_view_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_mul_permute_view_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 524288, 'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_mul_permute_view_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x1), ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 4096*y0), tmp0, ymask)
''', device_str='cuda')


# kernel path: /home/Shuai/AIAA3201_Project/AIAA-3102-Proj_3/.tmp/torchinductor_Shuai/5j/c5jg4pwdo3brogkgsmrirxm2t42vt6fv6lcyvj7uvmmuapbhjoxs.py
# Topologically Sorted Source Nodes: [getitem_12, repeat, pos], Original ATen: [aten.unsqueeze, aten.repeat, aten._to_copy]
# Source node to ATen node mapping:
#   getitem_12 => unsqueeze_24
#   pos => convert_element_type_56
#   repeat => repeat
# Graph fragment:
#   %arg42_1 : Tensor "f32[64, 64, 64][1, 4096, 64]cuda:0" = PlaceHolder[target=arg42_1]
#   %unsqueeze_24 : Tensor "f32[1, 64, 64, 64][64, 1, 4096, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg42_1, 0), kwargs = {})
#   %repeat : Tensor "f32[1, 64, 64, 64][262144, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze_24, [1, 1, 1, 1]), kwargs = {})
#   %convert_element_type_56 : Tensor "bf16[1, 64, 64, 64][262144, 4096, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%repeat, torch.bfloat16), kwargs = {})
#   return %convert_element_type_56
triton_poi_fused__to_copy_repeat_unsqueeze_30 = async_compile.triton('triton_poi_fused__to_copy_repeat_unsqueeze_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_repeat_unsqueeze_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 1048576, 'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_repeat_unsqueeze_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x1), ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 4096*y0), tmp1, ymask)
''', device_str='cuda')

def partition_0(args):
    arg19_1, arg20_1, arg21_1, arg2_1, arg0_1, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1 = args
    args.clear()
    assert_size_stride(arg19_1, (1, 256, 64, 64), (256, 1, 16384, 256))
    assert_size_stride(arg20_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg2_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg0_1, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1_1, (4, ), (1, ))
    assert_size_stride(arg3_1, (4, ), (1, ))
    assert_size_stride(arg4_1, (4, ), (1, ))
    assert_size_stride(arg5_1, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg6_1, (16, ), (1, ))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (16, ), (1, ))
    assert_size_stride(arg9_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (1024, 256), (256, 1))
    assert_size_stride(arg27_1, (1024, ), (1, ))
    assert_size_stride(arg28_1, (256, 1024), (1024, 1))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (1024, 256), (256, 1))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (256, 1024), (1024, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg41_1, (64, ), (1, ))
    assert_size_stride(arg42_1, (64, 64, 64), (1, 4096, 64))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 256, 64, 64), (1048576, 1, 16384, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_0.run(arg19_1, buf0, 1048576, stream=stream0)
        del arg19_1
        buf1 = empty_strided_cuda((256, 256, 1, 1), (256, 1, 65536, 65536), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg20_1, buf1, 65536, stream=stream0)
        del arg20_1
        buf2 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg21_1, buf2, 256, stream=stream0)
        del arg21_1
        buf3 = empty_strided_cuda((4096, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_convolution_3.run(buf2, buf0, buf1, buf3, 64, 1, 1, stream=stream0)
        buf4 = reinterpret_tensor(buf0, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_0.run(arg2_1, buf4, 1048576, stream=stream0)
        del arg2_1
        buf5 = empty_strided_cuda((4, 1, 3, 3), (9, 9, 3, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(arg0_1, buf5, 36, stream=stream0)
        del arg0_1
        buf6 = empty_strided_cuda((1, 4, 512, 512), (1048576, 262144, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_convolution_5.run(buf4, buf5, buf6, 1024, 1, 1, stream=stream0)
        del buf5
        buf7 = empty_strided_cuda((1, 1, 512, 512), (262144, 512, 512, 1), torch.bfloat16)
        buf8 = empty_strided_cuda((1, 1, 512, 512), (262144, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, u, sub, pow_1, s], Original ATen: [aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_mean_pow_sub_6.run(buf6, arg1_1, buf7, buf8, 262144, stream=stream0)
        buf10 = reinterpret_tensor(buf4, (1, 4, 512, 512), (1048576, 1, 2048, 4), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [getitem, input_1, sub_1, add, sqrt, x, mul, getitem_1, x_1, input_2, input_3], Original ATen: [aten.unsqueeze, aten._to_copy, aten.convolution, aten.sub, aten.add, aten.sqrt, aten.div, aten.mul, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_convolution_div_gelu_mul_sqrt_sub_unsqueeze_7.run(arg3_1, buf6, arg1_1, buf7, buf8, arg4_1, buf10, 262144, 4, stream=stream0)
        del arg1_1
        del arg3_1
        del arg4_1
        del buf8
        buf11 = empty_strided_cuda((16, 4, 3, 3), (36, 1, 12, 4), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(arg5_1, buf11, 64, 9, stream=stream0)
        del arg5_1
        buf12 = reinterpret_tensor(buf6, (1, 16, 256, 256), (1048576, 1, 4096, 16), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_convolution_gelu_9.run(buf10, buf11, buf12, 256, 1, 1, stream=stream0)
        del buf11
        buf16 = reinterpret_tensor(buf10, (1, 16, 256, 256), (1048576, 1, 4096, 16), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [getitem_2, input_2, input_3, u_1, sub_3, sub_2, pow_2, s_1, add_2, sqrt_1, x_2, mul_1, getitem_3, x_3, input_4, input_5], Original ATen: [aten.unsqueeze, aten.gelu, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_10.run(buf12, arg6_1, arg7_1, arg8_1, buf16, 65536, 16, stream=stream0)
        del arg6_1
        del arg7_1
        del arg8_1
        buf17 = empty_strided_cuda((64, 16, 3, 3), (144, 1, 48, 16), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(arg9_1, buf17, 1024, 9, stream=stream0)
        del arg9_1
        buf18 = reinterpret_tensor(buf12, (1, 64, 128, 128), (1048576, 1, 8192, 64), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_convolution_gelu_12.run(buf16, buf17, buf18, 128, 1, 1, stream=stream0)
        del buf17
        buf22 = reinterpret_tensor(buf16, (1, 64, 128, 128), (1048576, 1, 8192, 64), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [getitem_4, input_4, input_5, u_2, sub_5, sub_4, pow_3, s_2, add_4, sqrt_2, x_4, mul_2, getitem_5, x_5, input_6, input_7], Original ATen: [aten.unsqueeze, aten.gelu, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_13.run(buf18, arg10_1, arg11_1, arg12_1, buf22, 16384, 64, stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        buf23 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(arg13_1, buf23, 16384, 9, stream=stream0)
        del arg13_1
        buf24 = reinterpret_tensor(buf18, (1, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_convolution_gelu_15.run(buf22, buf23, buf24, 32, 2, 1, stream=stream0)
        del buf23
        buf28 = reinterpret_tensor(buf22, (1, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [getitem_6, input_6, input_7, u_3, sub_7, sub_6, pow_4, s_3, add_6, sqrt_3, x_6, mul_3, getitem_7, x_7, input_8, input_9], Original ATen: [aten.unsqueeze, aten.gelu, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_unsqueeze_16.run(buf24, arg14_1, arg15_1, arg16_1, buf28, 4096, 256, stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        buf29 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg17_1, buf29, 65536, stream=stream0)
        del arg17_1
        buf30 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg18_1, buf30, 256, stream=stream0)
        del arg18_1
        buf31 = reinterpret_tensor(buf24, (4096, 256), (256, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.gelu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_convolution_3.run(buf30, buf28, buf29, buf31, 64, 1, 1, stream=stream0)
        del buf28
        del buf29
        buf32 = reinterpret_tensor(buf3, (1, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_8, input_8, input_9, x_9], Original ATen: [aten._to_copy, aten.convolution, aten.gelu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_convolution_gelu_17.run(buf32, buf31, 1048576, stream=stream0)
        del buf31
        buf33 = empty_strided_cuda((256, 1, 7, 7), (49, 49, 7, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_18.run(arg22_1, buf33, 12544, stream=stream0)
        del arg22_1
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._to_copy, aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf34, (1, 256, 64, 64), (1048576, 1, 16384, 256), 'unknown_op')
        buf37 = reinterpret_tensor(buf34, (1, 64, 64, 256), (1048576, 16384, 256, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [getitem_8, x_10, u_4, sub_9, sub_8, pow_5, s_4, add_9, sqrt_4, x_11, mul_4, getitem_9, x_12, x_13, x_14], Original ATen: [aten.unsqueeze, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul, aten.permute]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_convolution_div_mean_mul_permute_pow_sqrt_sub_unsqueeze_19.run(buf37, arg23_1, arg24_1, arg25_1, 4096, 256, stream=stream0)
        del arg23_1
        del arg24_1
        del arg25_1
        buf38 = reinterpret_tensor(buf7, (1024, 256), (256, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_20.run(arg26_1, buf38, 262144, stream=stream0)
        del arg26_1
        buf39 = empty_strided_cuda((64, 64, 1024), (65536, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [getitem_8, x_10, u_4, sub_9, sub_8, pow_5, s_4, add_9, sqrt_4, x_11, mul_4, getitem_9, x_12, x_13, x_14], Original ATen: [aten.unsqueeze, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.add, aten.sqrt, aten.div, aten.mul, aten.permute, aten.view, aten.t, aten.expand, aten.bmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_bmm_convolution_div_expand_mean_mul_permute_pow_sqrt_sub_t_unsqueeze_view_21.run(buf37, buf38, buf39, 8, 64, 1, stream=stream0)
        del buf37
        buf40 = reinterpret_tensor(buf39, (1, 64, 64, 1024), (4194304, 65536, 1024, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_14, x_15], Original ATen: [aten.view, aten._to_copy, aten.add, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_gelu_view_22.run(buf40, arg27_1, 4194304, stream=stream0)
        del arg27_1
        buf41 = reinterpret_tensor(buf38, (256, 1024), (1024, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_20.run(arg28_1, buf41, 262144, stream=stream0)
        del arg28_1
        buf42 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg29_1, buf42, 256, stream=stream0)
        del arg29_1
        buf43 = empty_strided_cuda((4096, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_16, x_14, x_15], Original ATen: [aten._to_copy, aten.view, aten.add, aten.gelu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_gelu_t_view_23.run(buf42, buf40, buf41, buf43, 128, 1, 1, stream=stream0)
        del buf42
        buf44 = empty_strided_cuda((1, 256, 64, 64), (1048576, 1, 16384, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_16, x_17, x_18, x_19, x_20], Original ATen: [aten.view, aten.mul, aten.permute, aten.add, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mul_permute_view_24.run(buf32, arg30_1, buf43, buf44, 1048576, stream=stream0)
        buf45 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_18.run(arg31_1, buf45, 12544, stream=stream0)
        del arg31_1
        # Topologically Sorted Source Nodes: [x_16, x_17, x_18, x_19, x_20], Original ATen: [aten.view, aten.mul, aten.permute, aten.add, aten._to_copy, aten.convolution]
        buf46 = extern_kernels.convolution(buf44, buf45, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf46, (1, 256, 64, 64), (1048576, 1, 16384, 256), 'unknown_op')
        del buf44
        del buf45
        buf49 = reinterpret_tensor(buf46, (1, 64, 64, 256), (1048576, 16384, 256, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [getitem_10, x_16, x_17, x_18, x_19, x_20, u_5, sub_11, sub_10, pow_6, s_5, add_12, sqrt_5, x_21, mul_6, getitem_11, x_22, x_23, x_24], Original ATen: [aten.unsqueeze, aten.view, aten.mul, aten.permute, aten.add, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_convolution_div_mean_mul_permute_pow_sqrt_sub_unsqueeze_19.run(buf49, arg32_1, arg33_1, arg34_1, 4096, 256, stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        buf50 = reinterpret_tensor(buf41, (1024, 256), (256, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_20.run(arg35_1, buf50, 262144, stream=stream0)
        del arg35_1
        buf51 = reinterpret_tensor(buf40, (64, 64, 1024), (65536, 1024, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [getitem_10, x_16, x_17, x_18, x_19, x_20, u_5, sub_11, sub_10, pow_6, s_5, add_12, sqrt_5, x_21, mul_6, getitem_11, x_22, x_23, x_24], Original ATen: [aten.unsqueeze, aten.view, aten.mul, aten.permute, aten.add, aten._to_copy, aten.convolution, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.t, aten.expand, aten.bmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_bmm_convolution_div_expand_mean_mul_permute_pow_sqrt_sub_t_unsqueeze_view_21.run(buf49, buf50, buf51, 8, 64, 1, stream=stream0)
        del buf49
        del buf50
        buf52 = reinterpret_tensor(buf51, (1, 64, 64, 1024), (4194304, 65536, 1024, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_24, x_25], Original ATen: [aten.view, aten._to_copy, aten.add, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_gelu_view_22.run(buf52, arg36_1, 4194304, stream=stream0)
        del arg36_1
        buf53 = empty_strided_cuda((256, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_20.run(arg37_1, buf53, 262144, stream=stream0)
        del arg37_1
        buf54 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg38_1, buf54, 256, stream=stream0)
        del arg38_1
        buf55 = empty_strided_cuda((4096, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_26, x_24, x_25], Original ATen: [aten._to_copy, aten.view, aten.add, aten.gelu, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_addmm_gelu_t_view_23.run(buf54, buf52, buf53, buf55, 128, 1, 1, stream=stream0)
        del buf52
        del buf54
        buf56 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_16, x_17, x_18, x_19, x_26, x_27, x_28, x_29, x_30], Original ATen: [aten.view, aten.mul, aten.permute, aten.add, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_convolution_mul_permute_view_25.run(buf56, arg30_1, buf43, arg39_1, buf55, 1048576, stream=stream0)
        del arg30_1
        del arg39_1
        del buf43
        del buf55
        buf57 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 16384, 16384), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_26.run(arg40_1, buf57, 16384, stream=stream0)
        del arg40_1
        buf58 = empty_strided_cuda((64, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_27.run(arg41_1, buf58, 64, stream=stream0)
        del arg41_1
        buf59 = reinterpret_tensor(buf53, (4096, 64), (64, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [x_16, x_17, x_18, x_19, x_26, x_27, x_28, x_29, x_30], Original ATen: [aten.view, aten.mul, aten.permute, aten.add, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_tem_fused__to_copy_add_convolution_mul_permute_view_28.run(buf58, buf56, buf57, buf59, 64, 1, 1, stream=stream0)
        del buf56
        del buf57
        del buf58
        buf60 = empty_strided_cuda((1, 64, 64, 64), (262144, 4096, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_16, x_17, x_18, x_19, x_26, x_27, x_28, x_29, x_30], Original ATen: [aten.view, aten.mul, aten.permute, aten.add, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_convolution_mul_permute_view_29.run(buf59, buf60, 64, 4096, stream=stream0)
        buf61 = reinterpret_tensor(buf59, (1, 64, 64, 64), (262144, 4096, 64, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [getitem_12, repeat, pos], Original ATen: [aten.unsqueeze, aten.repeat, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_repeat_unsqueeze_30.run(arg42_1, buf61, 64, 4096, stream=stream0)
        del arg42_1
    return (buf60, buf61, )


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1 = args
        args.clear()
        partition0_args = [arg19_1, arg20_1, arg21_1, arg2_1, arg0_1, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1]
        del arg19_1, arg20_1, arg21_1, arg2_1, arg0_1, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1
        (buf60, buf61) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf60, buf61, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1, 256, 64, 64), (256, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, 64, 64), (1, 4096, 64), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
