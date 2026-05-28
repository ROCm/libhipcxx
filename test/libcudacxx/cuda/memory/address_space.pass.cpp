//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cuda/memory>

struct MyStruct
{
  int v;
};

__device__ int global_var;
__constant__ int constant_var;

__global__ void test_kernel(const _CCCL_GRID_CONSTANT MyStruct grid_constant_var)
{
  using cuda::device::address_space;
  using cuda::device::is_address_from;
  using cuda::device::is_object_from;
  __shared__ int shared_var;
  int local_var{};

  // 1. Test non-volatile pointers/objects
  {
    assert(is_address_from(&global_var, address_space::global));
    assert(is_address_from(&shared_var, address_space::shared));
#if !_CCCL_HIP_COMPILATION()
    // NOTE(HIP/AMD): AMD GCN does not expose a `__builtin_amdgcn_is_constant`
    // builtin, so `__isConstant` cannot detect `__constant__` global
    // variables on HIP -- they are reported as global memory by the existing
    // address-space builtins (see <libhipcxx/__amd/amd_utils.h>). Skip the constant-
    // address-space assertions on HIP until the HIP runtime / compiler
    // exposes the necessary builtin. Ported from upgrade/3.1.4.
    assert(is_address_from(&constant_var, address_space::constant));
#endif // !_CCCL_HIP_COMPILATION()
    assert(is_address_from(&local_var, address_space::local));
    assert(is_address_from(&grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    // todo: test address_space::cluster_shared

    assert(is_object_from(global_var, address_space::global));
    assert(is_object_from(shared_var, address_space::shared));
#if !_CCCL_HIP_COMPILATION()
    // NOTE(HIP/AMD): see the matching note above.
    assert(is_object_from(constant_var, address_space::constant));
#endif // !_CCCL_HIP_COMPILATION()
    assert(is_object_from(local_var, address_space::local));
    assert(is_object_from(grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    // todo: test address_space::cluster_shared
  }

  // 2. Test volatile pointers/objects
  {
    volatile auto& v_global_var        = global_var;
    volatile auto& v_shared_var        = shared_var;
    volatile auto& v_constant_var      = constant_var;
    volatile auto& v_local_var         = local_var;
    volatile auto& v_grid_constant_var = grid_constant_var;

    assert(is_address_from(&v_global_var, address_space::global));
    assert(is_address_from(&v_shared_var, address_space::shared));
#if !_CCCL_HIP_COMPILATION()
    assert(is_address_from(&v_constant_var, address_space::constant));
#endif // !_CCCL_HIP_COMPILATION()
    assert(is_address_from(&v_local_var, address_space::local));
    assert(is_address_from(&v_grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    // todo: test address_space::cluster_shared

    assert(is_object_from(v_global_var, address_space::global));
    assert(is_object_from(v_shared_var, address_space::shared));
#if !_CCCL_HIP_COMPILATION()
    assert(is_object_from(v_constant_var, address_space::constant));
#endif // !_CCCL_HIP_COMPILATION()
    assert(is_object_from(v_local_var, address_space::local));
    assert(is_object_from(v_grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    // todo: test address_space::cluster_shared
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(MyStruct{}); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
