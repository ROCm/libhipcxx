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
  int local_var;

// <<<<<<< OLD CODE from d6a2e4317d (0f41a822a3) - COMMENTED OUT
//   assert(is_address_from(address_space::global, &global_var));
//   assert(is_address_from(address_space::shared, &shared_var));
//   // NOTE(HIP/AMD): AMD does not provide __builtin_amdgcn_is_constant to detect constant memory.
//   //                Constant memory and global memory are indistinguishable at runtime with
//   //                the available AMD GCN builtins (__builtin_amdgcn_is_shared, __builtin_amdgcn_is_private).
// #if !defined(_CCCL_HIP_COMPILER)
//   assert(is_address_from(address_space::constant, &constant_var));
// #endif
//   assert(is_address_from(address_space::local, &local_var));
//   assert(is_address_from(address_space::grid_constant, &grid_constant_var) == _CCCL_HAS_GRID_CONSTANT());
//
// =======
  assert(is_address_from(&global_var, address_space::global));
  assert(is_address_from(&shared_var, address_space::shared));
  assert(is_address_from(&constant_var, address_space::constant));
  assert(is_address_from(&local_var, address_space::local));
  assert(is_address_from(&grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
// >>>>>>> END NEW CODE (0f41a822a3)
  // todo: test address_space::cluster_shared

  assert(is_object_from(global_var, address_space::global));
  assert(is_object_from(shared_var, address_space::shared));
  assert(is_object_from(constant_var, address_space::constant));
  assert(is_object_from(local_var, address_space::local));
  assert(is_object_from(grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(MyStruct{}); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
