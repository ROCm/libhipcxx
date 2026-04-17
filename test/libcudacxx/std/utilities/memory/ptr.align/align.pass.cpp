//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

// #include <memory>

// void* align(size_t alignment, size_t size, void*& ptr, size_t& space);

#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "test_macros.h"

#if _CCCL_CUDA_COMPILATION()

struct MyStruct
{
  int v;
};

TEST_GLOBAL_VARIABLE int global_var;
__constant__ int constant_var;

__global__ void test_kernel(const _CCCL_GRID_CONSTANT MyStruct grid_constant_var)
{
  using cuda::device::address_space;
  using cuda::device::is_address_from;

  __shared__ int shared_var;
  int local_var;

  size_t space = 20;
  {
    void* global_ptr = &global_var;
    assert(is_address_from(cuda::std::align(4, 10, global_ptr, space), address_space::global));
  }
  {
    void* shared_ptr = &shared_var;
    assert(is_address_from(cuda::std::align(4, 10, shared_ptr, space), address_space::shared));
  }
  // NOTE(HIP/AMD): AMD does not provide __builtin_amdgcn_is_constant to detect constant memory.
  //                Constant memory and global memory are indistinguishable at runtime with
  //                the available AMD GCN builtins (__builtin_amdgcn_is_shared, __builtin_amdgcn_is_private).
#  if !defined(_CCCL_HIP_COMPILER)
  {
    void* constant_ptr = &constant_var;
    assert(is_address_from(cuda::std::align(4, 10, constant_ptr, space), address_space::constant));
  }
#  endif
  {
    void* local_ptr = &local_var;
    assert(is_address_from(cuda::std::align(4, 10, local_ptr, space), address_space::local));
  }
// Compilation with lang-14 with nvcc-12 stucks
#  if _CCCL_HAS_GRID_CONSTANT() && !_CCCL_COMPILER(CLANG, <=, 14) && !_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0)
  {
    void* grid_constant_ptr = const_cast<void*>(static_cast<const void*>(&grid_constant_var.v));
    assert(is_address_from(cuda::std::align(4, 10, grid_constant_ptr, space), address_space::grid_constant));
  }
#  endif // _CCCL_HAS_GRID_CONSTANT() && !_CCCL_COMPILER(CLANG, <= 14)
  // todo: test address_space::cluster_shared
}

#endif // _CCCL_CUDA_COMPILATION()

int main(int, char**)
{
  const unsigned N = 20;
  char buf[N];
  void* r;
  void* p             = &buf[0];
  cuda::std::size_t s = N;
  r                   = cuda::std::align(4, 10, p, s);
  assert(p == &buf[0]);
  assert(r == p);
  assert(s == N);

  p = &buf[1];
  s = N;
  r = cuda::std::align(4, 10, p, s);
  assert(p == &buf[4]);
  assert(r == p);
  assert(s == N - 3);

  p = &buf[2];
  s = N;
  r = cuda::std::align(4, 10, p, s);
  assert(p == &buf[4]);
  assert(r == p);
  assert(s == N - 2);

  p = &buf[3];
  s = N;
  r = cuda::std::align(4, 10, p, s);
  assert(p == &buf[4]);
  assert(r == p);
  assert(s == N - 1);

  p = &buf[4];
  s = N;
  r = cuda::std::align(4, 10, p, s);
  assert(p == &buf[4]);
  assert(r == p);
  assert(s == N);

  p = &buf[0];
  s = N;
  r = cuda::std::align(4, N, p, s);
  assert(p == &buf[0]);
  assert(r == p);
  assert(s == N);

  p = &buf[1];
  s = N - 1;
  r = cuda::std::align(4, N - 4, p, s);
  assert(p == &buf[4]);
  assert(r == p);
  assert(s == N - 4);

  p = &buf[1];
  s = N - 1;
  r = cuda::std::align(4, N - 3, p, s);
  assert(p == &buf[1]);
  assert(r == nullptr);
  assert(s == N - 1);

  p = &buf[0];
  s = N;
  r = cuda::std::align(1, N + 1, p, s);
  assert(p == &buf[0]);
  assert(r == nullptr);
  assert(s == N);

#if _CCCL_CUDA_COMPILATION()
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(MyStruct{}); assert(cudaDeviceSynchronize() == cudaSuccess);))
#endif // _CCCL_CUDA_COMPILATION()

  return 0;
}
