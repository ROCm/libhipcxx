//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// UNSUPPORTED: nvcc, nvhpc, nvc++

// <cuda/barrier>
// Sync level: MULTI-WAVE (independent per-thread copies across two waves).
//
// Two full wavefronts each issue their own single-thread cuda::memcpy_async
// for one element (global -> LDS).  All 2 * wave copies, spanning both waves,
// defer their completion to a single shared barrier initialized for the whole
// block.  arrive_and_wait() drains every copy across the wave boundary.

#include <cuda/barrier>
#include "test_macros.h"
#include "hip_wavefront_size.h"

__device__ int g_src[128];

__device__ int test()
{
  int const n = static_cast<int>(blockDim.x);
  int const tid = static_cast<int>(threadIdx.x);

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ int dest[];

  if (tid == 0)
  {
    init(&bar, n);
  }
  __syncthreads();

  // Each thread of both waves copies its own element independently.
  cuda::memcpy_async(&dest[tid], &g_src[tid], sizeof(int), bar);

  bar.arrive_and_wait();

  if (dest[tid] != g_src[tid])
    return 1;

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const threads = 2 * get_wavefront_size();
    cuda_block_count        = 1;
    cuda_thread_count       = threads;
    cuda_shared_memory_size = threads * sizeof(int);
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (
    for (int i = static_cast<int>(threadIdx.x); i < 128; i += static_cast<int>(blockDim.x))
      g_src[i] = 500 + i;
    __syncthreads();

    int result = test();
    return result;
  ))

  return 0;
}
