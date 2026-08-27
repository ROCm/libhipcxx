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
// Sync level: MULTI-WAVE (collective async STORE, LDS -> global, two waves).
//
// This is the store-direction counterpart to the load-oriented multi-wave
// tests.  Two full wavefronts each first write their own element into an LDS
// staging tile of 2 * wave elements.  After a block sync publishes those LDS
// writes, the whole block cooperatively issues a single block-wide
// cuda::memcpy_async that copies the staging tile back out to global memory
// (LDS -> global).  The library partitions the bytes across all participating
// threads of both waves, and the block-scoped barrier (initialized for
// 2 * wave participants) drains the store across the wave boundary.  Each
// thread then validates the slot that landed in global memory.

#include <cuda/barrier>
#include <hip/hip_cooperative_groups.h>
#include "test_macros.h"
#include "hip_wavefront_size.h"

namespace cg = cooperative_groups;

__device__ int g_dst[128];

__device__ int test()
{
  cg::thread_block block = cg::this_thread_block();
  int const n   = static_cast<int>(block.size());
  int const tid = static_cast<int>(block.thread_rank());

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ int staged[];

  if (tid == 0)
  {
    init(&bar, n);
  }
  block.sync();

  // Each thread of both waves stages its own element into LDS, then the
  // block sync makes every wave's LDS writes visible before the store.
  staged[tid] = 800 + tid;
  block.sync();

  // Whole block (two waves) cooperatively copies the staged tile LDS -> global.
  cuda::memcpy_async(block, &g_dst[0], &staged[0], sizeof(int) * n, bar);

  bar.arrive_and_wait();

  // Each thread validates the slot that landed in global memory.
  if (g_dst[tid] != 800 + tid)
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
      g_dst[i] = 0;
    __syncthreads();

    int result = test();
    return result;
  ))

  return 0;
}
