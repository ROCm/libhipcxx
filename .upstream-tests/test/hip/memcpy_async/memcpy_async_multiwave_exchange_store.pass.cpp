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
// Sync level: MULTI-WAVE (full read-modify-store round trip across two waves).
//
// Two full wavefronts drive a complete async pipeline through a single
// block-scoped barrier:
//
//   1. ASYNC READ : the whole block cooperatively issues one block-wide
//      cuda::memcpy_async that loads a tile of 2 * wave elements
//      (global g_src -> LDS dest).  arrive_and_wait() drains the load.
//   2. WAVE EXCHANGE: each thread reads from LDS the element its cross-wave
//      partner loaded (partner = (tid + wave) % n), so a thread in wave 0
//      consumes a slot wave 1 produced and vice versa.
//   3. MODIFY + WRITE BACK: the exchanged value is transformed and written
//      into a second LDS staging region (staged[tid]).  __syncthreads()
//      publishes every thread's staged write before the store is issued.
//   4. ASYNC STORE: the whole block cooperatively issues one block-wide
//      cuda::memcpy_async that writes the staged tile back out
//      (LDS staged -> global g_dst).  arrive_and_wait() drains the store.
//   5. VERIFY: each thread checks that the value that landed in global memory
//      equals the transform of the element its partner originally loaded.

#include <cuda/barrier>
#include <hip/hip_cooperative_groups.h>
#include "test_macros.h"
#include "hip_wavefront_size.h"

namespace cg = cooperative_groups;

constexpr int k_mod_offset = 1000;

__device__ int g_src[128];
__device__ int g_dst[128];

__device__ int test()
{
  cg::thread_block block = cg::this_thread_block();
  int const n    = static_cast<int>(block.size());
  int const wave = get_wavefront_size();
  int const tid  = static_cast<int>(block.thread_rank());

  // Cross-wave partner: wave 0 thread tid pairs with wave 1 thread tid + wave
  // and vice versa.  The mapping is symmetric (partner of partner == tid).
  int const partner = (tid + wave) % n;

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ int smem[];
  int* const dest   = smem;       // tile filled by the async read
  int* const staged = smem + n;   // tile consumed by the async store

  if (tid == 0)
  {
    init(&bar, n);
  }
  block.sync();

  // 1. ASYNC READ: whole block cooperatively loads the tile global -> LDS.
  cuda::memcpy_async(block, &dest[0], &g_src[0], sizeof(int) * n, bar);
  bar.arrive_and_wait();

  // 2. WAVE EXCHANGE: read the slot the cross-wave partner loaded.
  // 3. MODIFY + WRITE BACK: transform and stage into the second LDS region.
  staged[tid] = dest[partner] + k_mod_offset;
  block.sync();

  // 4. ASYNC STORE: whole block cooperatively writes the staged tile back
  //    out LDS -> global.
  cuda::memcpy_async(block, &g_dst[0], &staged[0], sizeof(int) * n, bar);
  bar.arrive_and_wait();

  // 5. VERIFY: the value now in global memory must be the transform of the
  //    element this thread's partner originally loaded.
  if (g_dst[tid] != g_src[partner] + k_mod_offset)
    return 1;

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const threads = 2 * get_wavefront_size();
    cuda_block_count        = 1;
    cuda_thread_count       = threads;
    cuda_shared_memory_size = 2 * threads * sizeof(int);
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (
    for (int i = static_cast<int>(threadIdx.x); i < 128; i += static_cast<int>(blockDim.x))
    {
      g_src[i] = 800 + i;
      g_dst[i] = 0;
    }
    __syncthreads();

    int result = test();
    return result;
  ))

  return 0;
}
