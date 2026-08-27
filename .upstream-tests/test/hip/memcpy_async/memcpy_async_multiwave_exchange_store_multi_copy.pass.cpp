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
// Sync level: MULTI-WAVE — multiple block-cooperative memcpy_async calls per
// single arrive_and_wait.
//
// Two full wavefronts issue MULTIPLE block-cooperative cuda::memcpy_async calls
// that all track against the same barrier phase.  A single arrive_and_wait()
// then drains all outstanding copies at once.  This validates that multiple
// collective async copies accumulated on the same barrier phase complete
// correctly before the barrier unblocks.
//
// The pipeline is:
//   1. ASYNC READ: k_tiles separate block-cooperative memcpy_async calls load
//      k_tiles tiles of n elements each (global -> LDS), all on the same
//      barrier.  One arrive_and_wait() drains all tiles.
//   2. WAVE EXCHANGE + MODIFY: for each tile, each thread reads the cross-wave
//      partner's element, transforms it, and stages into a write-back region.
//   3. ASYNC STORE: k_tiles separate block-cooperative memcpy_async calls write
//      the staged tiles back (LDS -> global), all on the same barrier.  One
//      arrive_and_wait() drains them all.
//   4. VERIFY: each thread checks all tiles.

#include <cuda/barrier>
#include <hip/hip_cooperative_groups.h>
#include "test_macros.h"
#include "hip_wavefront_size.h"

namespace cg = cooperative_groups;

constexpr int k_mod_offset = 1000;
constexpr int k_tiles      = 4;
constexpr int k_waves      = 2;
constexpr int k_max_wavefront_size = 64;
constexpr int k_max_threads = k_waves * k_max_wavefront_size;

__device__ int g_src[k_tiles * k_max_threads];
__device__ int g_dst[k_tiles * k_max_threads];

__device__ int test()
{
  cg::thread_block block = cg::this_thread_block();
  int const n    = static_cast<int>(block.size());
  int const wave = get_wavefront_size();
  int const tid  = static_cast<int>(block.thread_rank());
  int const total_elems = n * k_tiles;

  int const partner = (tid + wave) % n;

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ int smem[];
  int* const dest   = smem;                // k_tiles tiles filled by async reads
  int* const staged = smem + total_elems;  // k_tiles tiles for async stores

  if (tid == 0)
  {
    init(&bar, n);
  }
  block.sync();

  // 1. ASYNC READ: issue k_tiles block-cooperative copies on the same barrier.
  for (int t = 0; t < k_tiles; ++t)
  {
    cuda::memcpy_async(block, &dest[t * n], &g_src[t * n], sizeof(int) * n, bar);
  }
  bar.arrive_and_wait();

  // 2. WAVE EXCHANGE + 3. MODIFY: transform each tile.
  for (int t = 0; t < k_tiles; ++t)
  {
    staged[t * n + tid] = dest[t * n + partner] + k_mod_offset;
  }
  block.sync();

  // 4. ASYNC STORE: issue k_tiles block-cooperative copies on the same barrier.
  for (int t = 0; t < k_tiles; ++t)
  {
    cuda::memcpy_async(block, &g_dst[t * n], &staged[t * n], sizeof(int) * n, bar);
  }
  bar.arrive_and_wait();

  // 5. VERIFY
  for (int t = 0; t < k_tiles; ++t)
  {
    if (g_dst[t * n + tid] != g_src[t * n + partner] + k_mod_offset)
      return 1;
  }

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const threads = k_waves * get_wavefront_size();
    int const total_elems = threads * k_tiles;
    cuda_block_count        = 1;
    cuda_thread_count       = threads;
    cuda_shared_memory_size = 2 * total_elems * sizeof(int);
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (
    int const total_elems = static_cast<int>(blockDim.x) * k_tiles;
    for (int i = static_cast<int>(threadIdx.x); i < total_elems; i += static_cast<int>(blockDim.x))
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
