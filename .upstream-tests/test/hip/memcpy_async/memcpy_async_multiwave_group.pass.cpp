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
// Sync level: MULTI-WAVE (collective group copy across two wavefronts).
//
// Two full wavefronts cooperatively issue a single block-wide
// cuda::memcpy_async over a tile of 2 * wave elements (global -> LDS).  The
// library partitions the bytes across all participating threads of both waves,
// and the block-scoped barrier (initialized for 2 * wave participants) drains
// the copy across the wave boundary.
//
// After the drain, each thread validates a slot owned by a DIFFERENT thread in
// the OTHER wavefront (its sibling lane, idx = (tid + wave) % n) rather than
// its own.  Because the byte partition is opaque, the validating thread is not
// the one that copied that slot, which proves the cross-wave drain made every
// thread's LDS writes visible to every other thread once arrive_and_wait()
// returns.

#include <cuda/barrier>
#include <hip/hip_cooperative_groups.h>
#include "test_macros.h"
#include "hip_wavefront_size.h"

namespace cg = cooperative_groups;

__device__ int g_src[128];

__device__ int test()
{
  cg::thread_block block = cg::this_thread_block();
  int const n    = static_cast<int>(block.size());
  int const wave = get_wavefront_size();
  int const tid  = static_cast<int>(block.thread_rank());

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ int dest[];

  if (block.thread_rank() == 0)
  {
    init(&bar, n);
  }
  block.sync();

  // Whole block (two waves) cooperatively copies the tile global -> LDS.
  cuda::memcpy_async(block, &dest[0], &g_src[0], sizeof(int) * n, bar);

  bar.arrive_and_wait();

  // Cross-wave validation: each thread checks the sibling lane in the OTHER
  // wavefront (a slot a DIFFERENT thread copied), not its own.
  int const idx = (tid + wave) % n;
  if (dest[idx] != g_src[idx])
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
      g_src[i] = 400 + i;
    __syncthreads();

    int result = test();
    return result;
  ))

  return 0;
}
