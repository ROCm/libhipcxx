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
// Sync level: MULTI-WAVE (4 waves) — collective load then cross-wave exchange.
//
// Four full wavefronts load a tile of 4 * wave elements (global -> LDS): every
// thread with block rank tid issues its own cuda::memcpy_async for the single
// element at dest[tid] / g_src[tid].  The four waves therefore own four
// contiguous segments (wave w owns dest[w*wave .. (w+1)*wave)).
//
// After the block-scoped barrier (initialized for all 4 * wave participants)
// drains every thread's copy, each thread performs the EXCHANGE step: it reads
// the slots that the *other three* waves loaded (at its own lane) and compares
// each against the global reference value.  This proves the LDS writes from
// every wave are visible to every other wave once arrive_and_wait() returns.

#include <cuda/barrier>
#include <hip/hip_cooperative_groups.h>
#include "test_macros.h"
#include "hip_wavefront_size.h"

namespace cg = cooperative_groups;

constexpr int k_waves = 4;

__device__ int g_src[4 * 64]; // sized for the largest supported wavefront

__device__ int test()
{
  cg::thread_block block = cg::this_thread_block();
  int const n    = static_cast<int>(block.size());
  int const wave = get_wavefront_size();
  int const tid  = static_cast<int>(block.thread_rank());
  int const lane    = tid % wave;
  int const my_wave = tid / wave;

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ int dest[];

  if (tid == 0)
  {
    init(&bar, n);
  }
  block.sync();

  // Collective read: every thread of all 4 waves loads its own element
  // global -> LDS (thread with block rank tid owns dest[tid] / g_src[tid]).
  cuda::memcpy_async(&dest[tid], &g_src[tid], sizeof(int), bar);

  bar.arrive_and_wait();

  // Exchange: read the segments the OTHER waves loaded (at this thread's lane)
  // and compare each against the global reference value.
  for (int w = 0; w < k_waves; ++w)
  {
    if (w == my_wave)
      continue;

    int const idx = w * wave + lane;
    if (dest[idx] != g_src[idx])
      return 1;
  }

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const threads = k_waves * get_wavefront_size();
    cuda_block_count        = 1;
    cuda_thread_count       = threads;
    cuda_shared_memory_size = threads * sizeof(int);
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (
    for (int i = static_cast<int>(threadIdx.x); i < k_waves * 64; i += static_cast<int>(blockDim.x))
      g_src[i] = 700 + i;
    __syncthreads();

    int result = test();
    return result;
  ))

  return 0;
}
