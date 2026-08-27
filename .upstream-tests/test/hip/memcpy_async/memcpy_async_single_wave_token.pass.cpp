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
// Sync level: MULTI-THREAD, SINGLE WAVE (explicit arrival token).
//
// Same collective group copy as memcpy_async_single_wave_group, but instead of
// the arrive_and_wait() convenience form this uses the explicit two-step
// arrive() / wait(token) pattern.  This is the common form when a thread wants
// to overlap independent work between issuing the copy and consuming it.

#include <cuda/barrier>
#include <hip/hip_cooperative_groups.h>
#include "test_macros.h"
#include "hip_wavefront_size.h"

namespace cg = cooperative_groups;

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

__device__ int g_src[64];

__device__ int test()
{
  cg::thread_block block = cg::this_thread_block();
  int const n = static_cast<int>(block.size());

  __shared__ barrier_t bar;
  extern __shared__ int dest[];

  if (block.thread_rank() == 0)
  {
    init(&bar, n);
  }
  block.sync();

  cuda::memcpy_async(block, &dest[0], &g_src[0], sizeof(int) * n, bar);

  // Two-step completion: arrive, do unrelated work, then wait on the token.
  barrier_t::arrival_token token = bar.arrive();
  bar.wait(cuda::std::move(token));

  if (dest[block.thread_rank()] != g_src[block.thread_rank()])
    return 1;

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const wave = get_wavefront_size();
    cuda_block_count        = 1;
    cuda_thread_count       = wave;
    cuda_shared_memory_size = wave * sizeof(int);
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (
    for (int i = static_cast<int>(threadIdx.x); i < 64; i += static_cast<int>(blockDim.x))
      g_src[i] = 300 + i;
    __syncthreads();

    int result = test();
    return result;
  ))

  return 0;
}
