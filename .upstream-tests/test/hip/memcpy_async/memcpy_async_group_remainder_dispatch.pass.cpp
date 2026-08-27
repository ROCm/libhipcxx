//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
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
// UNSUPPORTED: pre-gfx1250

// <cuda/barrier>
// Sync level: MULTI-WAVE - collective group copy with a non-divisible byte count.
//
// Two full wavefronts cooperatively issue cuda::memcpy_async with a byte count
// that is not evenly divisible by the thread-block group size. This exercises
// the group dispatch tail path after the uniform per-thread partition for both
// global -> LDS and LDS -> global copies.

#include <cuda/barrier>
#include <hip/hip_cooperative_groups.h>
#include "test_macros.h"
#include "hip_wavefront_size.h"

namespace cg = cooperative_groups;

constexpr int k_extra_bytes        = 17;
constexpr int k_waves              = 2;
constexpr int k_max_wavefront_size = 64;
constexpr int k_max_threads        = k_waves * k_max_wavefront_size;
constexpr int k_max_copy_bytes     = k_max_threads + k_extra_bytes;

__device__ unsigned char g_src[k_max_copy_bytes];
__device__ unsigned char g_dst[k_max_copy_bytes];

__device__ int test()
{
  cg::thread_block block = cg::this_thread_block();
  int const n            = static_cast<int>(block.size());
  int const tid          = static_cast<int>(block.thread_rank());
  int const copy_bytes   = n + k_extra_bytes;

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ unsigned char smem[];
  unsigned char* const dest   = smem;
  unsigned char* const staged = smem + copy_bytes;

  if (tid == 0)
  {
    init(&bar, n);
  }
  block.sync();

  for (int i = tid; i < copy_bytes; i += n)
  {
    dest[i]   = 0;
    staged[i] = static_cast<unsigned char>((31 + i * 5) & 0xff);
  }
  block.sync();

  cuda::memcpy_async(block, &dest[0], &g_src[0], copy_bytes, bar);

  bar.arrive_and_wait();

  for (int i = tid; i < copy_bytes; i += n)
  {
    if (dest[i] != g_src[i])
      return 1;
  }
  block.sync();

  cuda::memcpy_async(block, &g_dst[0], &staged[0], copy_bytes, bar);

  bar.arrive_and_wait();

  for (int i = tid; i < copy_bytes; i += n)
  {
    if (g_dst[i] != staged[i])
      return 2;
  }

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const threads = k_waves * get_wavefront_size();
    cuda_block_count        = 1;
    cuda_thread_count       = threads;
    cuda_shared_memory_size = 2 * (threads + k_extra_bytes) * sizeof(unsigned char);
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (
    for (int i = static_cast<int>(threadIdx.x); i < k_max_copy_bytes; i += static_cast<int>(blockDim.x))
    {
      g_src[i] = static_cast<unsigned char>((11 + i * 3) & 0xff);
      g_dst[i] = 0;
    }
    __syncthreads();

    int result = test();
    return result;
  ))

  return 0;
}