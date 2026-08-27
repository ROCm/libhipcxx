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
// Sync level: MULTI-WAVE — per-thread memcpy_async with non-wave-aligned count.
//
// Two full wavefronts use the per-thread cuda::memcpy_async interface to copy
// one full wave plus a partial wave.  Threads with tid >= copy_elems do not
// issue any async copy but still participate in arrive_and_wait uniformly.
//
// Pipeline:
//   1. ASYNC READ: threads 0..copy_elems-1 each issue a per-thread
//      cuda::memcpy_async for one element (global -> LDS).  All block threads
//      arrive_and_wait.
//   2. EXCHANGE + MODIFY: threads whose cross-wave partner also has valid data
//      exchange; others just transform their own element.
//   3. ASYNC STORE: threads 0..copy_elems-1 each issue a per-thread
//      cuda::memcpy_async to write their staged value back (LDS -> global).
//      All block threads arrive_and_wait.
//   4. VERIFY: threads 0..copy_elems-1 check correctness.

#include <cuda/barrier>
#include <hip/hip_cooperative_groups.h>
#include "test_macros.h"
#include "hip_wavefront_size.h"

namespace cg = cooperative_groups;

constexpr int k_mod_offset       = 1000;
constexpr int k_extra_copy_elems = 18;
constexpr int k_waves            = 2;
constexpr int k_max_wavefront_size = 64;
constexpr int k_max_copy_elems     = k_max_wavefront_size + k_extra_copy_elems;

__device__ int g_src[k_max_copy_elems];
__device__ int g_dst[k_max_copy_elems];

__device__ int test()
{
  cg::thread_block block = cg::this_thread_block();
  int const n    = static_cast<int>(block.size());
  int const wave = get_wavefront_size();
  int const tid  = static_cast<int>(block.thread_rank());
  int const copy_elems = wave + k_extra_copy_elems;

  int const partner = (tid + wave) % n;

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ int smem[];
  int* const dest   = smem;
  int* const staged = smem + n;

  if (tid == 0)
  {
    init(&bar, n);
  }
  block.sync();

  // 1. ASYNC READ: only threads with tid < copy_elems issue a copy.
  if (tid < copy_elems)
  {
    cuda::memcpy_async(&dest[tid], &g_src[tid], sizeof(int), bar);
  }

  bar.arrive_and_wait();

  // 2. EXCHANGE + MODIFY
  if (tid < copy_elems)
  {
    if (partner < copy_elems)
      staged[tid] = dest[partner] + k_mod_offset;
    else
      staged[tid] = dest[tid] + k_mod_offset;
  }
  block.sync();

  // 3. ASYNC STORE: only threads with tid < copy_elems write back.
  if (tid < copy_elems)
  {
    cuda::memcpy_async(&g_dst[tid], &staged[tid], sizeof(int), bar);
  }

  bar.arrive_and_wait();

  // 4. VERIFY
  if (tid < copy_elems)
  {
    int expected;
    if (partner < copy_elems){
      expected = g_src[partner] + k_mod_offset;
    }
    else{
      expected = g_src[tid] + k_mod_offset;
    }
    if (g_dst[tid] != expected){
      return 1;
    }
  }

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const threads = k_waves * get_wavefront_size();
    cuda_block_count        = 1;
    cuda_thread_count       = threads;
    cuda_shared_memory_size = 2 * threads * sizeof(int);
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (
    int const copy_elems = get_wavefront_size() + k_extra_copy_elems;
    for (int i = static_cast<int>(threadIdx.x); i < copy_elems; i += static_cast<int>(blockDim.x))
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
