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
// Sync level: SINGLE THREAD.
//
// A single thread issues one cuda::memcpy_async for a whole array (global ->
// LDS) using the single-thread overload (no group argument), then drains the
// copy with a block-scoped barrier initialized for one participant.  This is
// the simplest memcpy_async usage pattern.

#include <cuda/barrier>
#include "test_macros.h"

constexpr int k_count = 8;

__device__ int g_src[k_count] = {10, 11, 12, 13, 14, 15, 16, 17};

__device__ int test()
{
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  __shared__ int dest[k_count];

  init(&bar, 1);

  // Single thread copies the entire array global -> LDS.
  cuda::memcpy_async(&dest[0], &g_src[0], sizeof(int) * k_count, bar);

  bar.arrive_and_wait();

  for (int i = 0; i < k_count; ++i)
  {
    if (dest[i] != g_src[i])
      return 1;
  }

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    cuda_block_count  = 1;
    cuda_thread_count = 1;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (
    int result = test();
    return result;
  ))

  return 0;
}
