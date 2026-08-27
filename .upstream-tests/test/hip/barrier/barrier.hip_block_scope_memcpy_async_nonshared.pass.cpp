// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// UNSUPPORTED: nvcc, nvhpc, nvc++

// <cuda/barrier>
// Verify cuda::memcpy_async with a device-global block-scope barrier.

#include <cuda/barrier>
#include "test_macros.h"

constexpr int k_count = 64;

__device__ int g_src[k_count];
__device__ cuda::barrier<cuda::thread_scope_block> g_bar;

__device__ int test()
{
  __shared__ int dest[k_count];

  if (threadIdx.x == 0) {
    init(&g_bar, 1);
    for (int i = 0; i < k_count; ++i) {
      dest[i] = -1;
    }

    cuda::memcpy_async(&dest[0], &g_src[0], sizeof(int) * k_count, g_bar);
    g_bar.arrive_and_wait();

    for (int i = 0; i < k_count; ++i) {
      if (dest[i] != g_src[i]) {
        return 1;
      }
    }
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
    for (int i = 0; i < k_count; ++i) {
      g_src[i] = 900 + i;
    }

    int result = test();
    return result;
  ))

  return 0;
}