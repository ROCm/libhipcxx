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
// Verify device-scope completion updates global state across phases.

#include <cuda/barrier>
#include "test_macros.h"

static constexpr int k_num_blocks = 1;
static constexpr int k_threads_per_block = 64;
static constexpr int k_total_threads = k_num_blocks * k_threads_per_block;

__device__ int global_phase = 0;
__device__ int global_completion_count = 0;

__device__ void device_completion()
{
  global_phase++;
  global_completion_count++;
}

using barrier_t = cuda::barrier<cuda::thread_scope_device, void(*)()>;
__device__ alignas(barrier_t) char global_bar_storage[sizeof(barrier_t)];

__device__ int test()
{
  barrier_t* global_bar = reinterpret_cast<barrier_t*>(global_bar_storage);

  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    init(global_bar, k_total_threads, device_completion);
    global_phase = 0;
    global_completion_count = 0;
  }
  __syncthreads();

  global_bar->arrive_and_wait();

  if (global_phase != 1)
  {
    return 1;
  }

  if (global_completion_count != 1)
  {
    return 2;
  }

  global_bar->arrive_and_wait();

  if (global_phase != 2)
  {
    return 3;
  }

  if (global_completion_count != 2)
  {
    return 4;
  }

  return 0;
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    cuda_block_count = k_num_blocks;
    cuda_thread_count = k_threads_per_block;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
