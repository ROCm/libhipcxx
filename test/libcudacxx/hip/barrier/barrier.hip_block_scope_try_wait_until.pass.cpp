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
// Verify try_wait_until() completion and ordering for block-scope barriers.

#include <cuda/barrier>
#include <cuda/std/chrono>
#include "test_macros.h"
#include "hip_wavefront_size.h"

__device__ int test_ordering_with_past_deadline()
{
  int const k_wave_size = get_wavefront_size();
  int const k_thread_count = 2 * k_wave_size;

  __shared__ hip::barrier<hip::thread_scope_block> barrier;
  __shared__ int sentinel;
  __shared__ hip::barrier<hip::thread_scope_block>::arrival_token token;

  if (threadIdx.x == 0)
  {
    init(&barrier, k_thread_count);
    sentinel = 100;
  }
  __syncthreads();

  int wave_id = threadIdx.x / k_wave_size;

  if (wave_id == 0)
  {
    if (threadIdx.x == 0)
    {
      sentinel = 300;
    }

    if (threadIdx.x == 0)
    {
      token = barrier.arrive(k_wave_size);
    }
  }

  if (threadIdx.x == k_wave_size)
  {
    (void) barrier.arrive(k_wave_size);
  }
  __syncthreads();

  auto my_token = token;
  auto now = hip::std::chrono::high_resolution_clock::now();
  auto past = now - hip::std::chrono::seconds(1);

  bool completed = barrier.try_wait_until(hip::std::move(my_token), past);

  if (!completed) return 1;

  if (sentinel != 300) return 2;

  return 0;
}

__device__ int test()
{
  return test_ordering_with_past_deadline();
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size = get_wavefront_size();
    int const k_thread_count = 2 * k_wave_size;

    cuda_block_count = 1;
    cuda_thread_count = k_thread_count;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
