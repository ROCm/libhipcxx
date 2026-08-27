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
// Verify a completion function can aggregate per-phase block data.

#include <cuda/barrier>
#include "test_macros.h"
#include "hip_wavefront_size.h"

static constexpr int k_num_phases = 5;

__device__ int test()
{
  int const k_wave_size = get_wavefront_size();
  int const k_thread_count = 2 * k_wave_size;

  extern __shared__ int partial_sums[];  // Dynamic shared memory
  __shared__ int block_result;           // Aggregated result

  auto completion = [&]()
  {
    block_result = 0;
    for (int i = 0; i < k_thread_count; i++)
    {
      block_result += partial_sums[i];
    }
  };

  using barrier_t = cuda::barrier<cuda::thread_scope_block, decltype(completion)>;
  __shared__ alignas(barrier_t) char bar_storage[sizeof(barrier_t)];
  barrier_t* bar = reinterpret_cast<barrier_t*>(bar_storage);

  if (threadIdx.x == 0)
  {
    init(bar, k_thread_count, completion);
  }
  __syncthreads();  // Legitimate: ensures init is visible

  for (int phase = 0; phase < k_num_phases; phase++)
  {
    partial_sums[threadIdx.x] = phase * k_thread_count + threadIdx.x;

    bar->arrive_and_wait();

    int expected = phase * k_thread_count * k_thread_count +
                   (k_thread_count * (k_thread_count - 1)) / 2;

    if (block_result != expected)
    {
      return (phase + 1);  // Report which phase failed.
    }
  }

  return 0;
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size = get_wavefront_size();
    int const k_thread_count = 2 * k_wave_size;

    size_t sharedMemSize = k_thread_count * sizeof(int);

    cuda_block_count = 1;
    cuda_thread_count = k_thread_count;
    cuda_shared_memory_size = sharedMemSize;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
