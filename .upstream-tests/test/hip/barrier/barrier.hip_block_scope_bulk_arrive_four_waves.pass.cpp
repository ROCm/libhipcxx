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
//
// Verify bulk arrivals from four wave leaders complete the phase and order each
// wave's payload writes. Each thread waits on a copy of its leader's token.

#include <cuda/barrier>
#include <cuda/std/cstdint>
#include "test_macros.h"
#include "hip_wavefront_size.h"

static constexpr int k_waves = 4;

__device__ int test_four_wave_bulk_arrive()
{
  int const k_wave_size = get_wavefront_size();
  int const k_total     = k_waves * k_wave_size;

  __shared__ hip::barrier<hip::thread_scope_block>          b;
  __shared__ hip::barrier<hip::thread_scope_block>          token_sync;  // Token broadcast.
  __shared__ hip::barrier<hip::thread_scope_block>::arrival_token wave_tokens[k_waves];

  extern __shared__ int payload[];

  if (threadIdx.x == 0) {
    init(&b, k_total);
    init(&token_sync, k_total);  // Token synchronization barrier.
  }
  __syncthreads();  // Legitimate: ensures both barrier inits are visible

  // Every thread publishes its payload slot before the wave leader arrives.
  payload[threadIdx.x] = threadIdx.x + 1;

  int  wave_id        = threadIdx.x / k_wave_size;
  bool is_wave_leader = (threadIdx.x % k_wave_size == 0);

  // Each wave leader issues arrive(k_wave_size) for its wave.
  if (is_wave_leader)
    wave_tokens[wave_id] = b.arrive(k_wave_size);

  // Ensure leader token writes are visible before any thread reads.
  token_sync.arrive_and_wait();

  // Copy before move because multiple threads share each token slot.
  auto my_token = wave_tokens[wave_id];
  b.wait(hip::std::move(my_token));

  // Stores made before any arrive() must be visible after wait().
  if (threadIdx.x == 0)
  {
    for (int i = 0; i < k_total; ++i)
    {
      if (payload[i] != i + 1)
      {
        return 1;
      }
    }
  }

  return 0;
}

__device__ int test()
{
  return test_four_wave_bulk_arrive();
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size = get_wavefront_size();
    int const k_total     = k_waves * k_wave_size;

    size_t sharedMemSize = k_total * sizeof(int);

    cuda_block_count        = 1;
    cuda_thread_count       = k_total;
    cuda_shared_memory_size = sharedMemSize;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}