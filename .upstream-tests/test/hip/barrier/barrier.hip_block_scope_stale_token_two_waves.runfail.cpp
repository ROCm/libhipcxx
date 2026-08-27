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
// REQUIRES: enable_undefined_behavior_tests

// <cuda/barrier>
// DOCUMENTATION TEST: stale token reuse can bypass synchronization across waves.
// The runfail result documents the token-lifetime contract violation.

#include <cuda/barrier>
#include <cuda/std/cstdint>
#include "test_macros.h"
#include "hip_wavefront_size.h"

__device__ int test_stale_token_two_waves()
{
  int const k_wave_size = get_wavefront_size();
  int const k_thread_count = 2 * k_wave_size;

  __shared__ hip::barrier<hip::thread_scope_block> barrier;
  __shared__ hip::std::uint64_t stale_token;
  __shared__ int sentinel;
  __shared__ int wave1_observed;

  if (threadIdx.x == 0)
  {
    init(&barrier, k_thread_count);
    sentinel = 0;
    wave1_observed = -1;  // Sentinel for "not yet observed"
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    sentinel = 100;
  }
  auto tok0 = barrier.arrive();
  if (threadIdx.x == 0)
  {
    stale_token = tok0;
  }
  barrier.wait(hip::std::move(tok0));

  if (threadIdx.x == 0)
  {
    sentinel = 200;
  }
  auto tok1 = barrier.arrive();
  barrier.wait(hip::std::move(tok1));

  // All threads arrive, but wave 1 waits with the stale phase-0 token.
  auto tok2 = barrier.arrive();

  if (threadIdx.x < k_wave_size)
  {
    barrier.wait(hip::std::move(tok2));

    if (threadIdx.x == 0)
    {
      sentinel = 300;
    }
  }
  else
  {
    // Stale token may bypass the fresh phase's synchronization.
    barrier.wait(hip::std::move(stale_token));

    if (threadIdx.x == k_wave_size)
    {
      wave1_observed = sentinel;
    }
  }

  // All threads must return the same value for HIP to propagate it correctly.
  return (wave1_observed == 200) ? 1 : 0;
}

__device__ int test()
{
  return test_stale_token_two_waves();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size = get_wavefront_size();
    int const k_thread_count = 2 * k_wave_size;

    cuda_block_count = 1;
    cuda_thread_count = k_thread_count;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (return test();))
  return 0;
}
