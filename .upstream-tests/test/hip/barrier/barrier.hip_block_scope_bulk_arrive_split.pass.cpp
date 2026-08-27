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
// Verify unequal bulk arrivals from two wave leaders complete the phase and
// order each wave's writes. The wave-convergent shape avoids same-wave split
// arrive/wait progress hazards.

#include <cuda/barrier>
#include "test_macros.h"
#include "hip_wavefront_size.h"

static constexpr int k_n = 8; // total expected arrivals; kept < k_wave

__device__ int test_bulk_arrive_split_contributions()
{
  __shared__ hip::barrier<hip::thread_scope_block> b;
  __shared__ int payload[k_n];

  int const k_wave = get_wavefront_size();

  if (threadIdx.x == 0)
    init(&b, k_n);
  __syncthreads();

  bool const is_wave0        = threadIdx.x < k_wave;
  bool const is_wave0_leader = threadIdx.x == 0;
  bool const is_wave1_leader = threadIdx.x == k_wave;

  if (is_wave0)
  {
    // Wave 0 contributes slots 0..k_n-2; extra lanes are padding.
    if (threadIdx.x < k_n - 1)
      payload[threadIdx.x] = threadIdx.x + 1;

    // Wave 0 leader contributes k_n - 1 arrivals.
    if (is_wave0_leader)
    {
      auto tok = b.arrive(k_n - 1);
      b.wait(hip::std::move(tok));

      // Wave 0 observes wave 1's cross-wave contribution after wait().
      if (payload[k_n - 1] != k_n)
      {
        return 1;
      }
    }
  }
  else
  {
    // Wave 1 leader contributes the remaining expected arrival.
    if (is_wave1_leader)
    {
      payload[k_n - 1] = k_n;

      auto tok = b.arrive(1);
      b.wait(hip::std::move(tok));

      // Wave 1 also observes wave 0's writes after wait().
      for (int i = 0; i < k_n - 1; ++i)
      {
        if (payload[i] != i + 1)
        {
          return 2;
        }
      }
    }
  }

  return 0;
}

__device__ int test()
{
  return test_bulk_arrive_split_contributions();
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave  = get_wavefront_size();
    cuda_block_count  = 1;
    cuda_thread_count = 2 * k_wave;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
