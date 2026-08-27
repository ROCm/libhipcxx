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

// <hip/barrier>
// Contrast stale-token bypass with a real-token wait to prove acquire ordering.
// The stale-token path observes the pre-update sentinel; the real-token path
// observes the value written before arrive().

#include <hip/barrier>
#include "test_macros.h"
#include "hip_wavefront_size.h"

__device__ int test()
{
  int const k_wave_size = get_wavefront_size();
  int const k_thread_count = 2 * k_wave_size;

  __shared__ hip::barrier<hip::thread_scope_block> bar;
  __shared__ int sentinel;
  __shared__ int result;

  if (threadIdx.x == 0)
  {
    init(&bar, k_thread_count);
    sentinel = 0;
    result = 0;
  }
  __syncthreads();

  // Reuse the same initialized barrier across phases; no reinit is needed.
  auto token0 = bar.arrive();
  auto stale_token = token0;
  bar.wait(hip::std::move(token0));

  auto token1 = bar.arrive();
  bar.wait(hip::std::move(token1));

  auto token2 = bar.arrive();
  auto later_phase_zero_token = token2;
  bar.wait(hip::std::move(token2));

  if (stale_token != later_phase_zero_token)
  {
    atomicCAS(&result, 0, 1);
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    sentinel = 400;
  }

  auto token = bar.arrive();
  bar.wait(hip::std::move(token));

  int observed = sentinel;
  if (observed != 400)
  {
    atomicCAS(&result, 0, 2);
  }
  __syncthreads();

  return result;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size = get_wavefront_size();
    int const k_thread_count = 2 * k_wave_size;

    cuda_block_count  = 1;
    cuda_thread_count = k_thread_count;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (
    int result = test();
    return result;
  ))

  return 0;
}