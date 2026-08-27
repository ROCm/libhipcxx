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
// Verify wait_parity() observes completed phase parities across cycles.

#include <cuda/barrier>
#include "test_macros.h"
#include "hip_wavefront_size.h"

__device__ int test_basic_parity_wait()
{
  int const k_thread_count = get_wavefront_size();

  __shared__ hip::barrier<hip::thread_scope_block> barrier;

  if (threadIdx.x == 0)
  {
    init(&barrier, k_thread_count);
  }
  __syncthreads();

  barrier.arrive_and_wait();

  barrier.wait_parity(false);

  barrier.arrive_and_wait();

  barrier.wait_parity(true);

  return 0;
}

__device__ int test_late_wait()
{
  int const k_thread_count = get_wavefront_size();

  __shared__ hip::barrier<hip::thread_scope_block> barrier;

  if (threadIdx.x == 0)
  {
    init(&barrier, k_thread_count);
  }
  __syncthreads();

  barrier.arrive_and_wait();

  barrier.wait_parity(false);

  barrier.arrive_and_wait();

  barrier.wait_parity(true);

  return 0;
}

__device__ int test_multi_phase_sequence()
{
  int const k_thread_count = get_wavefront_size();

  __shared__ hip::barrier<hip::thread_scope_block> barrier;

  if (threadIdx.x == 0)
  {
    init(&barrier, k_thread_count);
  }
  __syncthreads();

  barrier.arrive_and_wait();
  barrier.wait_parity(false);

  barrier.arrive_and_wait();
  barrier.wait_parity(true);

  barrier.arrive_and_wait();
  barrier.wait_parity(false);

  barrier.arrive_and_wait();
  barrier.wait_parity(true);

  return 0;
}

__device__ int test_split_arrive_wait()
{
  int const k_thread_count = get_wavefront_size();

  __shared__ hip::barrier<hip::thread_scope_block> barrier;
  __shared__ hip::barrier<hip::thread_scope_block>::arrival_token token0;
  __shared__ hip::barrier<hip::thread_scope_block>::arrival_token token1;

  if (threadIdx.x == 0)
  {
    init(&barrier, k_thread_count);
  }
  __syncthreads();

  if (threadIdx.x == 0)
  {
    token0 = barrier.arrive();
  }
  else
  {
    (void) barrier.arrive();
  }

  barrier.wait_parity(false);

  if (threadIdx.x == 0)
  {
    token1 = barrier.arrive();
  }
  else
  {
    (void) barrier.arrive();
  }

  barrier.wait_parity(true);

  return 0;
}

__device__ int test()
{
  int result = 0;

  result = test_basic_parity_wait();
  if (result != 0) return result;

  result = test_late_wait();
  if (result != 0) return result;

  result = test_multi_phase_sequence();
  if (result != 0) return result;

  result = test_split_arrive_wait();
  if (result != 0) return result;

  return 0;
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_thread_count = get_wavefront_size();

    cuda_block_count = 1;
    cuda_thread_count = k_thread_count;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
