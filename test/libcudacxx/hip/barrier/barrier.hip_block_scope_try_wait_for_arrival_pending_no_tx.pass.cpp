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
// try_wait_for times out through the polling path while arrivals remain pending.

#include <cuda/barrier>
#include <cuda/std/chrono>
#include "test_macros.h"
#include "hip_wavefront_size.h"

__device__ int test_single_wave_arrival_pending_no_tx_expected()
{
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  __shared__ cuda::barrier<cuda::thread_scope_block>::arrival_token token;

  if (threadIdx.x == 0) {
    init(&bar, 2);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    token = bar.arrive();
  }
  __syncthreads();

  auto local_token = token;
  bool completed = bar.try_wait_for(cuda::std::move(local_token), cuda::std::chrono::microseconds(10));

  return completed ? 1 : 0;
}

__device__ int test()
{
  return test_single_wave_arrival_pending_no_tx_expected();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    cuda_block_count = 1;
    cuda_thread_count = get_wavefront_size();
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (return test();))
  return 0;
}
