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
// Verify arrive_and_drop() both orders prior writes and reduces the expected
// count for the next phase. Wave 0 drops after phase 0; wave 1 completes phase
// 1 with the reduced count.

#include <cuda/barrier>
#include "test_macros.h"
#include "hip_wavefront_size.h"

__device__ int test_arrive_and_drop_clean()
{
  int const k_wave  = get_wavefront_size();
  int const k_total = 2 * k_wave;

  __shared__ hip::barrier<hip::thread_scope_block> b;

  extern __shared__ int payload[];

  __shared__ int result_shared;

  if (threadIdx.x == 0)
  {
    init(&b, k_total);
    result_shared = 0;
  }
  __syncthreads();

  bool const is_drop_wave = threadIdx.x < k_wave;

  // Use plain writes so only barrier synchronization can publish the payload.
  payload[threadIdx.x] = threadIdx.x + 1;

  if (is_drop_wave)
  {
    // Wave 0 contributes to phase 0, then drops from phase 1.
    b.arrive_and_drop();
  }
  else
  {
    b.arrive_and_wait();

    // After wait(), wave 1 must observe all phase-0 payload writes.
    if (threadIdx.x == k_wave)
    {
      for (int i = 0; i < k_total; ++i)
      {
        if (payload[i] != i + 1)
        {
          result_shared = 1;  // Phase 0 memory ordering failed
        }
      }
    }
  }

  // Phase 1 uses the expected count reduced by wave 0's drops.
  if (!is_drop_wave)
  {
    payload[threadIdx.x] = threadIdx.x + 1 + k_total;
    b.arrive_and_wait();

    // The phase completing proves expected was reduced to k_wave.
    if (threadIdx.x == k_wave)
    {
      for (int i = k_wave; i < k_total; ++i)
      {
        if (payload[i] != i + 1 + k_total)
        {
          result_shared = 2;  // Phase 1 intra-wave visibility failed
        }
      }
    }
  }

  // Dropped threads wait for validation to complete.
  __syncthreads();

  return result_shared;
}

__device__ int test()
{
  return test_arrive_and_drop_clean();
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave  = get_wavefront_size();
    int const k_total = 2 * k_wave;

    size_t sharedMemSize = k_total * sizeof(int);

    cuda_block_count        = 1;
    cuda_thread_count       = k_total;
    cuda_shared_memory_size = sharedMemSize;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
