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
// Calls arrive_and_drop_reduce_impl with a block-scope barrier (__device__ global storage),
// 4 waves, 2N payload (all ones), expected_sum = 2 * n_threads.

#include <cuda/barrier>
#include "test_macros.h"
#include "../../../hip_barrier_test_utils.h"
#include "hip_wavefront_size.h"
#include "../../../functional_test_common/arrive_and_drop_reduce_impl.h"


using namespace hip_test;

static constexpr int k_n_waves = 4;

__device__ hip::barrier<hip::thread_scope_block> g_bar;

__device__ void kernel(int /*gpuIndex*/, int* pErrCode)
{
  int const k_wave_size   = get_wavefront_size();
  int const k_n_threads   = k_n_waves * k_wave_size;

  extern __shared__ int payload[];

  if (threadIdx.x == 0)
  {
    init(&g_bar, k_n_threads);
  }
  payload[threadIdx.x] = 1;
  payload[threadIdx.x + k_n_threads] = 1;
  __syncthreads();

  test_arrive_and_drop_reduce(
      g_bar, payload, static_cast<int>(threadIdx.x), k_wave_size, k_n_waves, 2 * k_n_threads, pErrCode);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size   = get_wavefront_size();
    int const k_n_threads   = k_n_waves * k_wave_size;
    cuda_block_count        = 1;
    cuda_thread_count       = k_n_threads;
    cuda_shared_memory_size = 2 * k_n_threads * sizeof(int);
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (
    kernel(hip_gpu_index, &errorCodes[hip_gpu_index]);
    return errorCodes[hip_gpu_index];
  ))
  return 0;
}
