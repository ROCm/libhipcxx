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
// Verify arrive_and_drop() can trigger completion and reduce the expected count.

#include <cuda/barrier>
#include "test_macros.h"
#include "hip_wavefront_size.h"

__device__ int test()
{
  __shared__ int completion_counter;

  auto completion = [&]()
  {
    completion_counter++;
  };

  using barrier_t = cuda::barrier<cuda::thread_scope_block, decltype(completion)>;
  __shared__ alignas(alignof(barrier_t)) char bar_storage[sizeof(barrier_t)];
  barrier_t* bar = reinterpret_cast<barrier_t*>(bar_storage);

  int const k_wave         = get_wavefront_size();
  int const k_thread_count = 2 * k_wave;  // 2 waves

  if (threadIdx.x == 0)
  {
    init(bar, k_thread_count, completion);
    completion_counter = 0;
  }
  __syncthreads();  // Legitimate: ensures init is visible

  int wave_id = threadIdx.x / k_wave;

  bar->arrive_and_wait();

  if (completion_counter != 1) return 1;

  if (wave_id == 0)
  {
    bar->arrive_and_drop();
  }
  else
  {
    bar->arrive_and_wait();

    if (completion_counter != 2) return 2;
  }

  return 0;
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave         = get_wavefront_size();
    int const k_thread_count = 2 * k_wave;
    cuda_block_count         = 1;
    cuda_thread_count        = k_thread_count;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
