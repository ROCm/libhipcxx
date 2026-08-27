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
// Verify a completion function can advance per-phase pipeline state.

#include <cuda/barrier>
#include "test_macros.h"
#include "hip_wavefront_size.h"

static constexpr int k_num_iterations = 3;

enum PipelinePhase {
  LOAD = 0,
  COMPUTE = 1,
  STORE = 2
};

__device__ int test()
{
  int const k_wave_size = get_wavefront_size();
  int const k_thread_count = 2 * k_wave_size;

  __shared__ PipelinePhase current_phase;
  extern __shared__ int shared_buffers[];
  int* load_buffer = shared_buffers;
  int* compute_buffer = shared_buffers + k_thread_count;
  __shared__ int store_count;

  auto completion = [&]()
  {
    current_phase = static_cast<PipelinePhase>((current_phase + 1) % 3);
  };

  using barrier_t = cuda::barrier<cuda::thread_scope_block, decltype(completion)>;
  __shared__ alignas(barrier_t) char bar_storage[sizeof(barrier_t)];
  barrier_t* bar = reinterpret_cast<barrier_t*>(bar_storage);

  if (threadIdx.x == 0)
  {
    init(bar, k_thread_count, completion);
    current_phase = LOAD;
    store_count = 0;
  }
  __syncthreads();  // Legitimate: ensures init is visible

  for (int iteration = 0; iteration < k_num_iterations; iteration++)
  {
    if (current_phase != LOAD) return 1;

    load_buffer[threadIdx.x] = iteration * 1000 + threadIdx.x;

    bar->arrive_and_wait();  // Completion advances to COMPUTE

    if (current_phase != COMPUTE) return 2;

    compute_buffer[threadIdx.x] = load_buffer[threadIdx.x] * 2;

    bar->arrive_and_wait();  // Completion advances to STORE

    if (current_phase != STORE) return 3;

    if (threadIdx.x == 0)
    {
      store_count++;
      for (int i = 0; i < k_thread_count; i++)
      {
        int expected = (iteration * 1000 + i) * 2;
        if (compute_buffer[i] != expected) return 4;
      }
    }

    bar->arrive_and_wait();  // Completion advances back to LOAD for next iteration
  }

  if (current_phase != LOAD) return 5;
  if (store_count != k_num_iterations) return 6;

  return 0;
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size = get_wavefront_size();
    int const k_thread_count = 2 * k_wave_size;

    size_t sharedMemSize = 2 * k_thread_count * sizeof(int);

    cuda_block_count = 1;
    cuda_thread_count = k_thread_count;
    cuda_shared_memory_size = sharedMemSize;
  ))

  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
