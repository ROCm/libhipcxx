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

// ADDITIONAL_COMPILE_FLAGS: -DNO_MAIN_REPLACEMENT

// UNSUPPORTED: nvcc, nvhpc, nvc++

// <hip/barrier>
// Verify a two-wave producer-consumer workflow with double buffering.

#include <cassert>

#include "test_macros.h"
#include "hip_wavefront_size.h"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/barrier>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

int constexpr DEFAULT_DEVICE = 0;

using barrier_t = hip::barrier<hip::thread_scope_block>;

__device__ int threadIdxInWarp() {
  int const WAVE_SIZE = get_wavefront_size();
  return threadIdx.x % WAVE_SIZE;
}

__device__ void produce(barrier_t ready[], barrier_t filled[], float* buffer, int buffer_len, int N)
{
  for (int i = 0; i < N / buffer_len; ++i)
  {
    ready[i % 2].arrive_and_wait();
    int const idx = (i % 2) * buffer_len + threadIdxInWarp();
    int const val = i * buffer_len + threadIdxInWarp(); 
    buffer[idx]   = static_cast<float>(val);
    barrier_t::arrival_token token = filled[i % 2].arrive();
  }
}

__device__ void consume(barrier_t ready[], barrier_t filled[], float* buffer, int buffer_len, float* out, int N)
{
  barrier_t::arrival_token token1 = ready[0].arrive();
  barrier_t::arrival_token token2 = ready[1].arrive();
  for (int i = 0; i < N / buffer_len; ++i)
  {
    filled[i % 2].arrive_and_wait();
    int const ldsIdx                = (i % 2) * buffer_len + threadIdxInWarp();
    int const outIdx                = i * buffer_len + threadIdxInWarp();
    out[outIdx]                     = buffer[ldsIdx];
    barrier_t::arrival_token token3 = ready[i % 2].arrive();
  }
}

__global__ void producer_consumer_pattern(int N, float* out)
{
  int const WAVE_SIZE = get_wavefront_size();

  // Single block with two waves.
  assert(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0);
  assert(blockDim.x == 2 * WAVE_SIZE && blockDim.y == 1 && blockDim.z == 1);

  // Double buffer: buffer_0 starts at buffer, buffer_1 at buffer + WAVE_SIZE.
  extern __shared__ float buffer[];

  // bar[0..1] track ready buffers; bar[2..3] track filled buffers.
  __shared__ barrier_t bar[4];

  if (threadIdx.x < 4)
  {
    init(bar + threadIdx.x, blockDim.x);
  }
  __syncthreads();

  if (threadIdx.x < WAVE_SIZE)
  {
    produce(bar, bar + 2, buffer, WAVE_SIZE, N);
  }
  else
  {
    consume(bar, bar + 2, buffer, WAVE_SIZE, out, N);
  }
}

int main(int argc, char** argv)
{
  hipError_t err;
  hipDeviceProp_t prop;
  HIP_CALL(err, hipGetDeviceProperties(&prop, DEFAULT_DEVICE));

  int const WAVE_SIZE = prop.warpSize;

  int const NUM_ITERS = 4;
  int const NUM_ELEMS = WAVE_SIZE * NUM_ITERS;
  int const NUM_BYTES = NUM_ELEMS * sizeof(int);

  size_t sharedMemSize = 2 * WAVE_SIZE * sizeof(float);

  float* dOut;
  HIP_CALL(err, hipMalloc(&dOut, NUM_ELEMS * sizeof(float)));

  producer_consumer_pattern<<<1, 2 * WAVE_SIZE, sharedMemSize>>>(NUM_ELEMS, dOut);
  HIP_CALL(err, hipDeviceSynchronize());
  HIP_CALL(err, hipGetLastError());

  for (int i = 0; i < NUM_ELEMS; ++i)
  {
    assert(dOut[i] == i);
  }

  HIP_CALL(err, hipFree(dOut));
  return 0;
}
