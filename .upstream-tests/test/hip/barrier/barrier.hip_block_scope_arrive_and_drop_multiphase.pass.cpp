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
// Verify arrive_and_drop() shrinks the expected count across phases.
// Participation is wave-convergent: each phase drops one whole wave while the
// remaining waves wait, avoiding same-wave split arrive/wait progress hazards.
// Each phase uses a separate payload slice so validation does not race later
// writes from waves that continue participating.

#include <cuda/barrier>
#include "test_macros.h"
#include "hip_wavefront_size.h"

// Start with two waves and drop one per phase.
static constexpr int k_waves   = 2;
static constexpr int k_phases  = k_waves;

__device__ int test_arrive_and_drop_multiphase()
{
  int const k_wave_size = get_wavefront_size();
  int const k_initial   = k_waves * k_wave_size;

  __shared__ hip::barrier<hip::thread_scope_block> b;

  // Layout: payload[k_phases][k_initial].
  extern __shared__ int payload[];

  if (threadIdx.x == 0)
    init(&b, k_initial);
  __syncthreads();

  int const wave_id = threadIdx.x / k_wave_size;

  for (int phase = 0; phase < k_phases; ++phase)
  {
    int const active_waves = k_waves - phase;
    if (wave_id >= active_waves)
    {
      // Waves that dropped out must not re-enter later phases.
      return 1;
    }

    // Each phase writes to its own slice.
    payload[phase * k_initial + threadIdx.x] = phase + 1;

    int const drop_wave = active_waves - 1;

    // Drop exactly one whole wave per phase, except the final phase.
    if (active_waves > 1 && wave_id == drop_wave)
    {
      b.arrive_and_drop();
      break; // this wave no longer participates in subsequent phases
    }

    b.arrive_and_wait();

    // Stores made before any arrive() in this phase must be visible after wait().
    if (threadIdx.x == 0)
    {
      int const active_threads = active_waves * k_wave_size;
      for (int i = 0; i < active_threads; ++i)
      {
        if (payload[phase * k_initial + i] != phase + 1)
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
  return test_arrive_and_drop_multiphase();
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size  = get_wavefront_size();
    int const k_initial    = k_waves * k_wave_size;

    size_t sharedMemSize = k_phases * k_initial * sizeof(int);

    cuda_block_count       = 1;
    cuda_thread_count      = k_initial;
    cuda_shared_memory_size = sharedMemSize;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
