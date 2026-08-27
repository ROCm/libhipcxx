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
//
// System-scope host+device arrive_and_drop tree reduction.
//
// k_n_cpu_virt == k_n_waves: GPU waves and CPU virtuals halve in lockstep.
// Barrier init = k_n_waves + k_n_cpu_virt. Data is 2N (all 1s); stride = active_gpu + active_cpu.
// Final phase: CPU folds into data[0] and drops; GPU wave arrive_and_waits, does intra-wave, checks.
// expected_sum = 2 * (k_n_gpu + k_n_cpu_virt)

#include <cstdlib>
#include <hip/barrier>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "../../../hip_barrier_test_utils.h"
#include "hip_wavefront_size.h"

using namespace hip_test;

using barrier_t = hip::barrier<hip::thread_scope_system>;

static constexpr int k_n_waves    = 4; // must be a power of two
static constexpr int k_n_cpu_virt = k_n_waves;

struct GlobalData
{
  barrier_t barrier;
  int       data[]; // 2*(k_n_gpu + k_n_cpu_virt) elements, all initialized to 1
};

__device__ void kernel(int /*gpuIndex*/, int* pErrCode)
{
  int const wave_size   = get_wavefront_size();
  int const k_n_gpu     = k_n_waves * wave_size;
  auto*     g           = reinterpret_cast<GlobalData*>(pRawGlobalData);
  int const tid         = static_cast<int>(threadIdx.x);

  // Inter-wave phases. group_size = active_gpu + active_cpu; halves each iteration.
  int active_gpu = k_n_gpu;
  int active_cpu = k_n_cpu_virt;
  int group_size = k_n_gpu + k_n_cpu_virt;
  for (; active_gpu > wave_size; active_gpu /= 2, active_cpu /= 2, group_size /= 2)
  {
    if (tid >= active_gpu)
    {
      return; // already dropped in a prior phase
    }
    g->data[tid] += g->data[tid + group_size];

    int const half = active_gpu / 2;
    if (tid >= half)
    {
      g->barrier.arrive_and_drop();
      return;
    }
    g->barrier.arrive_and_wait();
  }

  // Full wave fold while 1 host thread is still active
  g->data[tid] += g->data[tid + group_size];
  g->barrier.arrive_and_wait();

  // Wait for host to fold it's final partial sum into data[0]
  g->barrier.arrive_and_wait();

  // Finish intra-wave reduction; data[0] already includes the CPU contribution.
  for (int half = wave_size / 2; half > 0; half /= 2)
  {
    if (tid < half)
    {
      g->data[tid] += g->data[tid + half];
    }
  }

  if (tid == 0)
  {
    int const expect_sum = 2 * (k_n_gpu + k_n_cpu_virt);
    recordIfNeq(g->data[0], expect_sum, 1, pErrCode);
  }
}

__host__ int hostSideWork()
{
  auto*     g         = reinterpret_cast<GlobalData*>(pRawGlobalData);
  int const wave_size = get_wavefront_size();
  int const k_n_gpu   = k_n_waves * wave_size;

  // Inter-wave phases: mirror the GPU loop using the same group_size halving.
  int active_cpu = k_n_cpu_virt;
  int active_gpu = k_n_gpu;
  int group_size = k_n_gpu + k_n_cpu_virt;
  while (active_cpu > 1)
  {
    int const half_cpu = active_cpu / 2;

    // Every active CPU virtual folds using the shared group_size stride.
    // CPU virtual j has logical index active_gpu + j in the flat data array.
    for (int j = 0; j < active_cpu; ++j)
    {
      g->data[active_gpu + j] += g->data[active_gpu + j + group_size];
    }

    // Upper half drops (no bulk arrive_and_drop), lower half bulk-arrives then waits.
    for (int i = 0; i < half_cpu; ++i)
    {
      g->barrier.arrive_and_drop();
    }
    auto token = g->barrier.arrive(half_cpu);
    g->barrier.wait(std::move(token));

    active_cpu  = half_cpu;
    active_gpu /= 2;
    group_size /= 2;
  }

  // Final fold while 1 GPU wave is still active
  g->data[active_gpu] += g->data[active_gpu + group_size];
  g->barrier.arrive_and_wait();

  // Fold host's final partial sum into data[0] while GPU waits
  g->data[0] += g->data[active_gpu];
  g->barrier.arrive_and_drop();
  return 0;
}

__host__ void hostSidePrep()
{
  int const wave_size   = get_wavefront_size();
  int const k_n_gpu     = k_n_waves * wave_size;
  int const total       = k_n_gpu + k_n_cpu_virt;

  size_t const alloc_size = offsetof(GlobalData, data) + 2 * total * sizeof(int);
  GlobalData*  g;
  hipMallocManaged(&g, alloc_size);

  for (int i = 0; i < 2 * total; ++i)
  {
    g->data[i] = 1;
  }
  init(&g->barrier, k_n_gpu + k_n_cpu_virt);
  pRawGlobalData = g;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size = get_wavefront_size();
    cuda_block_count      = 1;
    cuda_thread_count     = k_n_waves * k_wave_size;
    hostSidePrep();
    hostSideWorkFunc = hostSideWork;
    return 0;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (
    kernel(hip_gpu_index, &errorCodes[hip_gpu_index]);
    return errorCodes[hip_gpu_index];
  ))
  return 0;
}
