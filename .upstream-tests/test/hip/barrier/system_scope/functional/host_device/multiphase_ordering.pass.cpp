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
// Functional test: system-scope barrier with host and 1 GPU warp.
// n_threads = k_wave_size + 1. GPU threads have ids [0, k_wave_size),
// host has id k_wave_size. The host's mirror is thread 0 — they write into
// each other's slots across the host/GPU boundary. Runs 4 phases.

#include <cstdlib>
#include <hip/barrier>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "../../../hip_barrier_test_utils.h"
#include "hip_wavefront_size.h"
#include "../../../functional_test_common/multiphase_ordering_impl.h"

using namespace hip_test;

using barrier_t = hip::barrier<hip::thread_scope_system>;

static constexpr int k_n_phases = 4;

struct GlobalData
{
  barrier_t barrier;
  int       payload[]; ///< payload[0..n_threads): allocated via hipMallocManaged.
};

__device__ void kernel(int /*gpuIndex*/, int* pErrCode)
{
  int const k_wave_size = get_wavefront_size();
  int const k_n_threads = k_wave_size + k_num_host_arrivals;

  auto* g = reinterpret_cast<GlobalData*>(pRawGlobalData);

  test_arrive_and_wait_multiphase_ordering(
      g->barrier,
      g->payload,
      static_cast<int>(threadIdx.x), // GPU ids: [0, k_wave_size)
      k_n_threads,
      k_n_phases,
      pErrCode);
}

__host__ int hostSideWork()
{
  int const k_wave_size = get_wavefront_size();
  int const k_n_threads = k_wave_size + k_num_host_arrivals;

  auto* g = reinterpret_cast<GlobalData*>(pRawGlobalData);
  int err = 0;

  test_arrive_and_wait_multiphase_ordering(
      g->barrier,
      g->payload,
      k_wave_size, // host id = k_wave_size (mirrors thread 0)
      k_n_threads,
      k_n_phases,
      &err);

  return err;
}

__host__ void hostSidePrep()
{
  int const k_wave_size = get_wavefront_size();
  int const k_n_threads = k_wave_size + k_num_host_arrivals;

  size_t const alloc_size =
      offsetof(GlobalData, payload) + k_n_phases * k_n_threads * sizeof(int);
  GlobalData* g;
  hipMallocManaged(&g, alloc_size);
  for (int i = 0; i < k_n_phases * k_n_threads; ++i)
  {
    g->payload[i] = -1;
  }
  init(&g->barrier, k_n_threads);
  pRawGlobalData = g;
}

HIP_CPU_GPU_TEST_MAIN()
