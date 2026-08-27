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
// ADDITIONAL_COMPILE_FLAGS: -DUSE_COOPERATIVE_LAUNCH

// <cuda/barrier>
//
// Functional test: device-scope barrier with __device__ global storage.
// 2 blocks of 1 warp each participate across the inter-block boundary. Each
// thread writes into its mirror's slot, arrives, verifies the token phase,
// waits, then verifies the mirror's write is visible. Runs 4 phases.

#include <cuda/barrier>
#include <hip/hip_runtime.h>
#include "test_macros.h"
#include "../../../hip_barrier_test_utils.h"
#include "hip_wavefront_size.h"
#include "../../../functional_test_common/multiphase_ordering_impl.h"
#include "hip/hip_cooperative_groups.h"

using namespace hip_test;

using barrier_t = hip::barrier<hip::thread_scope_device>;

static constexpr int k_n_phases = 4;

__device__ barrier_t g_bar;
__device__ int* g_payload;

__device__ void kernel(int /*gpuIndex*/, int* pErrCode)
{
  int const k_wave_size = get_wavefront_size();
  int const k_n_threads = 2 * k_wave_size;
  auto const group = cooperative_groups::this_grid();

  if (group.thread_rank() == 0)
  {
    g_payload = new int[k_n_phases * k_n_threads];
    init(&g_bar, k_n_threads);
  }
  group.sync();
  for (unsigned int i = 0; i < k_n_phases; ++i)
  {
    g_payload[i * k_n_threads + group.thread_rank()] = -1;
  }
  group.sync();

  test_arrive_and_wait_multiphase_ordering(
      g_bar, g_payload, group.thread_rank(), k_n_threads, k_n_phases, pErrCode);

  group.sync();
  if (group.thread_rank() == 0) {
    delete[] g_payload;
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    cuda_block_count  = 2;
    cuda_thread_count = get_wavefront_size();
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (
    kernel(hip_gpu_index, &errorCodes[hip_gpu_index]);
    return errorCodes[hip_gpu_index];
  ))
  return 0;
}
