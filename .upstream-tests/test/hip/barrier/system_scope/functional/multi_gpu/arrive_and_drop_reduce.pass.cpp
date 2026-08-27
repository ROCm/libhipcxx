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
// ADDITIONAL_COMPILE_FLAGS: -DLIBHIPCXX_GPU_COUNT=2

// <hip/barrier>
//
// Calls arrive_and_drop_reduce_impl with a system-scope barrier across 2 GPUs,
// 2 waves per GPU (4 total). 2N payload (all ones), expected_sum = 2 * n_threads.
// thread_id = gpu_index * k_waves_per_gpu * wave_size + threadIdx.x.

#include <cstdlib>
#include <hip/barrier>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "../../../hip_barrier_test_utils.h"
#include "hip_wavefront_size.h"
#include "../../../functional_test_common/arrive_and_drop_reduce_impl.h"

using namespace hip_test;

using barrier_t = hip::barrier<hip::thread_scope_system>;

static constexpr int k_waves_per_gpu = 2;
static constexpr int k_n_waves       = HIP_GPU_COUNT * k_waves_per_gpu;

struct GlobalData
{
  barrier_t barrier;
  int       payload[]; ///< payload[0..2*n_threads]: allocated via hipMallocManaged.
};

__device__ void kernel(int gpuIndex, int* pErrCode)
{
  int const k_wave_size   = get_wavefront_size();
  int const k_n_threads   = k_n_waves * k_wave_size;

  auto* g             = reinterpret_cast<GlobalData*>(pRawGlobalData);
  int const thread_id = gpuIndex * k_waves_per_gpu * k_wave_size + static_cast<int>(threadIdx.x);

  test_arrive_and_drop_reduce(
      g->barrier, g->payload, thread_id, k_wave_size, k_n_waves, 2 * k_n_threads, pErrCode);
}

__host__ void hostSidePrep()
{
  int const k_wave_size     = get_wavefront_size();
  int const k_n_threads     = k_n_waves * k_wave_size;
  int const k_payload_slots = 2 * k_n_threads;

  size_t const alloc_size =
      offsetof(GlobalData, payload) + k_payload_slots * sizeof(int);
  GlobalData* g;
  hipMallocManaged(&g, alloc_size);
  for (int i = 0; i < k_payload_slots; ++i)
  {
    g->payload[i] = 1;
  }
  init(&g->barrier, k_n_threads);
  pRawGlobalData = g;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
               int const k_wave_size = get_wavefront_size();
               cuda_block_count = 1;
               cuda_thread_count = k_waves_per_gpu * k_wave_size;
               hostSidePrep();
               return 0;
              ))
  NV_IF_TARGET(NV_IS_DEVICE, (
               kernel(hip_gpu_index, &errorCodes[hip_gpu_index]);
               return errorCodes[hip_gpu_index];
              ))
}
