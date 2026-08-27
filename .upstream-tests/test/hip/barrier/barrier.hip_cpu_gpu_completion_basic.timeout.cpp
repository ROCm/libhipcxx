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
// REQUIRES: enable_undefined_behavior_tests

// <hip/barrier>
// All parties arrive_and_wait on a system-scope barrier with a completion
// function.
// Demonstrates that a system scope barrier will only have a completion function
// for either a host or a device (never both).
// What this means practically is that:
// - A system scope barrier (bootstrapped as demonstrated) may only have a __host__ completion function
// - A system scope barrier with a CPU participant may only utilize a completion function if the program guarantees that
//   the host will be the one to complete the barrier

#include <cstdlib>
#include <hip/barrier>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "hip_barrier_test_utils.h"
#include "hip_wavefront_size.h"

using barrier_t = hip::barrier<hip::thread_scope_system, void(*)()>;
#include "hip_cpu_gpu_global_data.h"

using namespace hip_test;

// The host inits the barrier, so the function pointer passed must be a __host__ function pointer
__host__ void do_completion()
{
  auto* pGlobalData = reinterpret_cast<GlobalData*>(pRawGlobalData);
  __atomic_fetch_add(&pGlobalData->x, 1, __ATOMIC_RELAXED);
}

__device__ void kernel(int /*gpuIndex*/, int* pErrCode)
{
  auto* pGlobalData = reinterpret_cast<GlobalData*>(pRawGlobalData);
  // If the device arrive completes the barrier, it will attempt to call a host function pointer and the behavior is
  // undefined
  pGlobalData->barrier.arrive_and_wait();
  recordIfNeq(1, pGlobalData->x, EXIT_FAILURE, pErrCode);
}

__host__ void hostSidePrep()
{
  GlobalData* pGlobalData;
  hipMallocManaged(&pGlobalData, sizeof(GlobalData));
  pGlobalData->x = 0;
  pRawGlobalData = pGlobalData;
  init(&pGlobalData->barrier,
       k_num_host_arrivals + HIP_GPU_COUNT * cuda_block_count * cuda_thread_count,
       do_completion);
}

__host__ int hostSideWork()
{
  auto* pGlobalData = reinterpret_cast<GlobalData*>(pRawGlobalData);
  // If the host's arrive completes the barrier, the host completion function is called and there is no issue
  pGlobalData->barrier.arrive_and_wait();
  return pGlobalData->x == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
}

HIP_CPU_GPU_TEST_MAIN()
