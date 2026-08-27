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
// Functional test: device-scope barrier with __device__ global storage and completion function.
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

using barrier_t = hip::barrier<hip::thread_scope_device, void (*)()>;

static constexpr int k_n_phases = 4;

__device__ alignas(barrier_t) hip::std::byte g_bar_raw[sizeof(barrier_t)];
__device__ int*                              g_payload;

__device__ void kernel(int /*gpuIndex*/, int* pErrCode)
{
  auto* g_bar = reinterpret_cast<barrier_t*>(g_bar_raw);

  int const k_wave_size = get_wavefront_size();
  int const k_n_threads = 2 * k_wave_size;
  int const thread_id   = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  init_barrier(g_bar, k_n_threads, barrier_no_op_completion);
  cooperative_groups::this_grid().sync();

  test_arrive_and_wait_multiphase_ordering(
      *g_bar, g_payload, thread_id, k_n_threads, k_n_phases, pErrCode);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const k_wave_size = get_wavefront_size();
    int const k_n_threads = 2 * k_wave_size;

    int* d_payload_ptr;
    hipMalloc(&d_payload_ptr, k_n_phases * k_n_threads * sizeof(int));
    hipMemset(d_payload_ptr, 0xFF, k_n_phases * k_n_threads * sizeof(int));
    hipMemcpyToSymbol(HIP_SYMBOL(g_payload), &d_payload_ptr, sizeof(d_payload_ptr));

    cuda_block_count  = 2;
    cuda_thread_count = k_wave_size;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (
    kernel(hip_gpu_index, &errorCodes[hip_gpu_index]);
    return errorCodes[hip_gpu_index];
  ))
  return 0;
}
