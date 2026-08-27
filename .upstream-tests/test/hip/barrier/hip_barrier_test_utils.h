//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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

#ifndef LIBHIPCXX_HIP_BARRIER_TEST_UTILS
#define LIBHIPCXX_HIP_BARRIER_TEST_UTILS

namespace hip_test {

static constexpr int k_num_host_arrivals = 1;

__host__ __device__ inline void barrier_no_op_completion() {}

// init_barrier: calls 3-arg init for real completion types, 2-arg for __empty_completion.
template <hip::thread_scope Scope, class CompletionF>
__host__ __device__ void init_barrier(hip::barrier<Scope, CompletionF>* bar, hip::std::ptrdiff_t n, CompletionF fn)
{
  init(bar, n, fn);
}
template <hip::thread_scope Scope>
__host__ __device__ void init_barrier(hip::barrier<Scope>* bar, hip::std::ptrdiff_t n, hip::std::__empty_completion)
{
  init(bar, n);
}

// HIP_CPU_GPU_TEST_MAIN — generates the test's main() for CPU/GPU barrier tests.
//
// The test file writes a main() that is transformed by force_include_hip.h
// (#define main __host__ __device__ fake_main). The harness then drives the
// following call flow:
//
//   harness main()
//     -> fake_main() [host path]          (sets up thread counts and barriers)
//         -> hostSidePrep()               (allocate managed memory, init barrier)
//         -> hostSideWorkFunc = hostSideWork
//     -> fake_main_kernel<<<...>>>()      (launches GPU kernel)
//         -> fake_main() [device path]    (device half of the same function)
//             -> kernel(gpuIndex, pErrCode)
//     -> hostSideWorkFunc()               (runs concurrently with kernel)
//
// Each test file MUST define these three callbacks:
//   void hostSidePrep()
//     Called on the host before the kernel launches. Allocates managed memory,
//     initialises the barrier, and stores the pointer in pRawGlobalData.
//
//   __host__ int hostSideWork()
//     The host's participation in the test (arrives at the barrier, validates
//     results). Return EXIT_SUCCESS (0) or EXIT_FAILURE (non-zero).
//
//   __device__ void kernel(int gpuIndex, int* pErrCode)
//     The device's participation in the test. Record errors with recordIfNeq()
//     rather than returning early, so threads that must still arrive at the
//     barrier can continue to do so.
//
// NOTE: This macro makes use of many global variables defined in
//       force_include_hip.h
#define HIP_CPU_GPU_TEST_MAIN()                          \
  int main(int, char**)                                  \
  {                                                      \
    NV_IF_TARGET(NV_IS_HOST, (                           \
      int const k_wave_size = get_wavefront_size();      \
      cuda_block_count      = 1;                         \
      cuda_thread_count     = 1 * k_wave_size;           \
      hostSidePrep();                                    \
      hostSideWorkFunc = hostSideWork;                   \
      return 0;                                          \
    ))                                                   \
    NV_IF_TARGET(NV_IS_DEVICE, (                         \
      kernel(hip_gpu_index, &errorCodes[hip_gpu_index]); \
      return errorCodes[hip_gpu_index];                  \
    ))                                                   \
    return 0;                                            \
  }

#define SINGLE_PRINTF(...) \
  if (threadIdx.x == 0)    \
  {                        \
    printf(__VA_ARGS__);   \
  }

// Record errCode into *pErrCode if a != b and no error is already set.
// Does NOT return early — use this when the caller must still participate in barriers.
// Device: uses atomicCAS (multiple threads may race to record).
// Host: uses a plain conditional store (host participant is always single-threaded).
template <typename T>
__host__ __device__ void recordIfNeq(T a, T b, int errCode, int* const pErrCode)
{
  if (a != b)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (atomicCAS(pErrCode, 0, errCode);))
    NV_IF_TARGET(NV_IS_HOST,   (if (*pErrCode == 0) *pErrCode = errCode;))
  }
}

} // namespace hip_test

#endif // LIBHIPCXX_HIP_BARRIER_TEST_UTILS