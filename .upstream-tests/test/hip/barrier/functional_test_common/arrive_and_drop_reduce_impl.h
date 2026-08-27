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

// Functional test shared logic: wave-granular tree reduction using arrive_and_drop().
// Verifies ordering (dropped-thread writes visible after wait) and count reduction
// (barrier unblocks with the correct reduced count each phase, no deadlock).
//
// Caller contract (2N-payload model):
//   data        - 2 * n_threads ints, all initialized before call (typically to 1).
//   bar         - initialized with n_threads (actual participant count).
//   n_waves     - n_threads / wave_size; must be a power of two >= 2.
//   wave_size   - threads per wave; n_threads must be a power-of-two multiple.
//   expected_sum - data[0] value after reduction (typically 2 * n_threads).
//
// Each inter-wave phase: data[tid] += data[tid + active_threads], upper half drops.
// Final intra-wave phase folds within the surviving wave (no barrier needed).

#ifndef LIBHIPCXX_BARRIER_FUNCTIONAL_ARRIVE_AND_DROP_REDUCE_IMPL_H
#define LIBHIPCXX_BARRIER_FUNCTIONAL_ARRIVE_AND_DROP_REDUCE_IMPL_H

#include "../hip_barrier_test_utils.h"

namespace hip_test {

template <class B>
__host__ __device__ void test_arrive_and_drop_reduce(
    B&   bar,
    int* data,
    int  thread_id,
    int  wave_size,
    int  n_waves,
    int  expected_sum,
    int* pErrCode)
{
  for (int active_waves = n_waves; active_waves > 1; active_waves /= 2)
  {
    int const active_threads = active_waves * wave_size;
    int const half           = active_threads / 2;

    if (thread_id >= active_threads)
    {
      return; // already dropped in a prior phase
    }
    data[thread_id] += data[thread_id + active_threads];

    if (thread_id >= half)
    {
      bar.arrive_and_drop();
      return;
    }

    bar.arrive_and_wait();
  }

  // No need for sync because all threads in a wave are in lock-step
  for (int half = wave_size; half > 0; half /= 2)
  {
    if (thread_id < half)
    {
      data[thread_id] += data[thread_id + half];
    }
  }

  // One wave remains. Thread 0 checks the accumulated sum.
  if (thread_id == 0)
  {
    recordIfNeq(data[0], expected_sum, 1, pErrCode);
  }
}
}; // namespace hip_test

#endif // LIBHIPCXX_BARRIER_FUNCTIONAL_ARRIVE_AND_DROP_REDUCE_IMPL_H
