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

// Functional test shared logic: multiphase barrier ordering across a scope boundary.
//
// Verifies three properties per phase:
//   1. Cross-boundary write ordering: each participant writes to the mirror slot
//      of its counterpart on the other side of the scope boundary. After wait(),
//      every participant verifies its own slot was written by the expected mirror.
//   2. Phase token alternates: the arrival token is 0 for even phases and
//      non-zero for odd phases, confirming the barrier's internal phase bit flips.
//   3. The barrier recycles correctly for n_phases iterations.
//
// Participant model:
//   n_threads participants should be split into two groups across the scope boundary.
//   Each participant has a unique thread_id in [0, n_threads).
//
//   Each phase each thread:
//     writes to its mirror's slot
//     arrive() -> token
//     verifies the token value
//     waits on token
//     verifies it's slot now contains its mirror's write     
//
// Per-phase payload slices (size = n_phases * n_threads) eliminate the race
// between phase P's post-wait read and phase P+1's pre-arrive write.
//
// Caller contract:
//   - bar must already be initialised with expected count = n_threads.
//   - payload is an int array of size [n_phases * n_threads], accessible by
//     all participants (shared memory for block scope, managed memory for
//     system scope, etc) initialized to -1s
//   - thread_id is the caller's unique index in [0, n_threads).
//   - n_threads is the total participant count (= barrier's expected count).
//   - n_phases is the number of phases to run (>= 3 recommended).
//   - Errors are written atomically via recordIfNeq into *pErrCode.
//     The function does NOT return early — all phases run regardless of errors,
//     so all participants continue to arrive and no deadlock occurs.
//
// AMD warp-divergence safety:
//   arrive() and wait() are called unconditionally by all participants.
//   The payload write and verification are indexed by thread_id — no branch
//   on thread_id at barrier call sites. Safe for warp-uniform execution.

#ifndef LIBHIPCXX_BARRIER_FUNCTIONAL_ARRIVE_AND_WAIT_MULTIPHASE_ORDERING_IMPL_H
#define LIBHIPCXX_BARRIER_FUNCTIONAL_ARRIVE_AND_WAIT_MULTIPHASE_ORDERING_IMPL_H

#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include "../hip_barrier_test_utils.h"

namespace hip_test {

// Error codes: 100 + phase for token mismatch, 200 + phase for payload mismatch.

template <class B>
__host__ __device__ void test_arrive_and_wait_multiphase_ordering(
    B&   bar,
    int* payload,
    int  thread_id,
    int  n_threads,
    int  n_phases,
    int* pErrCode)
{
  int const mirror_id = n_threads - 1 - thread_id;

  for (int phase = 0; phase < n_phases; ++phase)
  {
    // Write this thread's value into the mirror's slot for this phase.
    // The mirror reads it back after wait() to verify cross-boundary ordering.
    int const base             = phase * n_threads;
    payload[base + mirror_id] = thread_id;

    typename B::arrival_token token = bar.arrive();

    // Even phases: token phase bit = 0. Odd phases: token phase bit = 1.
    bool const expected_phase_is_odd = (phase % 2 != 0);
    bool const token_is_odd_phase  = (static_cast<hip::std::uint64_t>(token)) != 0;
    recordIfNeq(token_is_odd_phase, expected_phase_is_odd, 100 + phase, pErrCode);

    bar.wait(hip::std::move(token));

    // After wait: payload[base + thread_id] must hold the value written by
    // the mirror thread (mirror_id). This read is in the same phase slice as
    // the write above, but into a different slot — no race.
    recordIfNeq(payload[base + thread_id], mirror_id, 200 + phase, pErrCode);
  }
}

} // namespace hip_test

#endif // LIBHIPCXX_BARRIER_FUNCTIONAL_ARRIVE_AND_WAIT_MULTIPHASE_ORDERING_IMPL_H
