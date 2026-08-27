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

// <cuda/barrier>
// DOCUMENTATION TEST: arrive(__update > expected) for the device-global
// non-shared __empty_completion path. This violates the arrival-count
// precondition and is undefined behavior.
// The test inspects the packed phase bit instead of waiting indefinitely.

#include <cuda/barrier>
#include <cuda/std/cstdint>
#include "test_macros.h"

__device__ hip::barrier<hip::thread_scope_block> g_over_arrive_barrier;

__device__ int test_over_arrive_corrupts_phase_nonshared()
{
  init(&g_over_arrive_barrier, 1); // expected=1, phase=0

  // Invalid over-arrival; observed corruption leaves the phase bit unchanged.
  auto tok = g_over_arrive_barrier.arrive(2);

  // Read the active native phase state. Device-global barriers use the
  // software fallback state, so the native phase bit is bit 63.
  constexpr hip::std::uint64_t phase_bit = 1ull << 63;
  auto* raw = reinterpret_cast<volatile hip::std::uint64_t*>(hip::device::barrier_native_handle(g_over_arrive_barrier));
  hip::std::uint64_t state = __atomic_load_n(raw, __ATOMIC_ACQUIRE);

  hip::std::uint64_t token_phase   = hip::std::uint64_t(tok) & phase_bit;
  hip::std::uint64_t current_phase = state & phase_bit;

  // If the phase bit still matches the token, wait() would not complete.
  return (current_phase == token_phase) ? 1 : 0;
}

__device__ int test()
{
  return test_over_arrive_corrupts_phase_nonshared();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (return test();))
  return 0;
}