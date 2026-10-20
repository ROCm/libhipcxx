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

// Shared phase-reuse smoke test logic, templated on barrier type.
// Each function takes a dedicated barrier by reference — callers must provide
// one barrier per function (re-init writes new state into the same object).

#ifndef LIBHIPCXX_BARRIER_PHASE_REUSE_IMPL_H
#define LIBHIPCXX_BARRIER_PHASE_REUSE_IMPL_H

#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include "../hip_barrier_test_utils.h"

namespace hip_test {

// Phase alternates 0 -> 1 -> 0 across three arrive()+wait() cycles.
template <hip::thread_scope Scope, class CompletionF = hip::std::__empty_completion>
__host__ __device__ int test_phase_alternates_across_cycles(hip::barrier<Scope, CompletionF>& bar, CompletionF fn = {})
{
  init_barrier(&bar, 1, fn);

  auto tok0 = bar.arrive();
  if (tok0 != 0) return 1;
  bar.wait(hip::std::move(tok0));

  auto tok1 = bar.arrive();
  if (tok1 == 0) return 1;
  bar.wait(hip::std::move(tok1));

  auto tok2 = bar.arrive();
  if (tok2 != 0) return 1;
  bar.wait(hip::std::move(tok2));

  return 0;
}

// Phase must not flip until exactly N arrivals have occurred.
template <hip::thread_scope Scope, class CompletionF = hip::std::__empty_completion>
__host__ __device__ int test_phase_flip_requires_full_expected_count(hip::barrier<Scope, CompletionF>& bar, CompletionF fn = {})
{
  init_barrier(&bar, 3, fn);

  auto tok_a = bar.arrive();
  if (tok_a != 0) return 1;
  auto tok_b = bar.arrive();
  if (tok_b != 0) return 1;
  auto tok_c = bar.arrive(); // 3rd of 3: triggers flip
  if (tok_c != 0) return 1;  // sampled before the flip
  bar.wait(hip::std::move(tok_c));

  auto tok_d = bar.arrive();
  if (tok_d == 0) return 1;
  auto tok_e = bar.arrive();
  if (tok_e == 0) return 1;
  auto tok_f = bar.arrive(); // 3rd of 3: triggers flip back
  if (tok_f == 0) return 1;
  bar.wait(hip::std::move(tok_f));

  auto tok_g = bar.arrive();
  if (tok_g != 0) return 1;

  return 0;
}

// arrive(n) counts as n arrivals; phase must not flip until the total reaches
// expected.
template <hip::thread_scope Scope, class CompletionF = hip::std::__empty_completion>
__host__ __device__ int test_bulk_arrive_counts_correctly(hip::barrier<Scope, CompletionF>& bar, CompletionF fn = {})
{
  init_barrier(&bar, 4, fn);

  auto tok_bulk = bar.arrive(2); // 2 of 4; no flip
  if (tok_bulk != 0) return 1;
  auto tok_1 = bar.arrive();     // 3 of 4; no flip
  if (tok_1 != 0) return 1;
  auto tok_2 = bar.arrive();     // 4 of 4; triggers flip
  if (tok_2 != 0) return 1;      // sampled before the flip
  bar.wait(hip::std::move(tok_2));

  auto tok_next = bar.arrive();
  if (tok_next == 0) return 1;

  return 0;
}

// All arrive() calls within the same phase return the same token value.
template <hip::thread_scope Scope, class CompletionF = hip::std::__empty_completion>
__host__ __device__ int test_token_encodes_only_phase(hip::barrier<Scope, CompletionF>& bar, CompletionF fn = {})
{
  init_barrier(&bar, 3, fn);

  auto tok0 = bar.arrive();
  auto tok1 = bar.arrive();
  auto tok2 = bar.arrive(); // 3rd of 3: triggers flip
  if (tok0 != 0) return 1;
  if (tok1 != 0) return 1;
  if (tok2 != 0) return 1;
  bar.wait(hip::std::move(tok2));

  auto tok3 = bar.arrive();
  auto tok4 = bar.arrive();
  auto tok5 = bar.arrive(); // 3rd of 3: triggers flip back
  if (tok3 == 0) return 1;
  if (tok4 == 0) return 1;
  if (tok5 == 0) return 1;
  bar.wait(hip::std::move(tok5));

  return 0;
}

} // namespace hip_test

#endif // LIBHIPCXX_BARRIER_PHASE_REUSE_IMPL_H
