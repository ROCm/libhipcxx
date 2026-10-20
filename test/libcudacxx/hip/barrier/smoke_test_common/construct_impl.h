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

// Shared construction smoke test logic, templated on barrier type.
// The caller provides storage and passes barriers by reference.

#ifndef LIBHIPCXX_BARRIER_CONSTRUCT_IMPL_H
#define LIBHIPCXX_BARRIER_CONSTRUCT_IMPL_H

#include <cuda/barrier>
#include <cuda/std/cstddef>
#include "../hip_barrier_test_utils.h"

namespace hip_test {

// construct_barrier: placement-new with completion for real types, without for __empty_completion.
template <hip::thread_scope Scope, class CompletionF>
__host__ __device__ hip::barrier<Scope, CompletionF>* construct_barrier(hip::std::byte* raw, hip::std::ptrdiff_t n, CompletionF fn)
{
  return new (raw) hip::barrier<Scope, CompletionF>(n, fn);
}
template <hip::thread_scope Scope>
__host__ __device__ hip::barrier<Scope>* construct_barrier(hip::std::byte* raw, hip::std::ptrdiff_t n, hip::std::__empty_completion)
{
  return new (raw) hip::barrier<Scope>(n);
}

// Canonical path: default-constructed storage followed by init().
template <hip::thread_scope Scope, class CompletionF = hip::std::__empty_completion>
__host__ __device__ void test_default_init_and_init(hip::barrier<Scope, CompletionF>& bar, CompletionF fn = {})
{
  init_barrier(&bar, 1, fn);
}

// Value construction via placement-new into caller-provided raw storage.
template <hip::thread_scope Scope, class CompletionF = hip::std::__empty_completion>
__host__ __device__ int test_placement_new(hip::std::byte* raw, CompletionF fn = {})
{
  using barrier_t = hip::barrier<Scope, CompletionF>;
  auto* bp = construct_barrier<Scope, CompletionF>(raw, 1, fn);
  bp->~barrier_t();
  return 0;
}

// expected=0 is a supported degenerate initialisation case.
template <hip::thread_scope Scope, class CompletionF = hip::std::__empty_completion>
__host__ __device__ void test_init_expected_zero(hip::barrier<Scope, CompletionF>& bar, CompletionF fn = {})
{
  init_barrier(&bar, 0, fn);
}

} // namespace hip_test

#endif // LIBHIPCXX_BARRIER_CONSTRUCT_IMPL_H
