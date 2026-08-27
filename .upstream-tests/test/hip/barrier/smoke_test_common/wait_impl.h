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

// Shared wait() smoke test logic, templated on barrier type.
// The caller provides storage and passes the barrier by reference.

#ifndef LIBHIPCXX_BARRIER_WAIT_IMPL_H
#define LIBHIPCXX_BARRIER_WAIT_IMPL_H

#include <cuda/std/utility>
#include "../hip_barrier_test_utils.h"

namespace hip_test {

// wait() unblocks after a single arrive() on a barrier with expected=1.
template <hip::thread_scope Scope, class CompletionF = hip::std::__empty_completion>
__host__ __device__ void test_wait_unblocks_after_single_arrive(hip::barrier<Scope, CompletionF>& bar, CompletionF fn = {})
{
  init_barrier(&bar, 1, fn);
  auto tok = bar.arrive();
  bar.wait(hip::std::move(tok));
}

// wait() unblocks when arrive(n) fully satisfies a barrier with expected=n.
template <hip::thread_scope Scope, class CompletionF = hip::std::__empty_completion>
__host__ __device__ void test_wait_unblocks_after_arrive_update(hip::barrier<Scope, CompletionF>& bar, CompletionF fn = {})
{
  init_barrier(&bar, 2, fn);
  auto tok = bar.arrive(2);
  bar.wait(hip::std::move(tok));
}

} // namespace hip_test

#endif // LIBHIPCXX_BARRIER_WAIT_IMPL_H
