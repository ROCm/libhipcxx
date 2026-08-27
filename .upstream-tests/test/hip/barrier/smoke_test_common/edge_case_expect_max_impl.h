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

// Shared edge-case smoke test: barrier initialised with max() and satisfied
// by a single arrive(max()) call. Templated on barrier type.

#ifndef LIBHIPCXX_BARRIER_EDGE_CASE_EXPECT_MAX_IMPL_H
#define LIBHIPCXX_BARRIER_EDGE_CASE_EXPECT_MAX_IMPL_H

#include "../hip_barrier_test_utils.h"

namespace hip_test {

template <hip::thread_scope Scope, class CompletionF = hip::std::__empty_completion>
__host__ __device__ int test_edge_case_expect_max(hip::barrier<Scope, CompletionF>& bar, CompletionF fn = {})
{
  init_barrier(&bar, hip::barrier<Scope, CompletionF>::max(), fn);
  bar.wait(bar.arrive(hip::barrier<Scope, CompletionF>::max()));
  return 0;
}

} // namespace hip_test

#endif // LIBHIPCXX_BARRIER_EDGE_CASE_EXPECT_MAX_IMPL_H
