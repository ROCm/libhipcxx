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

// <hip/barrier>
// Verify construction via init(), placement-new, and expected=0 for
// barrier<thread_scope_system> from the host. The host is the sole participant.

#include <hip/barrier>
#include "../../../smoke_test_common/construct_impl.h"

using namespace hip_test;

constexpr hip::thread_scope Scope = hip::thread_scope_system;
using barrier_t = hip::barrier<Scope>;

static barrier_t g_bar;
static barrier_t g_zero;
alignas(barrier_t) static hip::std::byte g_raw[sizeof(barrier_t)];

__host__ int hostSideWork()
{
  test_default_init_and_init(g_bar);
  test_placement_new<Scope>(g_raw);
  test_init_expected_zero(g_zero);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    cuda_block_count  = 1;
    cuda_thread_count = 1;
    hostSideWorkFunc  = hostSideWork;
    return 0;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (return 0;))
}
