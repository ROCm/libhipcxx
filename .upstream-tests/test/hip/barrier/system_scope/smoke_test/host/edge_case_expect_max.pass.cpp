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
// Verify barrier<thread_scope_system> can be initialised with max() and that
// arrive(max()) satisfies it in one call, from the host. The host is the sole
// participant.

#include <hip/barrier>
#include "test_macros.h"
#include "../../../smoke_test_common/edge_case_expect_max_impl.h"

using namespace hip_test;

using barrier_t = hip::barrier<hip::thread_scope_system>;

static barrier_t g_bar;

__host__ int hostSideWork()
{
  return test_edge_case_expect_max(g_bar);
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
