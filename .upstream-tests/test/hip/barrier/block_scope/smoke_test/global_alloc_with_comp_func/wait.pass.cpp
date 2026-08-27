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

// <cuda/barrier>
// Verify wait() for non-shared barriers with completion function.

#include <cuda/barrier>
#include "test_macros.h"
#include "../../../hip_barrier_test_utils.h"
#include "../../../smoke_test_common/wait_impl.h"

using namespace hip_test;

using barrier_t = hip::barrier<hip::thread_scope_block, void (*)()>;

__device__ alignas(barrier_t) hip::std::byte g_raw_a[sizeof(barrier_t)];
__device__ alignas(barrier_t) hip::std::byte g_raw_b[sizeof(barrier_t)];

__device__ void test()
{
  auto* ba = reinterpret_cast<barrier_t*>(g_raw_a);
  auto* bb = reinterpret_cast<barrier_t*>(g_raw_b);
  test_wait_unblocks_after_single_arrive(*ba, barrier_no_op_completion);
  test_wait_unblocks_after_arrive_update(*bb, barrier_no_op_completion);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
  return 0;
}
