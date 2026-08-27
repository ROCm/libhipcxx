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
// Verify phase and token semantics across multiple arrive()+wait() cycles for
// barrier<thread_scope_system>.

#include <hip/barrier>
#include "test_macros.h"
#include "../../../smoke_test_common/phase_reuse_impl.h"

using namespace hip_test;

using barrier_t = hip::barrier<hip::thread_scope_system>;
__device__ barrier_t g_a, g_b, g_c, g_d;

__device__ int test()
{
  int result = 0;
  result |= test_phase_alternates_across_cycles(g_a);
  result |= test_phase_flip_requires_full_expected_count(g_b);
  result |= test_bulk_arrive_counts_correctly(g_c);
  result |= test_token_encodes_only_phase(g_d);
  return result;
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_HOST, (
    cuda_block_count  = 1;
    cuda_thread_count = 1;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
