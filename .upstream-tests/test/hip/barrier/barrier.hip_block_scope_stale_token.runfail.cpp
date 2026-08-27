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
// DOCUMENTATION TEST: stale tokens must not be reused across barrier phases.
// Tokens encode phase parity, not generation; runfail documents the violation.

#include <cuda/barrier>
#include <cuda/std/cstdint>
#include "test_macros.h"

__device__ int test_stale_token_bypasses_wait_on_phase_rollover()
{
  __shared__ hip::barrier<hip::thread_scope_block> b;
  init(&b, 1);

  auto tok0 = b.arrive();
  b.wait(hip::std::move(tok0));

  auto tok1 = b.arrive();
  b.wait(hip::std::move(tok1));

  auto tok2 = b.arrive();
  b.wait(hip::std::move(tok2));

  // Value 0 has the same phase bit as the token from the first phase.
  hip::std::uint64_t stale{0};
  b.wait(hip::std::move(stale));

  return 1; 
}

__device__ int test()
{
  return test_stale_token_bypasses_wait_on_phase_rollover();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (return test();))
  return 0;
}
