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
//
// Verify arrive(n) completes a phase, orders prior writes, and returns the
// expected phase tokens across repeated cycles.

#include <cuda/barrier>
#include <cuda/std/cstdint>
#include "test_macros.h"

static constexpr hip::std::uint64_t k_phase_bit = 1ull << 63;
static constexpr int                k_n          = 16;

// Single-thread bulk arrive covers the full expected count.
__device__ int test_bulk_arrive_ordering()
{
  __shared__ hip::barrier<hip::thread_scope_block> b;
  __shared__ int payload[k_n];
  init(&b, k_n);

  for (int i = 0; i < k_n; ++i)
    payload[i] = i + 1;

  // Single arrive(k_n) satisfies the full expected count.
  auto tok = b.arrive(k_n);
  if (tok != 0) return 1; // phase 0 token sampled before the flip
  b.wait(hip::std::move(tok));

  // Stores before arrive(k_n) must be visible after wait().
  for (int i = 0; i < k_n; ++i)
  {
    if (payload[i] != i + 1)
    {
      return 2;
    }
  }

  return 0;
}

// arrive(n) phase tokens alternate across successive cycles.
__device__ int test_bulk_arrive_phase_tokens()
{
  __shared__ hip::barrier<hip::thread_scope_block> b;
  init(&b, 4);

  auto tok0 = b.arrive(4);
  if (tok0 != 0) return 1;
  b.wait(hip::std::move(tok0));

  auto tok1 = b.arrive(4);
  if (tok1 != k_phase_bit) return 2;
  b.wait(hip::std::move(tok1));

  auto tok2 = b.arrive(4);
  if (tok2 != 0) return 3;
  b.wait(hip::std::move(tok2));

  return 0;
}

__device__ int test()
{
  int result = 0;
  result = test_bulk_arrive_ordering();
  if (result != 0) return result;

  result = test_bulk_arrive_phase_tokens();
  return result;
}

int main(int, char**)
{
  int result = 0;
  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
