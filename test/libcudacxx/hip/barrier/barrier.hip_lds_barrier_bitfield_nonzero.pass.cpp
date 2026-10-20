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
// Verify non-zero __lds_barrier_t bitfields and setter round-trips.

#include <cuda/barrier>
#include "test_macros.h"

__device__ int test_nonzero_bitfield_reading()
{
  hip::__lds_barrier_t bar_manual;

  // Packed value: init=1, phase=3, pending=100.
  bar_manual.value = 0x0000000160000064ULL;

  if (bar_manual.pending_count != 100) return 1;
  if (bar_manual.phase != 3) return 2;
  if (bar_manual.init_count != 1) return 3;

  hip::__lds_barrier_t bar_api;
  bar_api.value = 0;

  bar_api.set_pending_count(100);
  bar_api.phase = 3;
  bar_api.set_init_count(1);

  if (bar_api.pending_count != 100) return 4;
  if (bar_api.phase != 3) return 5;
  if (bar_api.init_count != 1) return 6;

  if (bar_api.value != bar_manual.value) return 7;

  return 0;
}

__device__ int test()
{
  return test_nonzero_bitfield_reading();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    cuda_block_count = 1;
    cuda_thread_count = 1;
  ))

  int result = 0;
  NV_IF_TARGET(NV_IS_DEVICE, (result = test();))
  return result;
}
