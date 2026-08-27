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
// A host-only system-scope barrier with a __host__ completion function.
// The host is the sole participant and is guaranteed to be the last to arrive,
// so calling a __host__ completion function is safe.

#include <cstdlib>
#include <hip/barrier>
#include <hip/hip_runtime.h>

using barrier_t = hip::barrier<hip::thread_scope_system, void(*)()>;

static barrier_t g_bar;
static int        g_completion_count   = 0;

__host__ void do_completion() { g_completion_count++; }

__host__ int hostSideWork()
{
  g_completion_count = 0;
  init(&g_bar, 1, do_completion);

  g_bar.arrive_and_wait();
  if (g_completion_count != 1) return 1;

  g_bar.arrive_and_wait();
  if (g_completion_count != 2) return 2;

  return 0;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    cuda_block_count  = 1;
    cuda_thread_count = 1;

    hostSideWorkFunc = hostSideWork;
    return 0;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (return 0;))
  return 0;
}
