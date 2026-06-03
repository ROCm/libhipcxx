//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// FORCE_ALL_WARNINGS.

// NOTE(HIP/AMD): this .compile.fail test relies on the clang::lifetimebound
// -Wdangling diagnostic being escalated to a hard error (via -Werror) so that
// compilation fails. HIPRTC's COMGR runtime compiler emits -Wdangling but does
// NOT honor -Werror to turn it into an error, so compilation succeeds and the
// expected failure cannot be observed. This is a HIPRTC/COMGR driver limitation
// (NVRTC makes the diagnostic a hard error by default), not a libhipcxx gap.
// UNSUPPORTED: hiprtc

#include <cuda/std/__cccl/attributes.h>

#include "test_macros.h"

#if _CCCL_HAS_CPP_ATTRIBUTE(clang::lifetimebound) || _CCCL_COMPILER(CLANG)
#elif _CCCL_HAS_CPP_ATTRIBUTE(msvc::lifetimebound) || _CCCL_COMPILER(MSVC, >=, 19, 37)
#else
#  error "lifetimebound attribute not supported"
#endif

struct S
{
  char data[32];

  __host__ __device__ const char* get() const _CCCL_LIFETIMEBOUND
  {
    return data;
  }
};

__host__ __device__ bool test()
{
  auto sv = S{"abc"}.get();
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
