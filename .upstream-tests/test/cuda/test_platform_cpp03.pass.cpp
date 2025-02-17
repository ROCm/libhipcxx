//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
// UNSUPPORTED: hipcc

// UNSUPPORTED: nvhpc, nvc++

#include <nv/target>

#if !defined(__CUDACC_RTC__)
#include <assert.h>
#include <stdio.h>
#endif

#ifdef __CUDACC__
# define HD_ANNO __host__ __device__
#else
# define HD_ANNO
#endif

template <typename T>
HD_ANNO bool unused(T) {return true;}

// Assert macro interferes with preprocessing, wrap it in a function
HD_ANNO inline void check_v(bool result) {
  assert(result);
}

HD_ANNO void test() {
#  if defined(__CUDA_ARCH__)
  int arch_val = __CUDA_ARCH__;
#  else
  int arch_val = 0;
#  endif

  unused(arch_val);

  NV_IF_TARGET(
    NV_IS_HOST,
      check_v(arch_val == 0);
  )

  NV_IF_TARGET(
    NV_IS_DEVICE,
      check_v(arch_val == __CUDA_ARCH__);
  )

  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
      check_v(arch_val == 0);,
      check_v(arch_val == __CUDA_ARCH__);
  )
}

int main(int argc, char ** argv)
{
    test();
    return 0;
}
