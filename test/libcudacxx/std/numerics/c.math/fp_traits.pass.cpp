//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

// <cmath>

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "fp_compare.h"
#include "test_macros.h"

TEST_FUNC void test_isgreater(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__hip_bfloat16) 0, (__hip_bfloat16) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__hip_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreater((__hip_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::isgreater(-1.0, 0.F) == false);
}

TEST_FUNC void test_isgreaterequal(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((long double) 0, (long double) 0)), bool>),
                "");
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__hip_bfloat16) 0, (__hip_bfloat16) 0)), bool>),
                "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__hip_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isgreaterequal((__hip_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::isgreaterequal(-1.0, 0.F) == false);
}

TEST_FUNC void test_isless(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__hip_bfloat16) 0, (__hip_bfloat16) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__hip_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isless((__hip_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::isless(-1.0, 0.F) == true);
}

TEST_FUNC void test_islessequal(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__hip_bfloat16) 0, (__hip_bfloat16) 0)), bool>),
                "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__hip_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessequal((__hip_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::islessequal(-1.0, 0.F) == true);
}

TEST_FUNC void test_islessgreater(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__hip_bfloat16) 0, (__hip_bfloat16) 0)), bool>),
                "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__hip_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::islessgreater((__hip_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::islessgreater(-1.0, 0.F) == true);
}

TEST_FUNC void test_isunordered(float val)
{
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((float) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((float) 0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((float) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered(0, (double) 0)), bool>) );
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((double) 0, (long double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((long double) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((long double) 0, (double) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((long double) 0, (long double) 0)), bool>) );
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__half) 0, (__half) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__half) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__half) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__hip_bfloat16) 0, (__hip_bfloat16) 0)), bool>),
                "");
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__hip_bfloat16) 0, (float) 0)), bool>) );
  static_assert((cuda::std::is_same_v<decltype(cuda::std::isunordered((__hip_bfloat16) 0, (double) 0)), bool>) );
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::isunordered(-1.0, 0.F) == false);
}

TEST_FUNC void test(float val)
{
  test_isgreater(val);
  test_isgreaterequal(val);
  test_isless(val);
  test_islessequal(val);
  test_islessgreater(val);
  test_isunordered(val);
}

__global__ void test_global_kernel(float* val)
{
  test(*val);
}

int main(int, char**)
{
  volatile float val = 1.0f;
  test(val);

  return 0;
}
