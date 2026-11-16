//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef TEST_CMATH_COMPARISON_H
#define TEST_CMATH_COMPARISON_H

#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename T>
TEST_FUNC constexpr bool eq(T lhs, T rhs) noexcept
{
  return lhs == rhs;
}

template <typename T, typename U, cuda::std::enable_if_t<cuda::std::is_arithmetic_v<U>, int> = 0>
TEST_FUNC constexpr bool eq(T lhs, U rhs) noexcept
{
  return eq(lhs, T(rhs));
}

#if _CCCL_HAS_NVFP16()
TEST_FUNC bool eq(__half lhs, __half rhs) noexcept
{
#  if _CCCL_CTK_AT_LEAST(12, 2)
  return ::__heq(lhs, rhs);
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv !_CCCL_CTK_AT_LEAST(12, 2) vvv
  return ::__half2float(lhs) == ::__half2float(rhs);
#  endif // !_CCCL_CTK_AT_LEAST(12, 2)
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
TEST_FUNC bool eq(__hip_bfloat16 lhs, __hip_bfloat16 rhs) noexcept
{
#  if _CCCL_CTK_AT_LEAST(12, 2)
  return ::__heq(lhs, rhs);
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv !_CCCL_CTK_AT_LEAST(12, 2) vvv
  return ::__bfloat162float(lhs) == ::__bfloat162float(rhs);
#  endif // !_CCCL_CTK_AT_LEAST(12, 2)
}
#endif // _CCCL_HAS_NVBF16()

template <class Integer, cuda::std::enable_if_t<cuda::std::is_integral_v<Integer>, int> = 0>
TEST_FUNC bool is_about(Integer x, Integer y)
{
  return true;
}

TEST_FUNC bool is_about(float x, float y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-6);
}

TEST_FUNC bool is_about(double x, double y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-14);
}

#if _CCCL_HAS_LONG_DOUBLE()
TEST_FUNC bool is_about(long double x, long double y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-14);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
TEST_FUNC bool is_about(__half x, __half y)
{
  return (cuda::std::fabs((x - y) / (x + y)) <= __half(1e-3));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
TEST_FUNC bool is_about(__hip_bfloat16 x, __hip_bfloat16 y)
{
  return (cuda::std::fabs((x - y) / (x + y)) <= __hip_bfloat16(5e-3));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#endif // TEST_CMATH_COMPARISON_H
