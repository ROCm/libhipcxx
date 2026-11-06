//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

// XFAIL: enable-tile
// nvbug6077402: error: "call to non-tile function not supported!"

// <cuda/std/complex>

// template<Arithmetic T>
//   T
//   real(const T& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "../cases.h"
#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4244) // conversion from 'const double' to 'int', possible loss of data

template <class T, int x, class Target>
TEST_FUNC void test_nonconstexpr()
{
  static_assert((cuda::std::is_same<decltype(cuda::std::real(T(x))), Target>::value));
  assert(cuda::std::real(T(x)) == T(x));
}

template <class T, int x>
TEST_FUNC void test(typename cuda::std::enable_if<cuda::std::is_integral<T>::value>::type* = 0)
{
  test_nonconstexpr<T, x, double>();

  static_assert((cuda::std::is_same<decltype(cuda::std::real(T(x))), double>::value));
  assert(cuda::std::real(x) == x);
  constexpr T val{x};
  static_assert(cuda::std::real(val) == val);
  constexpr cuda::std::complex<T> t{val, val};
  static_assert(t.real() == x);
}

template <class T, int x>
TEST_FUNC void test(typename cuda::std::enable_if<!cuda::std::is_integral<T>::value>::type* = 0)
{
  test_nonconstexpr<T, x, T>();

  static_assert((cuda::std::is_same<decltype(cuda::std::real(T(x))), T>::value));
  assert(cuda::std::real(x) == x);
  constexpr T val{x};
  static_assert(cuda::std::real(val) == val);
  constexpr cuda::std::complex<T> t{val, val};
  static_assert(t.real() == x);
}

template <class T>
TEST_FUNC void test_nonconstexpr()
{
  test_nonconstexpr<T, 0, T>();
  test_nonconstexpr<T, 1, T>();
  test_nonconstexpr<T, 10, T>();
}

template <class T>
TEST_FUNC void test()
{
  test<T, 0>();
  test<T, 1>();
  test<T, 10>();
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test_nonconstexpr<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_nonconstexpr<__hip_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  test<int>();
  test<unsigned>();
  test<long long>();

  return 0;
}
