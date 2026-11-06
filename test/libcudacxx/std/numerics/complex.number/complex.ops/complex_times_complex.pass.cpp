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

// template<class T>
//   complex<T>
//   operator*(const complex<T>& lhs, const complex<T>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
TEST_FUNC constexpr bool test()
{
  cuda::std::complex<T> lhs(1.5, 2.5);
  cuda::std::complex<T> rhs(1.5, 2.5);
  cuda::std::complex<T> x(-4.0, 7.5);
  assert(lhs * rhs == x);

  return true;
}

// test edges

template <class T>
TEST_FUNC void test_edges()
{
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    for (unsigned j = 0; j < N; ++j)
    {
      cuda::std::complex<T> r = testcases[i] * testcases[j];
      switch (classify(testcases[i]))
      {
        case zero:
          switch (classify(testcases[j]))
          {
            case zero:
              assert(classify(r) == zero);
              break;
            case non_zero:
              assert(classify(r) == zero);
              break;
            case inf:
              assert(classify(r) == NaN);
              break;
            case NaN:
              assert(classify(r) == NaN);
              break;
            case non_zero_nan:
              assert(classify(r) == NaN);
              break;
          }
          break;
        case non_zero:
          switch (classify(testcases[j]))
          {
            case zero:
              assert(classify(r) == zero);
              break;
            case non_zero:
              assert(classify(r) == non_zero);
              break;
            case inf:
              assert(classify(r) == inf);
              break;
            case NaN:
              assert(classify(r) == NaN);
              break;
            case non_zero_nan:
              assert(classify(r) == NaN);
              break;
          }
          break;
        case inf:
          switch (classify(testcases[j]))
          {
            case zero:
              assert(classify(r) == NaN);
              break;
            case non_zero:
              assert(classify(r) == inf);
              break;
            case inf:
              assert(classify(r) == inf);
              break;
            case NaN:
              assert(classify(r) == NaN);
              break;
            case non_zero_nan:
              assert(classify(r) == inf);
              break;
          }
          break;
        case NaN:
          switch (classify(testcases[j]))
          {
            case zero:
              assert(classify(r) == NaN);
              break;
            case non_zero:
              assert(classify(r) == NaN);
              break;
            case inf:
              assert(classify(r) == NaN);
              break;
            case NaN:
              assert(classify(r) == NaN);
              break;
            case non_zero_nan:
              assert(classify(r) == NaN);
              break;
          }
          break;
        case non_zero_nan:
          switch (classify(testcases[j]))
          {
            case zero:
              assert(classify(r) == NaN);
              break;
            case non_zero:
              assert(classify(r) == NaN);
              break;
            case inf:
              assert(classify(r) == inf);
              break;
            case NaN:
              assert(classify(r) == NaN);
              break;
            case non_zero_nan:
              assert(classify(r) == NaN);
              break;
          }
          break;
      }
    }
  }
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS()
#  if !TEST_COMPILER(GCC, <, 8) // GCC 7 does not support constexpr is_nan and friends
  static_assert(test<float>());
  static_assert(test<double>());
#    if _CCCL_HAS_LONG_DOUBLE()
  static_assert(test<long double>());
#    endif // _CCCL_HAS_LONG_DOUBLE()
#  endif // !TEST_COMPILER(GCC, <, 8)
#endif // _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__hip_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  test_edges<double>();
#if _LIBCUDACXX_HAS_NVFP16()
  test_edges<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_edges<__hip_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return 0;
}
