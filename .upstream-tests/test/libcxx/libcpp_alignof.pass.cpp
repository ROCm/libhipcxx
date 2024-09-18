// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
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
// -*- C++ -*-

// Test that _LIBCUDACXX_ALIGNOF acts the same as the C++11 keyword `alignof`, and
// not as the GNU extension `__alignof`. The former returns the minimal required
// alignment for a type, whereas the latter returns the preferred alignment.
//
// See llvm.org/PR39713

#include <hip/std/type_traits>
#include "test_macros.h"

#pragma nv_diag_suppress cuda_demote_unsupported_floating_point

template <class T>
__host__ __device__
void test() {
  static_assert(_LIBCUDACXX_ALIGNOF(T) == hip::std::alignment_of<T>::value, "");
  static_assert(_LIBCUDACXX_ALIGNOF(T) == TEST_ALIGNOF(T), "");
#if TEST_STD_VER >= 11
  static_assert(_LIBCUDACXX_ALIGNOF(T) == alignof(T), "");
#endif
#ifdef TEST_COMPILER_CLANG
  static_assert(_LIBCUDACXX_ALIGNOF(T) == _Alignof(T), "");
#endif
}

int main(int, char**) {
  test<int>();
  test<long long>();
  test<double>();
  test<long double>();
  return 0;
}
