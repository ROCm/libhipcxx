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

// <cuda/std/array>

// template <class T, size_t N >
// struct array

// Test the size and alignment matches that of an array of a given type.

#include <hip/std/array>
#include <hip/std/iterator>
#include <hip/std/type_traits>
#include <hip/std/cstddef>

#include "test_macros.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4324)
#endif

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

template <class T, size_t Size>
struct MyArray {
  T elems[Size];
};

template <class T, size_t Size>
__host__ __device__
void test() {
  typedef T CArrayT[Size == 0 ? 1 : Size];
  typedef hip::std::array<T, Size> ArrayT;
  typedef MyArray<T, Size == 0 ? 1 : Size> MyArrayT;
  static_assert(sizeof(ArrayT) == sizeof(CArrayT), "");
  static_assert(sizeof(ArrayT) == sizeof(MyArrayT), "");
  static_assert(TEST_ALIGNOF(ArrayT) == TEST_ALIGNOF(MyArrayT), "");
#if defined(_LIBCUDACXX_VERSION)
  ArrayT a{};
  unused(a);
  static_assert(sizeof(ArrayT) == sizeof(a.__elems_), "");
  #if !defined(_LIBCUDACXX_COMPILER_MSVC)
  static_assert(TEST_ALIGNOF(ArrayT) == __alignof__(a.__elems_), "");
  #endif
#endif
}

template <class T>
__host__ __device__
void test_type() {
  test<T, 1>();
  test<T, 42>();
  test<T, 0>();
}

struct TEST_ALIGNAS(TEST_ALIGNOF(hip::std::max_align_t) * 2) TestType1 {

};

struct TEST_ALIGNAS(TEST_ALIGNOF(hip::std::max_align_t) * 2) TestType2 {
  char data[1000];
};

//static_assert(sizeof(void*) == 4, "");

int main(int, char**) {
  test_type<char>();
  test_type<int>();
  test_type<double>();
#if !defined(__CUDA_ARCH__)
  test_type<long double>();
#endif
  test_type<hip::std::max_align_t>();
  test_type<TestType1>();
  test_type<TestType2>();

  return 0;
}
