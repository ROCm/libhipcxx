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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   struct tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// XFAIL: gcc-4.8, gcc-4.9

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
// hip::std::array not supported
//#include <hip/std/array>
#include <hip/std/type_traits>

#include "test_macros.h"

template <class T, size_t Size = sizeof(hip::std::tuple_size<T>)>
__host__ __device__ constexpr bool is_complete(int) { static_assert(Size > 0, ""); return true; }
template <class>
__host__ __device__ constexpr bool is_complete(long) { return false; }
template <class T>
__host__ __device__ constexpr bool is_complete() { return is_complete<T>(0); }

struct Dummy1 {};
struct Dummy2 {};

namespace hip {
namespace std {
template <> struct tuple_size<Dummy1> : public integral_constant<size_t, 0> {};
}
}

template <class T>
__host__ __device__ void test_complete() {
  static_assert(is_complete<T>(), "");
  static_assert(is_complete<const T>(), "");
  static_assert(is_complete<volatile T>(), "");
  static_assert(is_complete<const volatile T>(), "");
}

template <class T>
__host__ __device__ void test_incomplete() {
  static_assert(!is_complete<T>(), "");
  static_assert(!is_complete<const T>(), "");
  static_assert(!is_complete<volatile T>(), "");
  static_assert(!is_complete<const volatile T>(), "");
}


int main(int, char**)
{
  test_complete<hip::std::tuple<> >();
  test_complete<hip::std::tuple<int&> >();
  test_complete<hip::std::tuple<int&&, int&, void*>>();
  test_complete<hip::std::pair<int, long> >();
  // hip::std::array not supported
  //test_complete<hip::std::array<int, 5> >();
  test_complete<Dummy1>();

  test_incomplete<void>();
  test_incomplete<int>();
  test_incomplete<hip::std::tuple<int>&>();
  test_incomplete<Dummy2>();

  return 0;
}
