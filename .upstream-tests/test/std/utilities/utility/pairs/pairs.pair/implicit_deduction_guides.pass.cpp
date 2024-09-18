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

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides
// UNSUPPORTED: msvc
// UNSUPPORTED: nvrtc
// UNSUPPORTED: nvcc-10.3, nvcc-11.0, nvcc-11.1, nvcc-11.2, nvcc-11.3, nvcc-11.4

// GCC's implementation of class template deduction is still immature and runs
// into issues with libc++. However GCC accepts this code when compiling
// against libstdc++.
// XFAIL: gcc-4.8, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9, gcc-10, gcc-11

// Currently broken with Clang + NVCC.
// XFAIL: clang-6, clang-7

// <utility>

// Test that the constructors offered by hip::std::pair are formulated
// so they're compatible with implicit deduction guides, or if that's not
// possible that they provide explicit guides to make it work.

#include <hip/std/utility>
// cuda/std/memory not supported
// #include <hip/std/memory>
// hip::std::string not supported
// #include <hip/std/string>
#include <hip/std/cassert>

#include "test_macros.h"
#include "archetypes.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

// Overloads
// ---------------
// (1)  pair(const T1&, const T2&) -> pair<T1, T2>
// (2)  explicit pair(const T1&, const T2&) -> pair<T1, T2>
// (3)  pair(pair const& t) -> decltype(t)
// (4)  pair(pair&& t) -> decltype(t)
// (5)  pair(pair<U1, U2> const&) -> pair<U1, U2>
// (6)  explicit pair(pair<U1, U2> const&) -> pair<U1, U2>
// (7)  pair(pair<U1, U2> &&) -> pair<U1, U2>
// (8)  explicit pair(pair<U1, U2> &&) -> pair<U1, U2>
int main(int, char**)
{
  using E = ExplicitTestTypes::TestType;
  static_assert(!hip::std::is_convertible<E const&, E>::value, "");
  { // Testing (1)
    int const x = 42;
    hip::std::pair t1("abc", x);
    ASSERT_SAME_TYPE(decltype(t1), hip::std::pair<const char*, int>);
    unused(t1);
  }
  { // Testing (2)
    hip::std::pair p1(E{}, 42);
    ASSERT_SAME_TYPE(decltype(p1), hip::std::pair<E, int>);
    unused(p1);

    const E t{};
    hip::std::pair p2(t, E{});
    ASSERT_SAME_TYPE(decltype(p2), hip::std::pair<E, E>);
  }
  { // Testing (3, 5)
    hip::std::pair<double, decltype(nullptr)> const p(0.0, nullptr);
    hip::std::pair p1(p);
    unused(p1);
    ASSERT_SAME_TYPE(decltype(p1), hip::std::pair<double, decltype(nullptr)>);
  }
  { // Testing (3, 6)
    hip::std::pair<E, decltype(nullptr)> const p(E{}, nullptr);
    hip::std::pair p1(p);
    unused(p1);
    ASSERT_SAME_TYPE(decltype(p1), hip::std::pair<E, decltype(nullptr)>);
  }
  // hip::std::string not supported
  /*
  { // Testing (4, 7)
    hip::std::pair<hip::std::string, void*> p("abc", nullptr);
    hip::std::pair p1(hip::std::move(p));
    ASSERT_SAME_TYPE(decltype(p1), hip::std::pair<hip::std::string, void*>);
  }
  { // Testing (4, 8)
    hip::std::pair<hip::std::string, E> p("abc", E{});
    hip::std::pair p1(hip::std::move(p));
    ASSERT_SAME_TYPE(decltype(p1), hip::std::pair<hip::std::string, E>);
  }
  */
  return 0;
}
