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

// UNSUPPORTED: c++98, c++03, msvc
// UNSUPPORTED: nvrtc
// XFAIL: gcc-4

// <utility>

// template <class T, class U> struct pair;

// pair(pair const&) = default;
// pair(pair &&) = default;
// pair& operator=(pair const&);
// pair& operator=(pair&&);

// Test that the copy/move constructors and assignment operators are
// correctly defined or deleted based on the properties of `T` and `U`.

#include <hip/std/cassert>
// hip::std::string not supported
// #include <hip/std/string>
#include <hip/std/tuple>

#include "archetypes.h"

#include "test_macros.h"
using namespace ImplicitTypes; // Get implicitly archetypes

namespace ConstructorTest {

template <class T1, bool CanCopy = true, bool CanMove = CanCopy>
__host__ __device__ void test() {
  using P1 = hip::std::pair<T1, int>;
  using P2 = hip::std::pair<int, T1>;
  static_assert(hip::std::is_copy_constructible<P1>::value == CanCopy, "");
  static_assert(hip::std::is_move_constructible<P1>::value == CanMove, "");
  static_assert(hip::std::is_copy_constructible<P2>::value == CanCopy, "");
  static_assert(hip::std::is_move_constructible<P2>::value == CanMove, "");
};

} // namespace ConstructorTest

__host__ __device__ void test_constructors_exist() {
  using namespace ConstructorTest;
  {
    test<int>();
    test<int &>();
    test<int &&, false, true>();
    test<const int>();
    test<const int &>();
    test<const int &&, false, true>();
  }
  {
    test<Copyable>();
    test<Copyable &>();
    test<Copyable &&, false, true>();
  }
  {
    test<NonCopyable, false>();
    test<NonCopyable &, true>();
    test<NonCopyable &&, false, true>();
  }
  {
    // Even though CopyOnly has an explicitly deleted move constructor
    // pair's move constructor is only implicitly deleted and therefore
    // it doesn't participate in overload resolution.
    test<CopyOnly, true, true>();
    test<CopyOnly &, true>();
    test<CopyOnly &&, false, true>();
  }
  {
    test<MoveOnly, false, true>();
    test<MoveOnly &, true>();
    test<MoveOnly &&, false, true>();
  }
}

namespace AssignmentOperatorTest {

template <class T1, bool CanCopy = true, bool CanMove = CanCopy>
__host__ __device__ void test() {
  using P1 = hip::std::pair<T1, int>;
  using P2 = hip::std::pair<int, T1>;
  static_assert(hip::std::is_copy_assignable<P1>::value == CanCopy, "");
  static_assert(hip::std::is_move_assignable<P1>::value == CanMove, "");
  static_assert(hip::std::is_copy_assignable<P2>::value == CanCopy, "");
  static_assert(hip::std::is_move_assignable<P2>::value == CanMove, "");
};

} // namespace AssignmentOperatorTest

__host__ __device__ void test_assignment_operator_exists() {
  using namespace AssignmentOperatorTest;
  {
    test<int>();
    test<int &>();
    test<int &&>();
    test<const int, false>();
    test<const int &, false>();
    test<const int &&, false>();
  }
  {
    test<Copyable>();
    test<Copyable &>();
    test<Copyable &&>();
  }
  {
    test<NonCopyable, false>();
    test<NonCopyable &, false>();
    test<NonCopyable &&, false>();
  }
  {
    test<CopyOnly, true>();
    test<CopyOnly &, true>();
    test<CopyOnly &&, true>();
  }
  {
    test<MoveOnly, false, true>();
    test<MoveOnly &, false, false>();
    test<MoveOnly &&, false, true>();
  }
}

int main(int, char**) {
  test_constructors_exist();
  test_assignment_operator_exists();

  return 0;
}
