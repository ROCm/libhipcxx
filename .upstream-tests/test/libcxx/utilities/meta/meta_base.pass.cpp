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
//

#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

struct Bomb;
template <int N, class T = Bomb >
struct BOOM {
  using Explode = typename T::BOOMBOOM;
};

using True = hip::std::true_type;
using False = hip::std::false_type;

__host__ __device__
void test_if() {
  ASSERT_SAME_TYPE(hip::std::_If<true, int, long>, int);
  ASSERT_SAME_TYPE(hip::std::_If<false, int, long>, long);
}

__host__ __device__
void test_and() {
  static_assert(hip::std::_And<True>::value, "");
  static_assert(!hip::std::_And<False>::value, "");
  static_assert(hip::std::_And<True, True>::value, "");
  static_assert(!hip::std::_And<False, BOOM<1> >::value, "");
  static_assert(!hip::std::_And<True, True, True, False, BOOM<2> >::value, "");
}

__host__ __device__
void test_or() {
  static_assert(hip::std::_Or<True>::value, "");
  static_assert(!hip::std::_Or<False>::value, "");
  static_assert(hip::std::_Or<False, True>::value, "");
  static_assert(hip::std::_Or<True, hip::std::_Not<BOOM<3> > >::value, "");
  static_assert(!hip::std::_Or<False, False>::value, "");
  static_assert(hip::std::_Or<True, BOOM<1> >::value, "");
  static_assert(hip::std::_Or<False, False, False, False, True, BOOM<2> >::value, "");
}

__host__ __device__
void test_combined() {
  static_assert(hip::std::_And<True, hip::std::_Or<False, True, BOOM<4> > >::value, "");
  static_assert(hip::std::_And<True, hip::std::_Or<False, True, BOOM<4> > >::value, "");
  static_assert(hip::std::_Not<hip::std::_And<True, False, BOOM<5> > >::value, "");
}

struct MemberTest {
  static const int foo;
  using type = long;

  __host__ __device__
  void func(int);
};
struct Empty {};
struct MemberTest2 {
  using foo = int;
};
template <class T>
using HasFooData = decltype(T::foo);
template <class T>
using HasFooType = typename T::foo;

template <class T, class U>
using FuncCallable = decltype(hip::std::declval<T>().func(hip::std::declval<U>()));
template <class T>
using BadCheck = typename T::DOES_NOT_EXIST;

__host__ __device__
void test_is_valid_trait() {
  static_assert(hip::std::_IsValidExpansion<HasFooData, MemberTest>::value, "");
  static_assert(!hip::std::_IsValidExpansion<HasFooType, MemberTest>::value, "");
  static_assert(!hip::std::_IsValidExpansion<HasFooData, MemberTest2>::value, "");
  static_assert(hip::std::_IsValidExpansion<HasFooType, MemberTest2>::value, "");
  static_assert(hip::std::_IsValidExpansion<FuncCallable, MemberTest, int>::value, "");
  static_assert(!hip::std::_IsValidExpansion<FuncCallable, MemberTest, void*>::value, "");
}

__host__ __device__
void test_first_and_second_type() {
  ASSERT_SAME_TYPE(hip::std::_FirstType<int, long, void*>, int);
  ASSERT_SAME_TYPE(hip::std::_FirstType<char>, char);
  ASSERT_SAME_TYPE(hip::std::_SecondType<char, long>, long);
  ASSERT_SAME_TYPE(hip::std::_SecondType<long long, int, void*>, int);
}

int main(int, char**) {
  return 0;
}
