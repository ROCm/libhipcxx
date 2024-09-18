//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// UNSUPPORTED: c++03, c++11

// <span>

// template<size_t N>
//     constexpr span(array<value_type, N>& arr) noexcept;
// template<size_t N>
//     constexpr span(const array<value_type, N>& arr) noexcept;
//
// Remarks: These constructors shall not participate in overload resolution unless:
//   — extent == dynamic_extent || N == extent is true, and
//   — remove_pointer_t<decltype(data(arr))>(*)[] is convertible to ElementType(*)[].
//


#include <hip/std/span>
#include <hip/std/array>
#include <hip/std/cassert>

#include "test_macros.h"

__host__ __device__
void checkCV()
{
    hip::std::array<int, 3> arr  = {1,2,3};
//  STL says these are not cromulent
//  std::array<const int,3> carr = {4,5,6};
//  std::array<volatile int, 3> varr = {7,8,9};
//  std::array<const volatile int, 3> cvarr = {1,3,5};

//  Types the same (dynamic sized)
    {
    hip::std::span<               int> s1{  arr};    // a hip::std::span<               int> pointing at int.
    }

//  Types the same (static sized)
    {
    hip::std::span<               int,3> s1{  arr};  // a hip::std::span<               int> pointing at int.
    }


//  types different (dynamic sized)
    {
    hip::std::span<const          int> s1{ arr};     // a hip::std::span<const          int> pointing at int.
    hip::std::span<      volatile int> s2{ arr};     // a hip::std::span<      volatile int> pointing at int.
    hip::std::span<      volatile int> s3{ arr};     // a hip::std::span<      volatile int> pointing at const int.
    hip::std::span<const volatile int> s4{ arr};     // a hip::std::span<const volatile int> pointing at int.
    }

//  types different (static sized)
    {
    hip::std::span<const          int,3> s1{ arr};   // a hip::std::span<const          int> pointing at int.
    hip::std::span<      volatile int,3> s2{ arr};   // a hip::std::span<      volatile int> pointing at int.
    hip::std::span<      volatile int,3> s3{ arr};   // a hip::std::span<      volatile int> pointing at const int.
    hip::std::span<const volatile int,3> s4{ arr};   // a hip::std::span<const volatile int> pointing at int.
    }
}

template <typename T, typename U>
__host__ __device__
TEST_CONSTEXPR_CXX17 bool testConstructorArray() {
  hip::std::array<U, 2> val = {U(), U()};
  ASSERT_NOEXCEPT(hip::std::span<T>{val});
  ASSERT_NOEXCEPT(hip::std::span<T, 2>{val});
  hip::std::span<T> s1{val};
  hip::std::span<T, 2> s2{val};
  return s1.data() == &val[0] && s1.size() == 2 && s2.data() == &val[0] &&
         s2.size() == 2;
}

template <typename T, typename U>
__host__ __device__
TEST_CONSTEXPR_CXX17 bool testConstructorConstArray() {
  const hip::std::array<U, 2> val = {U(), U()};
  ASSERT_NOEXCEPT(hip::std::span<const T>{val});
  ASSERT_NOEXCEPT(hip::std::span<const T, 2>{val});
  hip::std::span<const T> s1{val};
  hip::std::span<const T, 2> s2{val};
  return s1.data() == &val[0] && s1.size() == 2 && s2.data() == &val[0] &&
         s2.size() == 2;
}

template <typename T>
__host__ __device__
TEST_CONSTEXPR_CXX17 bool testConstructors() {
  STATIC_ASSERT_CXX17((testConstructorArray<T, T>()));
  STATIC_ASSERT_CXX17((testConstructorArray<const T, const T>()));
  STATIC_ASSERT_CXX17((testConstructorArray<const T, T>()));
  STATIC_ASSERT_CXX17((testConstructorConstArray<T, T>()));
  STATIC_ASSERT_CXX17((testConstructorConstArray<const T, const T>()));
  STATIC_ASSERT_CXX17((testConstructorConstArray<const T, T>()));

  return testConstructorArray<T, T>() &&
         testConstructorArray<const T, const T>() &&
         testConstructorArray<const T, T>() &&
         testConstructorConstArray<T, T>() &&
         testConstructorConstArray<const T, const T>() &&
         testConstructorConstArray<const T, T>();
}

struct A{};

int main(int, char**)
{
    assert(testConstructors<int>());
    assert(testConstructors<long>());
    assert(testConstructors<double>());
    assert(testConstructors<A>());

    assert(testConstructors<int*>());
    assert(testConstructors<const int*>());

    checkCV();

    return 0;
}
