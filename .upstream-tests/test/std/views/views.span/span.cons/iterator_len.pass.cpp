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

// template <class It>
// constexpr explicit(Extent != dynamic_extent) span(It first, size_type count);
//  If Extent is not equal to dynamic_extent, then count shall be equal to Extent.
//


#include <hip/std/span>
#include <hip/std/cassert>
#include <hip/std/iterator>
#include <hip/std/type_traits>

#include "test_macros.h"

template <size_t Extent>
__host__ __device__
constexpr bool test_constructibility() {
  struct Other {};
  static_assert(hip::std::is_constructible<hip::std::span<int, Extent>, int*, size_t>::value, "");
  static_assert(!hip::std::is_constructible<hip::std::span<int, Extent>, const int*, size_t>::value, "");
  static_assert(hip::std::is_constructible<hip::std::span<const int, Extent>, int*, size_t>::value, "");
  static_assert(hip::std::is_constructible<hip::std::span<const int, Extent>, const int*, size_t>::value, "");
  static_assert(!hip::std::is_constructible<hip::std::span<int, Extent>, volatile int*, size_t>::value, "");
  static_assert(!hip::std::is_constructible<hip::std::span<int, Extent>, const volatile int*, size_t>::value, "");
  static_assert(!hip::std::is_constructible<hip::std::span<const int, Extent>, volatile int*, size_t>::value, "");
  static_assert(!hip::std::is_constructible<hip::std::span<const int, Extent>, const volatile int*, size_t>::value, "");
  static_assert(!hip::std::is_constructible<hip::std::span<volatile int, Extent>, const int*, size_t>::value, "");
  static_assert(!hip::std::is_constructible<hip::std::span<volatile int, Extent>, const volatile int*, size_t>::value, "");
  static_assert(
      !hip::std::is_constructible<hip::std::span<int, Extent>, double*, size_t>::value, ""); // iterator type differs from span type
  static_assert(!hip::std::is_constructible<hip::std::span<int, Extent>, size_t, size_t>::value, "");
  static_assert(!hip::std::is_constructible<hip::std::span<int, Extent>, Other*, size_t>::value, ""); // unrelated iterator type

  return true;
}

template <class T>
__host__ __device__
constexpr bool test_ctor() {
  T val[2] = {};
  auto s1 = hip::std::span<T>(val, 2);
  auto s2 = hip::std::span<T, 2>(val, 2);
  assert(s1.data() == val && s1.size() == 2);
  assert(s2.data() == val && s2.size() == 2);
  return true;
}

__host__ __device__
constexpr bool test() {
  test_constructibility<hip::std::dynamic_extent>();
  test_constructibility<3>();

  struct A {};
  test_ctor<int>();
  test_ctor<A>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
