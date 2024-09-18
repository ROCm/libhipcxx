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

// <cuda/std/span>

// template <class It, class End>
// constexpr explicit(Extent != dynamic_extent) span(It first, End last);
// Requires: [first, last) shall be a valid range.
//   If Extent is not equal to dynamic_extent, then last - first shall be equal to Extent.
//

#include <hip/std/span>
#include <hip/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class T, class Sentinel>
__host__ __device__ constexpr bool test_ctor() {
  T val[2] = {};
  auto s1 = hip::std::span<T>(hip::std::begin(val), Sentinel(hip::std::end(val)));
  auto s2 = hip::std::span<T, 2>(hip::std::begin(val), Sentinel(hip::std::end(val)));
  assert(s1.data() == hip::std::data(val) && s1.size() == hip::std::size(val));
  assert(s2.data() == hip::std::data(val) && s2.size() == hip::std::size(val));
  return true;
}

template <size_t Extent>
__host__ __device__ constexpr void test_constructibility() {
  static_assert(hip::std::is_constructible_v<hip::std::span<int, Extent>, int*, int*>, "");
  static_assert(!hip::std::is_constructible_v<hip::std::span<int, Extent>, const int*, const int*>, "");
  static_assert(!hip::std::is_constructible_v<hip::std::span<int, Extent>, volatile int*, volatile int*>, "");
  static_assert(hip::std::is_constructible_v<hip::std::span<const int, Extent>, int*, int*>, "");
  static_assert(hip::std::is_constructible_v<hip::std::span<const int, Extent>, const int*, const int*>, "");
  static_assert(!hip::std::is_constructible_v<hip::std::span<const int, Extent>, volatile int*, volatile int*>, "");
  static_assert(hip::std::is_constructible_v<hip::std::span<volatile int, Extent>, int*, int*>, "");
  static_assert(!hip::std::is_constructible_v<hip::std::span<volatile int, Extent>, const int*, const int*>, "");
  static_assert(hip::std::is_constructible_v<hip::std::span<volatile int, Extent>, volatile int*, volatile int*>, "");
  static_assert(!hip::std::is_constructible_v<hip::std::span<int, Extent>, int*, float*>, ""); // types wrong
}

__host__ __device__ constexpr bool test() {
  test_constructibility<hip::std::dynamic_extent>();
  test_constructibility<3>();
  struct A {};
  assert((test_ctor<int, int*>()));
  //assert((test_ctor<int, sized_sentinel<int*>>()));
  assert((test_ctor<A, A*>()));
  //assert((test_ctor<A, sized_sentinel<A*>>()));
  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
