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
// UNSUPPORTED: c++03, c++11, c++14

// gcc does not support deduction guides until gcc-7 and that is buggy
// UNSUPPORTED: gcc-6, gcc-7

// <span>

//   template<class It, class EndOrSize>
//     span(It, EndOrSize) -> span<remove_reference_t<iter_reference_t<_It>>>;
//
//   template<class T, size_t N>
//     span(T (&)[N]) -> span<T, N>;
//
//   template<class T, size_t N>
//     span(array<T, N>&) -> span<T, N>;
//
//   template<class T, size_t N>
//     span(const array<T, N>&) -> span<const T, N>;
//
//   template<class R>
//     span(R&&) -> span<remove_reference_t<ranges::range_reference_t<R>>>;


#include <hip/std/span>
#include <hip/std/array>
#include <hip/std/cassert>
#include <hip/std/iterator>

#include "test_macros.h"


__host__ __device__
void test_iterator_sentinel() {
  int arr[] = {1, 2, 3};
  {
  hip::std::span s{hip::std::begin(arr), hip::std::end(arr)};
  ASSERT_SAME_TYPE(decltype(s), hip::std::span<int>);
  assert(s.size() == hip::std::size(arr));
  assert(s.data() == hip::std::data(arr));
  }
  {
  hip::std::span s{hip::std::begin(arr), 3};
  ASSERT_SAME_TYPE(decltype(s), hip::std::span<int>);
  assert(s.size() == hip::std::size(arr));
  assert(s.data() == hip::std::data(arr));
  }
}

__host__ __device__
void test_c_array() {
    {
    int arr[] = {1, 2, 3};
    hip::std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), hip::std::span<int, 3>);
    assert(s.size() == hip::std::size(arr));
    assert(s.data() == hip::std::data(arr));
    }

    {
    const int arr[] = {1,2,3};
    hip::std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), hip::std::span<const int, 3>);
    assert(s.size() == hip::std::size(arr));
    assert(s.data() == hip::std::data(arr));
    }
}

__host__ __device__
void test_std_array() {
    {
    hip::std::array<double, 4> arr = {1.0, 2.0, 3.0, 4.0};
    hip::std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), hip::std::span<double, 4>);
    assert(s.size() == arr.size());
    assert(s.data() == arr.data());
    }

    {
    const hip::std::array<long, 5> arr = {4, 5, 6, 7, 8};
    hip::std::span s{arr};
    ASSERT_SAME_TYPE(decltype(s), hip::std::span<const long, 5>);
    assert(s.size() == arr.size());
    assert(s.data() == arr.data());
    }
}

int main(int, char**)
{
  test_iterator_sentinel();
  test_c_array();
  test_std_array();

  return 0;
}
