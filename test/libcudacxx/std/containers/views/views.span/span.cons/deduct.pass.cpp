//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// XFAIL: enable-tile
// error: a non-__tile__ variable cannot be used in tile code

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

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
#  include <array>
#endif // !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)

#include "test_macros.h"

TEST_FUNC void test_iterator_sentinel()
{
  int arr[] = {1, 2, 3};
  {
    cuda::std::span s{cuda::std::begin(arr), cuda::std::end(arr)};
    static_assert(cuda::std::is_same_v<decltype(s), cuda::std::span<int>>);
    assert(s.size() == cuda::std::size(arr));
    assert(s.data() == cuda::std::data(arr));
  }
  {
    cuda::std::span s{cuda::std::begin(arr), 3};
    static_assert(cuda::std::is_same_v<decltype(s), cuda::std::span<int>>);
    assert(s.size() == cuda::std::size(arr));
    assert(s.data() == cuda::std::data(arr));
  }

  // P3029R1: deduction from `integral_constant`
  {
    cuda::std::span s{cuda::std::begin(arr), cuda::std::integral_constant<size_t, 3>{}};
    static_assert(cuda::std::is_same_v<decltype(s), cuda::std::span<int, 3>>);
    assert(s.size() == cuda::std::size(arr));
    assert(s.data() == cuda::std::data(arr));
  }
}

TEST_FUNC void test_c_array()
{
  {
    int arr[] = {1, 2, 3};
    cuda::std::span s{arr};
    static_assert(cuda::std::is_same_v<decltype(s), cuda::std::span<int, 3>>);
    assert(s.size() == cuda::std::size(arr));
    assert(s.data() == cuda::std::data(arr));
  }

  {
    const int arr[] = {1, 2, 3};
    cuda::std::span s{arr};
    static_assert(cuda::std::is_same_v<decltype(s), cuda::std::span<const int, 3>>);
    assert(s.size() == cuda::std::size(arr));
    assert(s.data() == cuda::std::data(arr));
  }
}

TEST_FUNC void test_cuda_std_array()
{
  {
    cuda::std::array<double, 4> arr = {1.0, 2.0, 3.0, 4.0};
    cuda::std::span s{arr};
    static_assert(cuda::std::is_same_v<decltype(s), cuda::std::span<double, 4>>);
    assert(s.size() == arr.size());
    assert(s.data() == arr.data());
  }

  {
    const cuda::std::array<long, 5> arr = {4, 5, 6, 7, 8};
    cuda::std::span s{arr};
    static_assert(cuda::std::is_same_v<decltype(s), cuda::std::span<const long, 5>>);
    assert(s.size() == arr.size());
    assert(s.data() == arr.data());
  }
}

#if !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)
void test_std_array()
{
  {
    std::array<double, 4> arr = {1.0, 2.0, 3.0, 4.0};
    cuda::std::span s{arr};
    static_assert(cuda::std::is_same_v<decltype(s), cuda::std::span<double, 4>>);
    assert(s.size() == arr.size());
    assert(s.data() == arr.data());
  }

  {
    const std::array<long, 5> arr = {4, 5, 6, 7, 8};
    cuda::std::span s{arr};
    static_assert(cuda::std::is_same_v<decltype(s), cuda::std::span<const long, 5>>);
    assert(s.size() == arr.size());
    assert(s.data() == arr.data());
  }
}
#endif // !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)

int main(int, char**)
{
  test_iterator_sentinel();
  test_c_array();
  test_cuda_std_array();
#if !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)
  NV_IF_TARGET(NV_IS_HOST, (test_std_array();))
#endif // !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)

  return 0;
}
