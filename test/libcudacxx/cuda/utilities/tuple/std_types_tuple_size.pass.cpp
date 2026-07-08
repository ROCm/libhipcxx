//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

// UNSUPPORTED: nvrtc, hiprtc

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include <array>
#include <complex>
#include <tuple>
#include <utility>

#include "test_macros.h"

template <class STD_TYPE, size_t Size>
TEST_FUNC constexpr void test()
{
  static_assert(cuda::std::tuple_size<STD_TYPE>::value == Size);
  static_assert(cuda::std::tuple_size<const STD_TYPE>::value == Size);
  static_assert(cuda::std::tuple_size<volatile STD_TYPE>::value == Size);
  static_assert(cuda::std::tuple_size<const volatile STD_TYPE>::value == Size);
}

TEST_FUNC constexpr bool test()
{
  // complex has a size of 2
  test<::std::complex<float>, 2>();
  test<::std::complex<double>, 2>();
#if _CCCL_HAS_NVFP16()
  test<::std::complex<__half>, 2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test<::std::complex<__nv_bfloat16>, 2>();
#endif // _CCCL_HAS_NVBF16()

  // pair always has a size of 2
  test<::std::pair<int, float>, 2>();

  // tuple has the size of the number of template arguments
  test<::std::tuple<int>, 1>();
  test<::std::tuple<int, int>, 2>();
  test<::std::tuple<int, int, int>, 3>();

  // array has the size of the number of elements
  test<::std::array<int, 4>, 4>();
  test<::std::array<int, 1337>, 1337>();

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test());
  return 0;
}
