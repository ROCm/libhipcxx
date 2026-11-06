//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/type_traits>

#include "cuda_fp_types.h"
#include "test_macros.h"

template <class T>
TEST_FUNC void test_is_bitwise_comparable()
{
  static_assert(cuda::is_bitwise_comparable<T>::value);
  static_assert(cuda::is_bitwise_comparable_v<T>);
  static_assert(cuda::is_bitwise_comparable_v<const T>);
  static_assert(cuda::is_bitwise_comparable_v<volatile T>);
  static_assert(cuda::is_bitwise_comparable_v<const volatile T>);
}

template <class T>
TEST_FUNC void test_is_not_bitwise_comparable()
{
  static_assert(!cuda::is_bitwise_comparable<T>::value);
  static_assert(!cuda::is_bitwise_comparable_v<T>);
  static_assert(!cuda::is_bitwise_comparable_v<const T>);
  static_assert(!cuda::is_bitwise_comparable_v<volatile T>);
  static_assert(!cuda::is_bitwise_comparable_v<const volatile T>);
}

TEST_FUNC void test_basic_types()
{
  // types with unique object representations
  test_is_bitwise_comparable<bool>();
  test_is_bitwise_comparable<int>();
  test_is_bitwise_comparable<unsigned>();
  test_is_bitwise_comparable<char>();
  test_is_bitwise_comparable<unsigned char>();
  test_is_bitwise_comparable<short>();
  test_is_bitwise_comparable<long long>();

  // types without unique object representations
  test_is_not_bitwise_comparable<float>();
  test_is_not_bitwise_comparable<double>();

  // arrays
  static_assert(cuda::is_bitwise_comparable_v<int[]>);
  static_assert(cuda::is_bitwise_comparable_v<const int[]>);
  static_assert(cuda::is_bitwise_comparable_v<int[4]>);
  static_assert(cuda::is_bitwise_comparable_v<const int[4]>);
  static_assert(!cuda::is_bitwise_comparable_v<float[]>);
  static_assert(!cuda::is_bitwise_comparable_v<const float[]>);

  // extended floating-point scalar types
#if _CCCL_HAS_NVFP16()
  test_is_not_bitwise_comparable<__half>();
  test_is_not_bitwise_comparable<__half2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_is_not_bitwise_comparable<__hip_bfloat16>();
  test_is_not_bitwise_comparable<__hip_bfloat162>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_is_not_bitwise_comparable<__hip_fp8_e4m3>();
  test_is_not_bitwise_comparable<__hip_fp8x2_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
}

int main(int, char**)
{
  test_basic_types();
  return 0;
}
