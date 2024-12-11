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


#ifndef LIBCXX_TEST_CONCEPTS_LANG_CONCEPTS_ARITHMETIC_H_
#define LIBCXX_TEST_CONCEPTS_LANG_CONCEPTS_ARITHMETIC_H_

#include <hip/std/concepts>

#include "test_macros.h"

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#if TEST_STD_VER > 17
// This overload should never be called. It exists solely to force subsumption.
template <hip::std::integral I>
__host__ __device__ constexpr bool CheckSubsumption(I) {
  return false;
}

template <hip::std::integral I>
requires hip::std::signed_integral<I> && (!hip::std::unsigned_integral<I>)
__host__ __device__ constexpr bool CheckSubsumption(I) {
  return hip::std::is_signed_v<I>;
}

template <hip::std::integral I>
requires hip::std::unsigned_integral<I> && (!hip::std::signed_integral<I>)
__host__ __device__ constexpr bool CheckSubsumption(I) {
  return hip::std::is_unsigned_v<I>;
}
#endif // TEST_STD_VER > 17

enum ClassicEnum { a, b, c };
enum class ScopedEnum { x, y, z };
struct EmptyStruct {};

#endif // LIBCXX_TEST_CONCEPTS_LANG_CONCEPTS_ARITHMETIC_H_
