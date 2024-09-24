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
// UNSUPPORTED: windows && (c++11 || c++14 || c++17)

// template<class T>
// concept integral = // see below

#include <hip/std/concepts>
#include <hip/std/type_traits>

#include "test_macros.h"
#include "arithmetic.h"

using hip::std::integral;

template <typename T>
__host__ __device__ constexpr bool CheckIntegralQualifiers() {
  constexpr bool result = integral<T>;
  static_assert(integral<const T> == result, "");
  static_assert(integral<volatile T> == result, "");
  static_assert(integral<const volatile T> == result, "");

  static_assert(!integral<T&>, "");
  static_assert(!integral<const T&>, "");
  static_assert(!integral<volatile T&>, "");
  static_assert(!integral<const volatile T&>, "");

  static_assert(!integral<T&&>, "");
  static_assert(!integral<const T&&>, "");
  static_assert(!integral<volatile T&&>, "");
  static_assert(!integral<const volatile T&&>, "");

  static_assert(!integral<T*>, "");
  static_assert(!integral<const T*>, "");
  static_assert(!integral<volatile T*>, "");
  static_assert(!integral<const volatile T*>, "");

  static_assert(!integral<T (*)()>, "");
  static_assert(!integral<T (&)()>, "");
  static_assert(!integral<T(&&)()>, "");

  return result;
}

// standard signed and unsigned integers
static_assert(CheckIntegralQualifiers<signed char>(), "");
static_assert(CheckIntegralQualifiers<unsigned char>(), "");
static_assert(CheckIntegralQualifiers<short>(), "");
static_assert(CheckIntegralQualifiers<unsigned short>(), "");
static_assert(CheckIntegralQualifiers<int>(), "");
static_assert(CheckIntegralQualifiers<unsigned int>(), "");
static_assert(CheckIntegralQualifiers<long>(), "");
static_assert(CheckIntegralQualifiers<unsigned long>(), "");
static_assert(CheckIntegralQualifiers<long long>(), "");
static_assert(CheckIntegralQualifiers<unsigned long long>(), "");

// extended integers
#ifndef TEST_HAS_NO_INT128_T
static_assert(CheckIntegralQualifiers<__int128_t>(), "");
static_assert(CheckIntegralQualifiers<__uint128_t>(), "");
#endif

// bool and char types are also integral
static_assert(CheckIntegralQualifiers<wchar_t>(), "");
static_assert(CheckIntegralQualifiers<bool>(), "");
static_assert(CheckIntegralQualifiers<char>(), "");
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
static_assert(CheckIntegralQualifiers<char8_t>(), "");
#endif // TEST_STD_VER > 17 && defined(__cpp_char8_t)
static_assert(CheckIntegralQualifiers<char16_t>(), "");
static_assert(CheckIntegralQualifiers<char32_t>(), "");

// types that aren't integral
static_assert(!integral<void>, "");
static_assert(!CheckIntegralQualifiers<float>(), "");
static_assert(!CheckIntegralQualifiers<double>(), "");
static_assert(!CheckIntegralQualifiers<long double>(), "");

static_assert(!CheckIntegralQualifiers<ClassicEnum>(), "");

static_assert(!CheckIntegralQualifiers<ScopedEnum>(), "");

static_assert(!CheckIntegralQualifiers<EmptyStruct>(), "");
static_assert(!CheckIntegralQualifiers<int EmptyStruct::*>(), "");
static_assert(!CheckIntegralQualifiers<int (EmptyStruct::*)()>(), "");

#if TEST_STD_VER > 17
static_assert(CheckSubsumption(0), "");
static_assert(CheckSubsumption(0U), "");
#endif // TEST_STD_VER > 17

int main(int, char**) { return 0; }
