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
// concept unsigned_integral = // see below

#include <hip/std/concepts>
#include <hip/std/type_traits>

#include "test_macros.h"
#include "arithmetic.h"

using hip::std::unsigned_integral;

template <typename T>
__host__ __device__ constexpr bool CheckUnsignedIntegralQualifiers() {
  constexpr bool result = unsigned_integral<T>;
  static_assert(unsigned_integral<const T> == result, "");
  static_assert(unsigned_integral<volatile T> == result, "");
  static_assert(unsigned_integral<const volatile T> == result, "");

  static_assert(!unsigned_integral<T&>, "");
  static_assert(!unsigned_integral<const T&>, "");
  static_assert(!unsigned_integral<volatile T&>, "");
  static_assert(!unsigned_integral<const volatile T&>, "");

  static_assert(!unsigned_integral<T&&>, "");
  static_assert(!unsigned_integral<const T&&>, "");
  static_assert(!unsigned_integral<volatile T&&>, "");
  static_assert(!unsigned_integral<const volatile T&&>, "");

  static_assert(!unsigned_integral<T*>, "");
  static_assert(!unsigned_integral<const T*>, "");
  static_assert(!unsigned_integral<volatile T*>, "");
  static_assert(!unsigned_integral<const volatile T*>, "");

  static_assert(!unsigned_integral<T (*)()>, "");
  static_assert(!unsigned_integral<T (&)()>, "");
  static_assert(!unsigned_integral<T(&&)()>, "");

  return result;
}

// standard unsigned types
static_assert(CheckUnsignedIntegralQualifiers<unsigned char>(), "");
static_assert(CheckUnsignedIntegralQualifiers<unsigned short>(), "");
static_assert(CheckUnsignedIntegralQualifiers<unsigned int>(), "");
static_assert(CheckUnsignedIntegralQualifiers<unsigned long>(), "");
static_assert(CheckUnsignedIntegralQualifiers<unsigned long long>(), "");

// Whether bool and character types are signed or unsigned is impl-defined
static_assert(CheckUnsignedIntegralQualifiers<wchar_t>() ==
              !hip::std::is_signed_v<wchar_t>, "");
static_assert(CheckUnsignedIntegralQualifiers<bool>() ==
              !hip::std::is_signed_v<bool>, "");
static_assert(CheckUnsignedIntegralQualifiers<char>() ==
              !hip::std::is_signed_v<char>, "");
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
static_assert(CheckUnsignedIntegralQualifiers<char8_t>() ==
              !hip::std::is_signed_v<char8_t>, "");
#endif // TEST_STD_VER > 17 && defined(__cpp_char8_t)
static_assert(CheckUnsignedIntegralQualifiers<char16_t>() ==
              !hip::std::is_signed_v<char16_t>, "");
static_assert(CheckUnsignedIntegralQualifiers<char32_t>() ==
              !hip::std::is_signed_v<char32_t>, "");

// extended integers
#ifndef TEST_HAS_NO_INT128_T
static_assert(CheckUnsignedIntegralQualifiers<__uint128_t>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<__int128_t>(), "");
#endif

// integer types that aren't unsigned integrals
static_assert(!CheckUnsignedIntegralQualifiers<signed char>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<short>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<int>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<long>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<long long>(), "");

static_assert(!unsigned_integral<void>, "");
static_assert(!CheckUnsignedIntegralQualifiers<float>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<double>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<long double>(), "");

static_assert(!CheckUnsignedIntegralQualifiers<ClassicEnum>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<ScopedEnum>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<EmptyStruct>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<int EmptyStruct::*>(), "");
static_assert(!CheckUnsignedIntegralQualifiers<int (EmptyStruct::*)()>(), "");

#if TEST_STD_VER > 17
static_assert(CheckSubsumption(0), "");
static_assert(CheckSubsumption(0U), "");
#endif // TEST_STD_VER > 17

int main(int, char**) { return 0; }
