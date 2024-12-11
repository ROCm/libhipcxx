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
// concept floating_point = // see below

#include <hip/std/concepts>
#include <hip/std/type_traits>

#include "test_macros.h"
#include "arithmetic.h"

using hip::std::floating_point;

template <typename T>
__host__ __device__ constexpr bool CheckFloatingPointQualifiers() {
  constexpr bool result = floating_point<T>;
  static_assert(floating_point<const T> == result, "");
  static_assert(floating_point<volatile T> == result, "");
  static_assert(floating_point<const volatile T> == result, "");

  static_assert(!floating_point<T&>, "");
  static_assert(!floating_point<const T&>, "");
  static_assert(!floating_point<volatile T&>, "");
  static_assert(!floating_point<const volatile T&>, "");

  static_assert(!floating_point<T&&>, "");
  static_assert(!floating_point<const T&&>, "");
  static_assert(!floating_point<volatile T&&>, "");
  static_assert(!floating_point<const volatile T&&>, "");

  static_assert(!floating_point<T*>, "");
  static_assert(!floating_point<const T*>, "");
  static_assert(!floating_point<volatile T*>, "");
  static_assert(!floating_point<const volatile T*>, "");

  static_assert(!floating_point<T (*)()>, "");
  static_assert(!floating_point<T (&)()>, "");
  static_assert(!floating_point<T(&&)()>, "");

  return result;
}

// floating-point types
static_assert(CheckFloatingPointQualifiers<float>(), "");
static_assert(CheckFloatingPointQualifiers<double>(), "");
static_assert(CheckFloatingPointQualifiers<long double>(), "");

// types that aren't floating-point
static_assert(!CheckFloatingPointQualifiers<signed char>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned char>(), "");
static_assert(!CheckFloatingPointQualifiers<short>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned short>(), "");
static_assert(!CheckFloatingPointQualifiers<int>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned int>(), "");
static_assert(!CheckFloatingPointQualifiers<long>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned long>(), "");
static_assert(!CheckFloatingPointQualifiers<long long>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned long long>(), "");
static_assert(!CheckFloatingPointQualifiers<wchar_t>(), "");
static_assert(!CheckFloatingPointQualifiers<bool>(), "");
static_assert(!CheckFloatingPointQualifiers<char>(), "");
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
static_assert(!CheckFloatingPointQualifiers<char8_t>(), "");
#endif // TEST_STD_VER > 17 && defined(__cpp_char8_t)
static_assert(!CheckFloatingPointQualifiers<char16_t>(), "");
static_assert(!CheckFloatingPointQualifiers<char32_t>(), "");
static_assert(!floating_point<void>, "");

static_assert(!CheckFloatingPointQualifiers<ClassicEnum>(), "");
static_assert(!CheckFloatingPointQualifiers<ScopedEnum>(), "");
static_assert(!CheckFloatingPointQualifiers<EmptyStruct>(), "");
static_assert(!CheckFloatingPointQualifiers<int EmptyStruct::*>(), "");
static_assert(!CheckFloatingPointQualifiers<int (EmptyStruct::*)()>(), "");

int main(int, char**) { return 0; }
