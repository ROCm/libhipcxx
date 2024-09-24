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

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: gcc-8, gcc-9
// UNSUPPORTED: windows && (c++11 || c++14 || c++17)

// template<class F, class... Args>
// concept strict_weak_order;

#include <hip/std/concepts>

#include "test_macros.h"
#if TEST_STD_VER > 17

struct S1 {};
struct S2 {};

struct R {
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
  __host__ __device__ bool operator()(S2, S1) const;
  __host__ __device__ bool operator()(S2, S2) const;
};

// clang-format off
template<class F, class T, class U>
requires hip::std::relation<F, T, U>
__host__ __device__ constexpr bool check_strict_weak_order_subsumes_relation() {
  return false;
}

template<class F, class T, class U>
requires hip::std::strict_weak_order<F, T, U> && true
__host__ __device__ constexpr bool check_strict_weak_order_subsumes_relation() {
  return true;
}
// clang-format on

static_assert(check_strict_weak_order_subsumes_relation<int (*)(int, int), int, int>(), "");
static_assert(check_strict_weak_order_subsumes_relation<int (*)(int, double), int, double>(), "");
static_assert(check_strict_weak_order_subsumes_relation<R, S1, S1>(), "");
static_assert(check_strict_weak_order_subsumes_relation<R, S1, S2>(), "");

// clang-format off
template<class F, class T, class U>
requires hip::std::relation<F, T, U> && true
__host__ __device__ constexpr bool check_relation_subsumes_strict_weak_order() {
  return true;
}

template<class F, class T, class U>
requires hip::std::strict_weak_order<F, T, U>
__host__ __device__ constexpr bool check_relation_subsumes_strict_weak_order() {
  return false;
}
// clang-format on

static_assert(check_relation_subsumes_strict_weak_order<int (*)(int, int), int, int>(), "");
static_assert(check_relation_subsumes_strict_weak_order<int (*)(int, double), int, double>(), "");
static_assert(check_relation_subsumes_strict_weak_order<R, S1, S1>(), "");
static_assert(check_relation_subsumes_strict_weak_order<R, S1, S2>(), "");

// clang-format off
template<class F, class T, class U>
requires hip::std::strict_weak_order<F, T, T> && hip::std::strict_weak_order<F, U, U>
__host__ __device__ constexpr bool check_strict_weak_order_subsumes_itself() {
  return false;
}

template<class F, class T, class U>
requires hip::std::strict_weak_order<F, T, U>
__host__ __device__ constexpr bool check_strict_weak_order_subsumes_itself() {
  return true;
}
// clang-format on

static_assert(check_strict_weak_order_subsumes_itself<int (*)(int, int), int, int>(), "");
static_assert(check_strict_weak_order_subsumes_itself<R, S1, S1>(), "");

#endif // TEST_STD_VER > 17

int main(int, char**)
{
  return 0;
}
