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

// template <class T> constexpr add_const<T>& as_const(T& t) noexcept;      // C++17
// template <class T>           add_const<T>& as_const(const T&&) = delete; // C++17

#include <hip/std/utility>
#include <hip/std/cassert>

#include "test_macros.h"

struct S {int i;};
__host__ __device__ bool operator==(const S& x, const S& y) { return x.i == y.i; }
__host__ __device__ bool operator==(const volatile S& x, const volatile S& y) { return x.i == y.i; }

template<typename T>
__host__ __device__ void test(T& t)
{
    static_assert(hip::std::is_const<typename hip::std::remove_reference<decltype(hip::std::as_const                  (t))>::type>::value, "");
    static_assert(hip::std::is_const<typename hip::std::remove_reference<decltype(hip::std::as_const<               T>(t))>::type>::value, "");
    static_assert(hip::std::is_const<typename hip::std::remove_reference<decltype(hip::std::as_const<const          T>(t))>::type>::value, "");
    static_assert(hip::std::is_const<typename hip::std::remove_reference<decltype(hip::std::as_const<volatile       T>(t))>::type>::value, "");
    static_assert(hip::std::is_const<typename hip::std::remove_reference<decltype(hip::std::as_const<const volatile T>(t))>::type>::value, "");

    assert(hip::std::as_const(t) == t);
    assert(hip::std::as_const<               T>(t) == t);
    assert(hip::std::as_const<const          T>(t) == t);
    assert(hip::std::as_const<volatile       T>(t) == t);
    assert(hip::std::as_const<const volatile T>(t) == t);
}

int main(int, char**)
{
    int i = 3;
    double d = 4.0;
    S s{2};
    test(i);
    test(d);
    test(s);

  return 0;
}
