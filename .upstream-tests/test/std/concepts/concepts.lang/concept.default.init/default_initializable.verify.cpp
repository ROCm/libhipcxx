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
//     concept default_initializable = constructible_from<T> &&
//     requires { T{}; } &&
//     is-default-initializable<T>;

#include <hip/std/concepts>
#include <hip/std/cassert>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#include "test_macros.h"

template<class T>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  brace_initializable_,
  requires() //
  ( T{}));

template<class T>
_LIBCUDACXX_CONCEPT brace_initializable = _LIBCUDACXX_FRAGMENT(brace_initializable_, T);

__host__ __device__ void test() {
    // LWG3149
    // Changed the concept from constructible_from<T>
    // to constructible_from<T> &&
    //    requires { T{}; } && is-default-initializable <T>
    struct S0 { explicit S0() = default; };
    S0 x0;
    S0 y0{};
    static_assert(hip::std::constructible_from<S0>, "");
    static_assert(brace_initializable<S0>, "");
    static_assert(hip::std::__default_initializable<S0>, "");
    static_assert(hip::std::default_initializable<S0>, "");

    struct S1 { S0 x; }; // Note: aggregate
    S1 x1;
    S1 y1{}; // expected-error {{chosen constructor is explicit in copy-initialization}}
    static_assert(hip::std::constructible_from<S1>, "");
    static_assert(!brace_initializable<S1>, "");
    static_assert(hip::std::__default_initializable<S1>, "");
    static_assert(!hip::std::default_initializable<S1>, "");

    const int x2; // expected-error {{default initialization of an object of const type 'const int'}}
    const int y2{};

    static_assert(hip::std::constructible_from<const int>, "");
    static_assert(brace_initializable<const int>, "");
    static_assert(!hip::std::__default_initializable<const int>, "");
    static_assert(!hip::std::default_initializable<const int>, "");

    const int x3[1]; // expected-error-re {{default initialization of an object of const type 'const int{{[ ]*}}[1]'}}
    const int y3[1]{};
    static_assert(hip::std::constructible_from<const int[1]>, "");
    static_assert(brace_initializable<const int[1]>, "");
    static_assert(!hip::std::__default_initializable<const int[1]>, "");
    static_assert(!hip::std::default_initializable<const int[1]>, "");

    // Zero-length array extension
    const int x4[]; // expected-error {{definition of variable with array type needs an explicit size or an initializer}}
    const int y4[]{};
    static_assert(!hip::std::constructible_from<const int[]>, "");
    static_assert(brace_initializable<const int[]>, "");
    static_assert(!hip::std::__default_initializable<const int[]>, "");
    static_assert(!hip::std::default_initializable<const int[]>, "");
}

int main(int, char**) {
    test();

    return 0;
}
