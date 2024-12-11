//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
//

// <cuda/std/type_traits>
// UNSUPPORTED: c++98, c++03, c++11

#include <hip/std/type_traits>

#include "test_macros.h"

struct A {};
struct B {
public:
    __host__ __device__ operator A() { return a; } A a;
};

class C { };
class D {
public:
    __host__ __device__ operator C() noexcept { return c; } C c;
};

int main(int, char**) {
    static_assert((hip::std::is_nothrow_convertible<int, double>::value), "");
    static_assert(!(hip::std::is_nothrow_convertible<int, char*>::value), "");

    static_assert(!(hip::std::is_nothrow_convertible<A, B>::value), "");
    static_assert((hip::std::is_nothrow_convertible<D, C>::value), "");

    static_assert((hip::std::is_nothrow_convertible_v<int, double>), "");
    static_assert(!(hip::std::is_nothrow_convertible_v<int, char*>), "");

    static_assert(!(hip::std::is_nothrow_convertible_v<A, B>), "");
    static_assert((hip::std::is_nothrow_convertible_v<D, C>), "");

    static_assert((hip::std::is_nothrow_convertible_v<const void, void>), "");
    static_assert((hip::std::is_nothrow_convertible_v<volatile void, void>), "");
    static_assert((hip::std::is_nothrow_convertible_v<void, const void>), "");
    static_assert((hip::std::is_nothrow_convertible_v<void, volatile void>), "");

    static_assert(!(hip::std::is_nothrow_convertible_v<int[], double[]>), "");
    static_assert(!(hip::std::is_nothrow_convertible_v<int[], int[]>), "");
    static_assert(!(hip::std::is_nothrow_convertible_v<int[10], int[10]>), "");
    static_assert(!(hip::std::is_nothrow_convertible_v<int[10], double[10]>), "");
    static_assert(!(hip::std::is_nothrow_convertible_v<int[5], double[10]>), "");
    static_assert(!(hip::std::is_nothrow_convertible_v<int[10], A[10]>), "");

    typedef void V();
    typedef int I();
    static_assert(!(hip::std::is_nothrow_convertible_v<V, V>), "");
    static_assert(!(hip::std::is_nothrow_convertible_v<V, I>), "");

    return 0;
}
