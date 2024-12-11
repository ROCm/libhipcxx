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

// type_traits

// is_trivially_assignable

// XFAIL: gcc-4.8, gcc-4.9

#include <hip/std/type_traits>
#include "test_macros.h"

template <class T, class U>
__host__ __device__
void test_is_trivially_assignable()
{
    static_assert(( hip::std::is_trivially_assignable<T, U>::value), "");
#if TEST_STD_VER > 11
    static_assert(( hip::std::is_trivially_assignable_v<T, U>), "");
#endif
}

template <class T, class U>
__host__ __device__
void test_is_not_trivially_assignable()
{
    static_assert((!hip::std::is_trivially_assignable<T, U>::value), "");
#if TEST_STD_VER > 11
    static_assert((!hip::std::is_trivially_assignable_v<T, U>), "");
#endif
}

struct A
{
};

struct B
{
    __host__ __device__
    void operator=(A);
};

struct C
{
    __host__ __device__
    void operator=(C&);  // not const
};

int main(int, char**)
{
    test_is_trivially_assignable<int&, int&> ();
    test_is_trivially_assignable<int&, int> ();
    test_is_trivially_assignable<int&, double> ();

    test_is_not_trivially_assignable<int, int&> ();
    test_is_not_trivially_assignable<int, int> ();
    test_is_not_trivially_assignable<B, A> ();
    test_is_not_trivially_assignable<A, B> ();
    test_is_not_trivially_assignable<C&, C&> ();

  return 0;
}
