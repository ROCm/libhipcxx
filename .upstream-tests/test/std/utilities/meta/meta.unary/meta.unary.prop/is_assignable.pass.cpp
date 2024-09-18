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

// is_assignable

#include <hip/std/type_traits>
#include "test_macros.h"

struct A
{
};

struct B
{
    __host__ __device__
    void operator=(A);
};

template <class T, class U>
__host__ __device__
void test_is_assignable()
{
    static_assert(( hip::std::is_assignable<T, U>::value), "");
#if TEST_STD_VER > 11
    static_assert(  hip::std::is_assignable_v<T, U>, "");
#endif
}

template <class T, class U>
__host__ __device__
void test_is_not_assignable()
{
    static_assert((!hip::std::is_assignable<T, U>::value), "");
#if TEST_STD_VER > 11
    static_assert( !hip::std::is_assignable_v<T, U>, "");
#endif
}

struct D;

#if TEST_STD_VER >= 11
struct C
{
    template <class U>
    __host__ __device__
    D operator,(U&&);
};

struct E
{
    __host__ __device__
    C operator=(int);
};
#endif

template <typename T>
struct X { T t; };

int main(int, char**)
{
    test_is_assignable<int&, int&> ();
    test_is_assignable<int&, int> ();
    test_is_assignable<int&, double> ();
    test_is_assignable<B, A> ();
    test_is_assignable<void*&, void*> ();

#if TEST_STD_VER >= 11
    test_is_assignable<E, int> ();

    test_is_not_assignable<int, int&> ();
    test_is_not_assignable<int, int> ();
#endif
    test_is_not_assignable<A, B> ();
    test_is_not_assignable<void, const void> ();
    test_is_not_assignable<const void, const void> ();
    test_is_not_assignable<int(), int> ();

//  pointer to incomplete template type
    test_is_assignable<X<D>*&, X<D>*> ();

  return 0;
}
