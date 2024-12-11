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

// XFAIL: gcc-4.8
 
// type_traits

// add_pointer
// If T names a referenceable type or a (possibly cv-qualified) void type then
//    the member typedef type shall name the same type as remove_reference_t<T>*;
//    otherwise, type shall name T.

#include <hip/std/type_traits>
#include "test_macros.h"

template <class T, class U>
__host__ __device__
void test_add_pointer()
{
    ASSERT_SAME_TYPE(U, typename hip::std::add_pointer<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U, hip::std::add_pointer_t<T>);
#endif
}

template <class F>
__host__ __device__
void test_function0()
{
    ASSERT_SAME_TYPE(F*, typename hip::std::add_pointer<F>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(F*, hip::std::add_pointer_t<F>);
#endif
}

template <class F>
__host__ __device__
void test_function1()
{
    ASSERT_SAME_TYPE(F, typename hip::std::add_pointer<F>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(F, hip::std::add_pointer_t<F>);
#endif
}

struct Foo {};

int main(int, char**)
{
    test_add_pointer<void, void*>();
    test_add_pointer<int, int*>();
    test_add_pointer<int[3], int(*)[3]>();
    test_add_pointer<int&, int*>();
    test_add_pointer<const int&, const int*>();
    test_add_pointer<int*, int**>();
    test_add_pointer<const int*, const int**>();
    test_add_pointer<Foo, Foo*>();

//  LWG 2101 specifically talks about add_pointer and functions.
//  The term of art is "a referenceable type", which a cv- or ref-qualified function is not.
    test_function0<void()>();
#if TEST_STD_VER >= 11
    test_function1<void() const>();
    test_function1<void() &>();
    test_function1<void() &&>();
    test_function1<void() const &>();
    test_function1<void() const &&>();
#endif

//  But a cv- or ref-qualified member function *is* "a referenceable type"
    test_function0<void (Foo::*)()>();
#if TEST_STD_VER >= 11
    test_function0<void (Foo::*)() const>();
    test_function0<void (Foo::*)() &>();
    test_function0<void (Foo::*)() &&>();
    test_function0<void (Foo::*)() const &>();
    test_function0<void (Foo::*)() const &&>();
#endif

  return 0;
}
