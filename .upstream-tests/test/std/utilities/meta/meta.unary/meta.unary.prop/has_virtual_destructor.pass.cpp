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

// has_virtual_destructor

#include <hip/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__
void test_has_virtual_destructor()
{
    static_assert( hip::std::has_virtual_destructor<T>::value, "");
    static_assert( hip::std::has_virtual_destructor<const T>::value, "");
    static_assert( hip::std::has_virtual_destructor<volatile T>::value, "");
    static_assert( hip::std::has_virtual_destructor<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::has_virtual_destructor_v<T>, "");
    static_assert( hip::std::has_virtual_destructor_v<const T>, "");
    static_assert( hip::std::has_virtual_destructor_v<volatile T>, "");
    static_assert( hip::std::has_virtual_destructor_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_has_not_virtual_destructor()
{
    static_assert(!hip::std::has_virtual_destructor<T>::value, "");
    static_assert(!hip::std::has_virtual_destructor<const T>::value, "");
    static_assert(!hip::std::has_virtual_destructor<volatile T>::value, "");
    static_assert(!hip::std::has_virtual_destructor<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::has_virtual_destructor_v<T>, "");
    static_assert(!hip::std::has_virtual_destructor_v<const T>, "");
    static_assert(!hip::std::has_virtual_destructor_v<volatile T>, "");
    static_assert(!hip::std::has_virtual_destructor_v<const volatile T>, "");
#endif
}

class Empty
{
};

class NotEmpty
{
    __host__ __device__
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    __host__ __device__
    virtual ~Abstract() = 0;
};

struct A
{
    __host__ __device__
    ~A();
};

int main(int, char**)
{
    test_has_not_virtual_destructor<void>();
    test_has_not_virtual_destructor<A>();
    test_has_not_virtual_destructor<int&>();
    test_has_not_virtual_destructor<Union>();
    test_has_not_virtual_destructor<Empty>();
    test_has_not_virtual_destructor<int>();
    test_has_not_virtual_destructor<double>();
    test_has_not_virtual_destructor<int*>();
    test_has_not_virtual_destructor<const int*>();
    test_has_not_virtual_destructor<char[3]>();
    test_has_not_virtual_destructor<char[]>();
    test_has_not_virtual_destructor<bit_zero>();

    test_has_virtual_destructor<Abstract>();
    test_has_virtual_destructor<NotEmpty>();

  return 0;
}
