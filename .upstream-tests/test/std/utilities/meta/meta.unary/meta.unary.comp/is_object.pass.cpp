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

// is_object

#include <hip/std/type_traits>
#include <hip/std/cstddef>         // for hip::std::nullptr_t
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_object()
{
    static_assert( hip::std::is_object<T>::value, "");
    static_assert( hip::std::is_object<const T>::value, "");
    static_assert( hip::std::is_object<volatile T>::value, "");
    static_assert( hip::std::is_object<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_object_v<T>, "");
    static_assert( hip::std::is_object_v<const T>, "");
    static_assert( hip::std::is_object_v<volatile T>, "");
    static_assert( hip::std::is_object_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_object()
{
    static_assert(!hip::std::is_object<T>::value, "");
    static_assert(!hip::std::is_object<const T>::value, "");
    static_assert(!hip::std::is_object<volatile T>::value, "");
    static_assert(!hip::std::is_object<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_object_v<T>, "");
    static_assert(!hip::std::is_object_v<const T>, "");
    static_assert(!hip::std::is_object_v<volatile T>, "");
    static_assert(!hip::std::is_object_v<const volatile T>, "");
#endif
}

class incomplete_type;

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

enum Enum {zero, one};

typedef void (*FunctionPtr)();


int main(int, char**)
{
// An object type is a (possibly cv-qualified) type that is not a function type,
// not a reference type, and not a void type.

    test_is_object<hip::std::nullptr_t>();
    test_is_object<void *>();
    test_is_object<char[3]>();
    test_is_object<char[]>();
    test_is_object<int>();
    test_is_object<int*>();
    test_is_object<Union>();
    test_is_object<int*>();
    test_is_object<const int*>();
    test_is_object<Enum>();
    test_is_object<incomplete_type>();
    test_is_object<bit_zero>();
    test_is_object<NotEmpty>();
    test_is_object<Abstract>();
    test_is_object<FunctionPtr>();
    test_is_object<int Empty::*>();
    test_is_object<void (Empty::*)(int)>();

    test_is_not_object<void>();
    test_is_not_object<int&>();
    test_is_not_object<int&&>();
    test_is_not_object<int(int)>();

  return 0;
}
