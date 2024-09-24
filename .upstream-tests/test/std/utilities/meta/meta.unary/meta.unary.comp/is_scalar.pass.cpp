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

// is_scalar

#include <hip/std/type_traits>
#include <hip/std/cstddef>         // for hip::std::nullptr_t
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_scalar()
{
    static_assert( hip::std::is_scalar<T>::value, "");
    static_assert( hip::std::is_scalar<const T>::value, "");
    static_assert( hip::std::is_scalar<volatile T>::value, "");
    static_assert( hip::std::is_scalar<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_scalar_v<T>, "");
    static_assert( hip::std::is_scalar_v<const T>, "");
    static_assert( hip::std::is_scalar_v<volatile T>, "");
    static_assert( hip::std::is_scalar_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_scalar()
{
    static_assert(!hip::std::is_scalar<T>::value, "");
    static_assert(!hip::std::is_scalar<const T>::value, "");
    static_assert(!hip::std::is_scalar<volatile T>::value, "");
    static_assert(!hip::std::is_scalar<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_scalar_v<T>, "");
    static_assert(!hip::std::is_scalar_v<const T>, "");
    static_assert(!hip::std::is_scalar_v<volatile T>, "");
    static_assert(!hip::std::is_scalar_v<const volatile T>, "");
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
//  Arithmetic types (3.9.1), enumeration types, pointer types, pointer to member types (3.9.2),
//    hip::std::nullptr_t, and cv-qualified versions of these types (3.9.3)
//    are collectively called scalar types.

    test_is_scalar<hip::std::nullptr_t>();
    test_is_scalar<short>();
    test_is_scalar<unsigned short>();
    test_is_scalar<int>();
    test_is_scalar<unsigned int>();
    test_is_scalar<long>();
    test_is_scalar<unsigned long>();
    test_is_scalar<bool>();
    test_is_scalar<char>();
    test_is_scalar<signed char>();
    test_is_scalar<unsigned char>();
    test_is_scalar<wchar_t>();
    test_is_scalar<double>();
    test_is_scalar<int*>();
    test_is_scalar<const int*>();
    test_is_scalar<int Empty::*>();
    test_is_scalar<void (Empty::*)(int)>();
    test_is_scalar<Enum>();
    test_is_scalar<FunctionPtr>();

    test_is_not_scalar<void>();
    test_is_not_scalar<int&>();
    test_is_not_scalar<int&&>();
    test_is_not_scalar<char[3]>();
    test_is_not_scalar<char[]>();
    test_is_not_scalar<Union>();
    test_is_not_scalar<Empty>();
    test_is_not_scalar<incomplete_type>();
    test_is_not_scalar<bit_zero>();
    test_is_not_scalar<NotEmpty>();
    test_is_not_scalar<Abstract>();
    test_is_not_scalar<int(int)>();

  return 0;
}
