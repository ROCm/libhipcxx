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

// is_literal_type

// is_literal_type has been deprecated in C++17
#pragma nv_diag_suppress 1215

#include <hip/std/type_traits>
#include <hip/std/cstddef>       // for hip::std::nullptr_t
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_literal_type()
{
    static_assert( hip::std::is_literal_type<T>::value, "");
    static_assert( hip::std::is_literal_type<const T>::value, "");
    static_assert( hip::std::is_literal_type<volatile T>::value, "");
    static_assert( hip::std::is_literal_type<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_literal_type_v<T>, "");
    static_assert( hip::std::is_literal_type_v<const T>, "");
    static_assert( hip::std::is_literal_type_v<volatile T>, "");
    static_assert( hip::std::is_literal_type_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_literal_type()
{
    static_assert(!hip::std::is_literal_type<T>::value, "");
    static_assert(!hip::std::is_literal_type<const T>::value, "");
    static_assert(!hip::std::is_literal_type<volatile T>::value, "");
    static_assert(!hip::std::is_literal_type<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_literal_type_v<T>, "");
    static_assert(!hip::std::is_literal_type_v<const T>, "");
    static_assert(!hip::std::is_literal_type_v<volatile T>, "");
    static_assert(!hip::std::is_literal_type_v<const volatile T>, "");
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

enum Enum {zero, one};

typedef void (*FunctionPtr)();

int main(int, char**)
{
#if TEST_STD_VER >= 11
    test_is_literal_type<hip::std::nullptr_t>();
#endif

// Before C++14, void was not a literal type
// In C++14, cv-void is a literal type
#if TEST_STD_VER < 14
    test_is_not_literal_type<void>();
#else
    test_is_literal_type<void>();
#endif

    test_is_literal_type<int>();
    test_is_literal_type<int*>();
    test_is_literal_type<const int*>();
    test_is_literal_type<int&>();
#if TEST_STD_VER >= 11
    test_is_literal_type<int&&>();
#endif
    test_is_literal_type<double>();
    test_is_literal_type<char[3]>();
    test_is_literal_type<char[]>();
    test_is_literal_type<Empty>();
    test_is_literal_type<bit_zero>();
    test_is_literal_type<Union>();
    test_is_literal_type<Enum>();
    test_is_literal_type<FunctionPtr>();

    test_is_not_literal_type<NotEmpty>();
    test_is_not_literal_type<Abstract>();

  return 0;
}
