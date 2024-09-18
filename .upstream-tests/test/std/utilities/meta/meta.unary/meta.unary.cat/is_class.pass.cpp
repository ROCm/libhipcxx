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

// is_class

#include <hip/std/type_traits>
#include <hip/std/cstddef>        // for hip::std::nullptr_t
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_class()
{
    static_assert( hip::std::is_class<T>::value, "");
    static_assert( hip::std::is_class<const T>::value, "");
    static_assert( hip::std::is_class<volatile T>::value, "");
    static_assert( hip::std::is_class<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_class_v<T>, "");
    static_assert( hip::std::is_class_v<const T>, "");
    static_assert( hip::std::is_class_v<volatile T>, "");
    static_assert( hip::std::is_class_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_class()
{
    static_assert(!hip::std::is_class<T>::value, "");
    static_assert(!hip::std::is_class<const T>::value, "");
    static_assert(!hip::std::is_class<volatile T>::value, "");
    static_assert(!hip::std::is_class<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_class_v<T>, "");
    static_assert(!hip::std::is_class_v<const T>, "");
    static_assert(!hip::std::is_class_v<volatile T>, "");
    static_assert(!hip::std::is_class_v<const volatile T>, "");
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
struct incomplete_type;

typedef void (*FunctionPtr)();

int main(int, char**)
{
    test_is_class<Empty>();
    test_is_class<bit_zero>();
    test_is_class<NotEmpty>();
    test_is_class<Abstract>();
    test_is_class<incomplete_type>();

#if TEST_STD_VER >= 11
// In C++03 we have an emulation of hip::std::nullptr_t
    test_is_not_class<hip::std::nullptr_t>();
#endif
    test_is_not_class<void>();
    test_is_not_class<int>();
    test_is_not_class<int&>();
#if TEST_STD_VER >= 11
    test_is_not_class<int&&>();
#endif
    test_is_not_class<int*>();
    test_is_not_class<double>();
    test_is_not_class<const int*>();
    test_is_not_class<char[3]>();
    test_is_not_class<char[]>();
    test_is_not_class<Enum>();
    test_is_not_class<Union>();
    test_is_not_class<FunctionPtr>();

  return 0;
}
