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

// is_fundamental

#include <hip/std/type_traits>
#include <hip/std/cstddef>         // for hip::std::nullptr_t
#include "test_macros.h"

#pragma nv_diag_suppress cuda_demote_unsupported_floating_point

template <class T>
__host__ __device__
void test_is_fundamental()
{
    static_assert( hip::std::is_fundamental<T>::value, "");
    static_assert( hip::std::is_fundamental<const T>::value, "");
    static_assert( hip::std::is_fundamental<volatile T>::value, "");
    static_assert( hip::std::is_fundamental<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_fundamental_v<T>, "");
    static_assert( hip::std::is_fundamental_v<const T>, "");
    static_assert( hip::std::is_fundamental_v<volatile T>, "");
    static_assert( hip::std::is_fundamental_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_fundamental()
{
    static_assert(!hip::std::is_fundamental<T>::value, "");
    static_assert(!hip::std::is_fundamental<const T>::value, "");
    static_assert(!hip::std::is_fundamental<volatile T>::value, "");
    static_assert(!hip::std::is_fundamental<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_fundamental_v<T>, "");
    static_assert(!hip::std::is_fundamental_v<const T>, "");
    static_assert(!hip::std::is_fundamental_v<volatile T>, "");
    static_assert(!hip::std::is_fundamental_v<const volatile T>, "");
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
    test_is_fundamental<hip::std::nullptr_t>();
    test_is_fundamental<void>();
    test_is_fundamental<short>();
    test_is_fundamental<unsigned short>();
    test_is_fundamental<int>();
    test_is_fundamental<unsigned int>();
    test_is_fundamental<long>();
    test_is_fundamental<unsigned long>();
    test_is_fundamental<long long>();
    test_is_fundamental<unsigned long long>();
    test_is_fundamental<bool>();
    test_is_fundamental<char>();
    test_is_fundamental<signed char>();
    test_is_fundamental<unsigned char>();
    test_is_fundamental<wchar_t>();
    test_is_fundamental<double>();
    test_is_fundamental<float>();
    test_is_fundamental<double>();
    test_is_fundamental<long double>();
    test_is_fundamental<char16_t>();
    test_is_fundamental<char32_t>();

    test_is_not_fundamental<char[3]>();
    test_is_not_fundamental<char[]>();
    test_is_not_fundamental<void *>();
    test_is_not_fundamental<FunctionPtr>();
    test_is_not_fundamental<int&>();
    test_is_not_fundamental<int&&>();
    test_is_not_fundamental<Union>();
    test_is_not_fundamental<Empty>();
    test_is_not_fundamental<incomplete_type>();
    test_is_not_fundamental<bit_zero>();
    test_is_not_fundamental<int*>();
    test_is_not_fundamental<const int*>();
    test_is_not_fundamental<Enum>();
    test_is_not_fundamental<NotEmpty>();
    test_is_not_fundamental<Abstract>();

  return 0;
}
