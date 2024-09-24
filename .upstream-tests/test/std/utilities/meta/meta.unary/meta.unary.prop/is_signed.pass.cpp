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

// is_signed

#include <hip/std/type_traits>
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_signed()
{
    static_assert( hip::std::is_signed<T>::value, "");
    static_assert( hip::std::is_signed<const T>::value, "");
    static_assert( hip::std::is_signed<volatile T>::value, "");
    static_assert( hip::std::is_signed<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_signed_v<T>, "");
    static_assert( hip::std::is_signed_v<const T>, "");
    static_assert( hip::std::is_signed_v<volatile T>, "");
    static_assert( hip::std::is_signed_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_signed()
{
    static_assert(!hip::std::is_signed<T>::value, "");
    static_assert(!hip::std::is_signed<const T>::value, "");
    static_assert(!hip::std::is_signed<volatile T>::value, "");
    static_assert(!hip::std::is_signed<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_signed_v<T>, "");
    static_assert(!hip::std::is_signed_v<const T>, "");
    static_assert(!hip::std::is_signed_v<volatile T>, "");
    static_assert(!hip::std::is_signed_v<const volatile T>, "");
#endif
}

class Class
{
public:
    __host__ __device__
    ~Class();
};

struct A; // incomplete

int main(int, char**)
{
    test_is_not_signed<void>();
    test_is_not_signed<int&>();
    test_is_not_signed<Class>();
    test_is_not_signed<int*>();
    test_is_not_signed<const int*>();
    test_is_not_signed<char[3]>();
    test_is_not_signed<char[]>();
    test_is_not_signed<bool>();
    test_is_not_signed<unsigned>();
    test_is_not_signed<A>();

    test_is_signed<int>();
    test_is_signed<double>();

#ifndef _LIBCUDACXX_HAS_NO_INT128
    test_is_signed<__int128_t>();
    test_is_not_signed<__uint128_t>();
#endif

  return 0;
}
