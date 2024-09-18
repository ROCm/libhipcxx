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

// is_nothrow_default_constructible

#include <hip/std/type_traits>
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_nothrow_default_constructible()
{
    static_assert( hip::std::is_nothrow_default_constructible<T>::value, "");
    static_assert( hip::std::is_nothrow_default_constructible<const T>::value, "");
    static_assert( hip::std::is_nothrow_default_constructible<volatile T>::value, "");
    static_assert( hip::std::is_nothrow_default_constructible<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_nothrow_default_constructible_v<T>, "");
    static_assert( hip::std::is_nothrow_default_constructible_v<const T>, "");
    static_assert( hip::std::is_nothrow_default_constructible_v<volatile T>, "");
    static_assert( hip::std::is_nothrow_default_constructible_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_has_not_nothrow_default_constructor()
{
    static_assert(!hip::std::is_nothrow_default_constructible<T>::value, "");
    static_assert(!hip::std::is_nothrow_default_constructible<const T>::value, "");
    static_assert(!hip::std::is_nothrow_default_constructible<volatile T>::value, "");
    static_assert(!hip::std::is_nothrow_default_constructible<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_nothrow_default_constructible_v<T>, "");
    static_assert(!hip::std::is_nothrow_default_constructible_v<const T>, "");
    static_assert(!hip::std::is_nothrow_default_constructible_v<volatile T>, "");
    static_assert(!hip::std::is_nothrow_default_constructible_v<const volatile T>, "");
#endif
}

class Empty
{
};

union Union {};

struct bit_zero
{
    int :  0;
};

struct A
{
    __host__ __device__
    A();
};

#if TEST_STD_VER >= 11
struct DThrows
{
    __host__ __device__
    DThrows()  noexcept(true) {}
    __host__ __device__
    ~DThrows() noexcept(false) {}
};
#endif

int main(int, char**)
{
    test_has_not_nothrow_default_constructor<void>();
    test_has_not_nothrow_default_constructor<int&>();
    test_has_not_nothrow_default_constructor<A>();
#if TEST_STD_VER >= 11 && !defined(__INTEL_COMPILER) && (!defined(TEST_COMPILER_C1XX) || 1920 <= _MSC_VER)
    test_has_not_nothrow_default_constructor<DThrows>(); // This is LWG2116
#endif

    test_is_nothrow_default_constructible<Union>();
    test_is_nothrow_default_constructible<Empty>();
    test_is_nothrow_default_constructible<int>();
    test_is_nothrow_default_constructible<double>();
    test_is_nothrow_default_constructible<int*>();
    test_is_nothrow_default_constructible<const int*>();
    test_is_nothrow_default_constructible<char[3]>();
    test_is_nothrow_default_constructible<bit_zero>();

  return 0;
}
