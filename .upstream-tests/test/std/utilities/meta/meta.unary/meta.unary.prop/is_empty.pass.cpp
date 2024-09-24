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

// is_empty

// T is a non-union class type with:
//  no non-static data members,
//  no unnamed bit-fields of non-zero length,
//  no virtual member functions,
//  no virtual base classes,
//  and no base class B for which is_empty_v<B> is false.


#include <hip/std/type_traits>
#include "test_macros.h"

template <class T>
__host__ __device__
void test_is_empty()
{
    static_assert( hip::std::is_empty<T>::value, "");
    static_assert( hip::std::is_empty<const T>::value, "");
    static_assert( hip::std::is_empty<volatile T>::value, "");
    static_assert( hip::std::is_empty<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_empty_v<T>, "");
    static_assert( hip::std::is_empty_v<const T>, "");
    static_assert( hip::std::is_empty_v<volatile T>, "");
    static_assert( hip::std::is_empty_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__
void test_is_not_empty()
{
    static_assert(!hip::std::is_empty<T>::value, "");
    static_assert(!hip::std::is_empty<const T>::value, "");
    static_assert(!hip::std::is_empty<volatile T>::value, "");
    static_assert(!hip::std::is_empty<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_empty_v<T>, "");
    static_assert(!hip::std::is_empty_v<const T>, "");
    static_assert(!hip::std::is_empty_v<volatile T>, "");
    static_assert(!hip::std::is_empty_v<const volatile T>, "");
#endif
}

class Empty {};
struct NotEmpty { int foo; };

class VirtualFn
{
    __host__ __device__
    virtual ~VirtualFn();
};

union Union {};

struct EmptyBase    : public Empty {};
struct VirtualBase  : virtual Empty {};
struct NotEmptyBase : public NotEmpty {};

struct StaticMember    { static const int foo; };
struct NonStaticMember {        int foo; };

struct bit_zero
{
    int :  0;
};

struct bit_one
{
    int :  1;
};

int main(int, char**)
{
    test_is_not_empty<void>();
    test_is_not_empty<int&>();
    test_is_not_empty<int>();
    test_is_not_empty<double>();
    test_is_not_empty<int*>();
    test_is_not_empty<const int*>();
    test_is_not_empty<char[3]>();
    test_is_not_empty<char[]>();
    test_is_not_empty<Union>();
    test_is_not_empty<NotEmpty>();
    test_is_not_empty<VirtualFn>();
    test_is_not_empty<VirtualBase>();
    test_is_not_empty<NotEmptyBase>();
    test_is_not_empty<NonStaticMember>();
//    test_is_not_empty<bit_one>();

    test_is_empty<Empty>();
    test_is_empty<EmptyBase>();
    test_is_empty<StaticMember>();
    test_is_empty<bit_zero>();

  return 0;
}
