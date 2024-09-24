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

// is_trivially_copy_assignable

// XFAIL: gcc-4.8, gcc-4.9

#include <hip/std/type_traits>
#include "test_macros.h"

template <class T>
__host__ __device__
void test_has_trivially_copy_assignable()
{
    static_assert( hip::std::is_trivially_copy_assignable<T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_trivially_copy_assignable_v<T>, "");
#endif
}

template <class T>
__host__ __device__
void test_has_not_trivially_copy_assignable()
{
    static_assert(!hip::std::is_trivially_copy_assignable<T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_trivially_copy_assignable_v<T>, "");
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
    A& operator=(const A&);
};

int main(int, char**)
{
    test_has_trivially_copy_assignable<int&>();
    test_has_trivially_copy_assignable<Union>();
    test_has_trivially_copy_assignable<Empty>();
    test_has_trivially_copy_assignable<int>();
    test_has_trivially_copy_assignable<double>();
    test_has_trivially_copy_assignable<int*>();
    test_has_trivially_copy_assignable<const int*>();
    test_has_trivially_copy_assignable<bit_zero>();

    test_has_not_trivially_copy_assignable<void>();
    test_has_not_trivially_copy_assignable<A>();
    test_has_not_trivially_copy_assignable<NotEmpty>();
    test_has_not_trivially_copy_assignable<Abstract>();
    test_has_not_trivially_copy_assignable<const Empty>();


  return 0;
}
