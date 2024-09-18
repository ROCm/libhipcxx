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

// UNSUPPORTED: c++98, c++03, c++11, nvrtc
// UNSUPPORTED: clang-3, clang-4, clang-5, apple-clang, gcc-5, gcc-6, msvc-19

// type_traits

// has_unique_object_representations

#include <hip/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__
void test_has_unique_object_representations()
{
    static_assert( hip::std::has_unique_object_representations<T>::value, "");
    static_assert( hip::std::has_unique_object_representations<const T>::value, "");
    static_assert( hip::std::has_unique_object_representations<volatile T>::value, "");
    static_assert( hip::std::has_unique_object_representations<const volatile T>::value, "");

    static_assert( hip::std::has_unique_object_representations_v<T>, "");
    static_assert( hip::std::has_unique_object_representations_v<const T>, "");
    static_assert( hip::std::has_unique_object_representations_v<volatile T>, "");
    static_assert( hip::std::has_unique_object_representations_v<const volatile T>, "");
}

template <class T>
__host__ __device__
void test_has_not_has_unique_object_representations()
{
    static_assert(!hip::std::has_unique_object_representations<T>::value, "");
    static_assert(!hip::std::has_unique_object_representations<const T>::value, "");
    static_assert(!hip::std::has_unique_object_representations<volatile T>::value, "");
    static_assert(!hip::std::has_unique_object_representations<const volatile T>::value, "");

    static_assert(!hip::std::has_unique_object_representations_v<T>, "");
    static_assert(!hip::std::has_unique_object_representations_v<const T>, "");
    static_assert(!hip::std::has_unique_object_representations_v<volatile T>, "");
    static_assert(!hip::std::has_unique_object_representations_v<const volatile T>, "");
}

class Empty
{
};

class NotEmpty
{
    __host__ __device__
    virtual ~NotEmpty();
};

union EmptyUnion {};
struct NonEmptyUnion {int x; unsigned y;};

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
    unsigned foo;
};

struct B
{
   char bar;
   int foo;
};


int main(int, char**)
{
    test_has_not_has_unique_object_representations<void>();
    test_has_not_has_unique_object_representations<Empty>();
    test_has_not_has_unique_object_representations<EmptyUnion>();
    test_has_not_has_unique_object_representations<NotEmpty>();
    test_has_not_has_unique_object_representations<bit_zero>();
    test_has_not_has_unique_object_representations<Abstract>();
    test_has_not_has_unique_object_representations<B>();

//  I would expect all three of these to have unique representations.
//  I would also expect that there are systems where they do not.
//     test_has_not_has_unique_object_representations<int&>();
//     test_has_not_has_unique_object_representations<int *>();
//     test_has_not_has_unique_object_representations<double>();


    test_has_unique_object_representations<unsigned>();
    test_has_unique_object_representations<NonEmptyUnion>();
    test_has_unique_object_representations<char[3]>();
    test_has_unique_object_representations<char[]>();


  return 0;
}
