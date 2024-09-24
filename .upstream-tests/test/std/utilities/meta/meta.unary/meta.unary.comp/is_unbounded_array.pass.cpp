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

// type_traits

// is_unbounded_array<T>
// T is an array type of unknown bound ([dcl.array])

#include <hip/std/type_traits>

#include "test_macros.h"

template <class T, bool B>
__host__ __device__
void test_array_imp()
{
    static_assert( B == hip::std::is_unbounded_array<T>::value, "" );
    static_assert( B == hip::std::is_unbounded_array_v<T>, "" );
}

template <class T, bool B>
__host__ __device__
void test_array()
{
    test_array_imp<T, B>();
    test_array_imp<const T, B>();
    test_array_imp<volatile T, B>();
    test_array_imp<const volatile T, B>();
}

typedef char array[3];
typedef char incomplete_array[];

class incomplete_type;

class Empty {};
union Union {};

class Abstract
{
    virtual ~Abstract() = 0;
};

enum Enum {zero, one};
typedef void (*FunctionPtr)();

int main(int, char**)
{
//  Non-array types
    test_array<void,           false>();
    test_array<hip::std::nullptr_t, false>();
    test_array<int,            false>();
    test_array<double,         false>();
    test_array<void *,         false>();
    test_array<int &,          false>();
    test_array<int &&,         false>();
    test_array<Empty,          false>();
    test_array<Union,          false>();
    test_array<Abstract,       false>();
    test_array<Enum,           false>();
    test_array<FunctionPtr,    false>();

//  Array types
    test_array<array,             false>();
    test_array<incomplete_array,  true>();
    test_array<incomplete_type[], true>();

  return 0;
}
