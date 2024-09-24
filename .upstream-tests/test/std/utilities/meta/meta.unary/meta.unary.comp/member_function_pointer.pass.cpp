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

// member_function_pointer

#include <hip/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__
void test_member_function_pointer_imp()
{
    static_assert(!hip::std::is_reference<T>::value, "");
    static_assert(!hip::std::is_arithmetic<T>::value, "");
    static_assert(!hip::std::is_fundamental<T>::value, "");
    static_assert( hip::std::is_object<T>::value, "");
    static_assert( hip::std::is_scalar<T>::value, "");
    static_assert( hip::std::is_compound<T>::value, "");
    static_assert( hip::std::is_member_pointer<T>::value, "");
}

template <class T>
__host__ __device__
void test_member_function_pointer()
{
    test_member_function_pointer_imp<T>();
    test_member_function_pointer_imp<const T>();
    test_member_function_pointer_imp<volatile T>();
    test_member_function_pointer_imp<const volatile T>();
}

class Class
{
};

int main(int, char**)
{
    test_member_function_pointer<void (Class::*)()>();
    test_member_function_pointer<void (Class::*)(int)>();
    test_member_function_pointer<void (Class::*)(int, char)>();

  return 0;
}
