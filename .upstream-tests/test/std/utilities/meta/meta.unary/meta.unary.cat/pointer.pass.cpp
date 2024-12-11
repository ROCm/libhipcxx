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

// pointer

#include <hip/std/type_traits>
#include "test_macros.h"


template <class T>
__host__ __device__
void test_pointer_imp()
{
    static_assert(!hip::std::is_void<T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_null_pointer<T>::value, "");
#endif
    static_assert(!hip::std::is_integral<T>::value, "");
    static_assert(!hip::std::is_floating_point<T>::value, "");
    static_assert(!hip::std::is_array<T>::value, "");
    static_assert( hip::std::is_pointer<T>::value, "");
    static_assert(!hip::std::is_lvalue_reference<T>::value, "");
    static_assert(!hip::std::is_rvalue_reference<T>::value, "");
    static_assert(!hip::std::is_member_object_pointer<T>::value, "");
    static_assert(!hip::std::is_member_function_pointer<T>::value, "");
    static_assert(!hip::std::is_enum<T>::value, "");
    static_assert(!hip::std::is_union<T>::value, "");
    static_assert(!hip::std::is_class<T>::value, "");
    static_assert(!hip::std::is_function<T>::value, "");
}

template <class T>
__host__ __device__
void test_pointer()
{
    test_pointer_imp<T>();
    test_pointer_imp<const T>();
    test_pointer_imp<volatile T>();
    test_pointer_imp<const volatile T>();
}

struct incomplete_type;

int main(int, char**)
{
    test_pointer<void*>();
    test_pointer<int*>();
    test_pointer<const int*>();
    test_pointer<void (*)(int)>();

//  LWG#2582
    static_assert(!hip::std::is_pointer<incomplete_type>::value, "");

  return 0;
}
