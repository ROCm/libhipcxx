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

// function

#include <hip/std/type_traits>
#include "test_macros.h"

using namespace std;

class Class {};

enum Enum1 {};
#if TEST_STD_VER >= 11
enum class Enum2 : int {};
#else
enum Enum2 {};
#endif

template <class T>
__host__ __device__
void test()
{
    static_assert(!hip::std::is_void<T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_null_pointer<T>::value, "");
#endif
    static_assert(!hip::std::is_integral<T>::value, "");
    static_assert(!hip::std::is_floating_point<T>::value, "");
    static_assert(!hip::std::is_array<T>::value, "");
    static_assert(!hip::std::is_pointer<T>::value, "");
    static_assert(!hip::std::is_lvalue_reference<T>::value, "");
    static_assert(!hip::std::is_rvalue_reference<T>::value, "");
    static_assert(!hip::std::is_member_object_pointer<T>::value, "");
    static_assert(!hip::std::is_member_function_pointer<T>::value, "");
    static_assert(!hip::std::is_enum<T>::value, "");
    static_assert(!hip::std::is_union<T>::value, "");
    static_assert(!hip::std::is_class<T>::value, "");
    static_assert( hip::std::is_function<T>::value, "");
}

// Since we can't actually add the const volatile and ref qualifiers once
// later let's use a macro to do it.
#define TEST_REGULAR(...)                 \
    test<__VA_ARGS__>();                  \
    test<__VA_ARGS__ const>();            \
    test<__VA_ARGS__ volatile>();         \
    test<__VA_ARGS__ const volatile>()


#define TEST_REF_QUALIFIED(...)           \
    test<__VA_ARGS__ &>();                \
    test<__VA_ARGS__ const &>();          \
    test<__VA_ARGS__ volatile &>();       \
    test<__VA_ARGS__ const volatile &>(); \
    test<__VA_ARGS__ &&>();               \
    test<__VA_ARGS__ const &&>();         \
    test<__VA_ARGS__ volatile &&>();      \
    test<__VA_ARGS__ const volatile &&>()

struct incomplete_type;

int main(int, char**)
{
    TEST_REGULAR( void () );
    TEST_REGULAR( void (int) );
    TEST_REGULAR( int (double) );
    TEST_REGULAR( int (double, char) );
    TEST_REGULAR( void (...) );
    TEST_REGULAR( void (int, ...) );
    TEST_REGULAR( int (double, ...) );
    TEST_REGULAR( int (double, char, ...) );
#if TEST_STD_VER >= 11
    TEST_REF_QUALIFIED( void () );
    TEST_REF_QUALIFIED( void (int) );
    TEST_REF_QUALIFIED( int (double) );
    TEST_REF_QUALIFIED( int (double, char) );
    TEST_REF_QUALIFIED( void (...) );
    TEST_REF_QUALIFIED( void (int, ...) );
    TEST_REF_QUALIFIED( int (double, ...) );
    TEST_REF_QUALIFIED( int (double, char, ...) );
#endif

//  LWG#2582
    static_assert(!hip::std::is_function<incomplete_type>::value, "");

  return 0;
}
