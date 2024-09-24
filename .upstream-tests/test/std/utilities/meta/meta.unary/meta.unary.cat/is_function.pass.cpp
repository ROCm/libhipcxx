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

// is_function

#include <hip/std/type_traits>
#include <hip/std/cstddef>        // for hip::std::nullptr_t

#include "test_macros.h"

// NOTE: On Windows the function `test_is_function<void()>` and
// `test_is_function<void() noexcept> has the same mangled despite being
// a distinct instantiation. This causes Clang to emit an error. However
// structs do not have this problem.

template <class T>
struct test_is_function {
    static_assert( hip::std::is_function<T>::value, "");
    static_assert( hip::std::is_function<const T>::value, "");
    static_assert( hip::std::is_function<volatile T>::value, "");
    static_assert( hip::std::is_function<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::is_function_v<T>, "");
    static_assert( hip::std::is_function_v<const T>, "");
    static_assert( hip::std::is_function_v<volatile T>, "");
    static_assert( hip::std::is_function_v<const volatile T>, "");
#endif
};

template <class T>
struct test_is_not_function {
    static_assert(!hip::std::is_function<T>::value, "");
    static_assert(!hip::std::is_function<const T>::value, "");
    static_assert(!hip::std::is_function<volatile T>::value, "");
    static_assert(!hip::std::is_function<const volatile T>::value, "");
#if TEST_STD_VER > 11
    static_assert(!hip::std::is_function_v<T>, "");
    static_assert(!hip::std::is_function_v<const T>, "");
    static_assert(!hip::std::is_function_v<volatile T>, "");
    static_assert(!hip::std::is_function_v<const volatile T>, "");
#endif
};

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
struct incomplete_type;

typedef void (*FunctionPtr)();

int main(int, char**)
{
    test_is_function<void(void)>();
    test_is_function<int(int)>();
    test_is_function<int(int, double)>();
    test_is_function<int(Abstract *)>();
    test_is_function<void(...)>();

  test_is_not_function<hip::std::nullptr_t>();
  test_is_not_function<void>();
  test_is_not_function<int>();
  test_is_not_function<int&>();
  test_is_not_function<int&&>();
  test_is_not_function<int*>();
  test_is_not_function<double>();
  test_is_not_function<char[3]>();
  test_is_not_function<char[]>();
  test_is_not_function<Union>();
  test_is_not_function<Enum>();
  test_is_not_function<FunctionPtr>(); // function pointer is not a function
  test_is_not_function<Empty>();
  test_is_not_function<bit_zero>();
  test_is_not_function<NotEmpty>();
  test_is_not_function<Abstract>();
  test_is_not_function<Abstract*>();
  test_is_not_function<incomplete_type>();

#if TEST_STD_VER >= 11
  test_is_function<void() noexcept>();
  test_is_function<void() const && noexcept>();
#endif

  return 0;
}
