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

// rank

#include <hip/std/type_traits>

#include "test_macros.h"

template <class T, unsigned A>
__host__ __device__
void test_rank()
{
    static_assert( hip::std::rank<T>::value == A, "");
    static_assert( hip::std::rank<const T>::value == A, "");
    static_assert( hip::std::rank<volatile T>::value == A, "");
    static_assert( hip::std::rank<const volatile T>::value == A, "");
#if TEST_STD_VER > 11
    static_assert( hip::std::rank_v<T> == A, "");
    static_assert( hip::std::rank_v<const T> == A, "");
    static_assert( hip::std::rank_v<volatile T> == A, "");
    static_assert( hip::std::rank_v<const volatile T> == A, "");
#endif
}

class Class
{
public:
    __host__ __device__
    ~Class();
};

int main(int, char**)
{
    test_rank<void, 0>();
    test_rank<int&, 0>();
    test_rank<Class, 0>();
    test_rank<int*, 0>();
    test_rank<const int*, 0>();
    test_rank<int, 0>();
    test_rank<double, 0>();
    test_rank<bool, 0>();
    test_rank<unsigned, 0>();

    test_rank<char[3], 1>();
    test_rank<char[][3], 2>();
    test_rank<char[][4][3], 3>();

  return 0;
}
