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

// <cuda/std/array>

// tuple_size<array<T, N> >::value

#include <hip/std/array>

#include "test_macros.h"

template <class T, hip::std::size_t N>
__host__ __device__
void test()
{
    {
    typedef hip::std::array<T, N> C;
    static_assert((hip::std::tuple_size<C>::value == N), "");
    }
    {
    typedef hip::std::array<T const, N> C;
    static_assert((hip::std::tuple_size<C>::value == N), "");
    }
    {
    typedef hip::std::array<T volatile, N> C;
    static_assert((hip::std::tuple_size<C>::value == N), "");
    }
    {
    typedef hip::std::array<T const volatile, N> C;
    static_assert((hip::std::tuple_size<C>::value == N), "");
    }
}

int main(int, char**)
{
    test<double, 0>();
    test<double, 3>();
    test<double, 5>();

  return 0;
}
