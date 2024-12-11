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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
// struct tuple_element<I, tuple<Types...> >
// {
//     typedef Ti type;
// };

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/type_traits>

#include "test_macros.h"

template <class T, hip::std::size_t N, class U>
__host__ __device__ void test()
{
    static_assert((hip::std::is_same<typename hip::std::tuple_element<N, T>::type, U>::value), "");
    static_assert((hip::std::is_same<typename hip::std::tuple_element<N, const T>::type, const U>::value), "");
    static_assert((hip::std::is_same<typename hip::std::tuple_element<N, volatile T>::type, volatile U>::value), "");
    static_assert((hip::std::is_same<typename hip::std::tuple_element<N, const volatile T>::type, const volatile U>::value), "");
#if TEST_STD_VER > 11
    static_assert((hip::std::is_same<typename hip::std::tuple_element_t<N, T>, U>::value), "");
    static_assert((hip::std::is_same<typename hip::std::tuple_element_t<N, const T>, const U>::value), "");
    static_assert((hip::std::is_same<typename hip::std::tuple_element_t<N, volatile T>, volatile U>::value), "");
    static_assert((hip::std::is_same<typename hip::std::tuple_element_t<N, const volatile T>, const volatile U>::value), "");
#endif
}

int main(int, char**)
{
    test<hip::std::tuple<int>, 0, int>();
    test<hip::std::tuple<char, int>, 0, char>();
    test<hip::std::tuple<char, int>, 1, int>();
    test<hip::std::tuple<int*, char, int>, 0, int*>();
    test<hip::std::tuple<int*, char, int>, 1, char>();
    test<hip::std::tuple<int*, char, int>, 2, int>();

  return 0;
}
