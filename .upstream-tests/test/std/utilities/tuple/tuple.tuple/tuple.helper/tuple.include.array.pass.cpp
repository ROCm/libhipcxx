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
//
//  LWG #2212 says that tuple_size and tuple_element must be
//     available after including <utility>

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
// hip::std::array not supported
//#include <hip/std/array>
#include <hip/std/type_traits>

#include "test_macros.h"

// hip::std::array not supported
/*
template <class T, hip::std::size_t N, class U, size_t idx>
__host__ __device__ void test()
{
    static_assert((hip::std::is_base_of<hip::std::integral_constant<hip::std::size_t, N>,
                                   hip::std::tuple_size<T> >::value), "");
    static_assert((hip::std::is_base_of<hip::std::integral_constant<hip::std::size_t, N>,
                                   hip::std::tuple_size<const T> >::value), "");
    static_assert((hip::std::is_base_of<hip::std::integral_constant<hip::std::size_t, N>,
                                   hip::std::tuple_size<volatile T> >::value), "");
    static_assert((hip::std::is_base_of<hip::std::integral_constant<hip::std::size_t, N>,
                                   hip::std::tuple_size<const volatile T> >::value), "");
    static_assert((hip::std::is_same<typename hip::std::tuple_element<idx, T>::type, U>::value), "");
    static_assert((hip::std::is_same<typename hip::std::tuple_element<idx, const T>::type, const U>::value), "");
    static_assert((hip::std::is_same<typename hip::std::tuple_element<idx, volatile T>::type, volatile U>::value), "");
    static_assert((hip::std::is_same<typename hip::std::tuple_element<idx, const volatile T>::type, const volatile U>::value), "");
}
*/

int main(int, char**)
{
    // hip::std::array not supported
    /*
    test<hip::std::array<int, 5>, 5, int, 0>();
    test<hip::std::array<int, 5>, 5, int, 1>();
    test<hip::std::array<const char *, 4>, 4, const char *, 3>();
    test<hip::std::array<volatile int, 4>, 4, volatile int, 3>();
    test<hip::std::array<char *, 3>, 3, char *, 1>();
    test<hip::std::array<char *, 3>, 3, char *, 2>();
    */

  return 0;
}
