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

// template<class... Types>
//     tuple<Types&&...> forward_as_tuple(Types&&... t);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

template <class Tuple>
__host__ __device__ void test0(const Tuple&)
{
    static_assert(hip::std::tuple_size<Tuple>::value == 0, "");
}

template <class Tuple>
__host__ __device__ void test1a(const Tuple& t)
{
    static_assert(hip::std::tuple_size<Tuple>::value == 1, "");
    static_assert(hip::std::is_same<typename hip::std::tuple_element<0, Tuple>::type, int&&>::value, "");
    assert(hip::std::get<0>(t) == 1);
}

template <class Tuple>
__host__ __device__ void test1b(const Tuple& t)
{
    static_assert(hip::std::tuple_size<Tuple>::value == 1, "");
    static_assert(hip::std::is_same<typename hip::std::tuple_element<0, Tuple>::type, int&>::value, "");
    assert(hip::std::get<0>(t) == 2);
}

template <class Tuple>
__host__ __device__ void test2a(const Tuple& t)
{
    static_assert(hip::std::tuple_size<Tuple>::value == 2, "");
    static_assert(hip::std::is_same<typename hip::std::tuple_element<0, Tuple>::type, double&>::value, "");
    static_assert(hip::std::is_same<typename hip::std::tuple_element<1, Tuple>::type, char&>::value, "");
    assert(hip::std::get<0>(t) == 2.5);
    assert(hip::std::get<1>(t) == 'a');
}

#if TEST_STD_VER > 11
template <class Tuple>
__host__ __device__ constexpr int test3(const Tuple&)
{
    return hip::std::tuple_size<Tuple>::value;
}
#endif

int main(int, char**)
{
    {
        test0(hip::std::forward_as_tuple());
    }
#if !(defined(_MSC_VER) && _MSC_VER < 1916)
    {
        test1a(hip::std::forward_as_tuple(1));
    }
#endif
    {
        int i = 2;
        test1b(hip::std::forward_as_tuple(i));
    }
    {
        double i = 2.5;
        char c = 'a';
        test2a(hip::std::forward_as_tuple(i, c));
#if TEST_STD_VER > 11
        static_assert ( test3 (hip::std::forward_as_tuple(i, c)) == 2, "" );
#endif
    }

  return 0;
}
