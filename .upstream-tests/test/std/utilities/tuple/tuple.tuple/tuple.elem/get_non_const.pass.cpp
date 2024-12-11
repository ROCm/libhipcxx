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
//   typename tuple_element<I, tuple<Types...> >::type&
//   get(tuple<Types...>& t);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
// hip::std::string not supported
//#include <hip/std/string>
#include <hip/std/cassert>

#include "test_macros.h"

#if TEST_STD_VER > 11

struct Empty {};

struct S {
   hip::std::tuple<int, Empty> a;
   int k;
   Empty e;
   __host__ __device__ constexpr S() : a{1,Empty{}}, k(hip::std::get<0>(a)), e(hip::std::get<1>(a)) {}
   };

__host__ __device__ constexpr hip::std::tuple<int, int> getP () { return { 3, 4 }; }
#endif

int main(int, char**)
{
    {
        typedef hip::std::tuple<int> T;
        T t(3);
        assert(hip::std::get<0>(t) == 3);
        hip::std::get<0>(t) = 2;
        assert(hip::std::get<0>(t) == 2);
    }
    // hip::std::string not supported
    /*
    {
        typedef hip::std::tuple<hip::std::string, int> T;
        T t("high", 5);
        assert(hip::std::get<0>(t) == "high");
        assert(hip::std::get<1>(t) == 5);
        hip::std::get<0>(t) = "four";
        hip::std::get<1>(t) = 4;
        assert(hip::std::get<0>(t) == "four");
        assert(hip::std::get<1>(t) == 4);
    }
    {
        typedef hip::std::tuple<double&, hip::std::string, int> T;
        double d = 1.5;
        T t(d, "high", 5);
        assert(hip::std::get<0>(t) == 1.5);
        assert(hip::std::get<1>(t) == "high");
        assert(hip::std::get<2>(t) == 5);
        hip::std::get<0>(t) = 2.5;
        hip::std::get<1>(t) = "four";
        hip::std::get<2>(t) = 4;
        assert(hip::std::get<0>(t) == 2.5);
        assert(hip::std::get<1>(t) == "four");
        assert(hip::std::get<2>(t) == 4);
        assert(d == 2.5);
    }
    */
#if TEST_STD_VER > 11
    { // get on an rvalue tuple
        static_assert ( hip::std::get<0> ( hip::std::make_tuple ( 0.0f, 1, 2.0, 3L )) == 0, "" );
        static_assert ( hip::std::get<1> ( hip::std::make_tuple ( 0.0f, 1, 2.0, 3L )) == 1, "" );
        static_assert ( hip::std::get<2> ( hip::std::make_tuple ( 0.0f, 1, 2.0, 3L )) == 2, "" );
        static_assert ( hip::std::get<3> ( hip::std::make_tuple ( 0.0f, 1, 2.0, 3L )) == 3, "" );
        static_assert(S().k == 1, "");
        static_assert(hip::std::get<1>(getP()) == 4, "");
    }
#endif


  return 0;
}
