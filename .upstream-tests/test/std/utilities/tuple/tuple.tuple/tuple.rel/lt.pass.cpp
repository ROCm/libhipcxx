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

// template<class... TTypes, class... UTypes>
//   bool
//   operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<class... TTypes, class... UTypes>
//   bool
//   operator>(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<class... TTypes, class... UTypes>
//   bool
//   operator<=(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<class... TTypes, class... UTypes>
//   bool
//   operator>=(const tuple<TTypes...>& t, const tuple<UTypes...>& u);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
// hip::std::string not supported
//#include <hip/std/string>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef hip::std::tuple<> T1;
        typedef hip::std::tuple<> T2;
        const T1 t1;
        const T2 t2;
        assert(!(t1 <  t2));
        assert( (t1 <= t2));
        assert(!(t1 >  t2));
        assert( (t1 >= t2));
    }
    {
        typedef hip::std::tuple<long> T1;
        typedef hip::std::tuple<double> T2;
        const T1 t1(1);
        const T2 t2(1);
        assert(!(t1 <  t2));
        assert( (t1 <= t2));
        assert(!(t1 >  t2));
        assert( (t1 >= t2));
    }
    {
        typedef hip::std::tuple<long> T1;
        typedef hip::std::tuple<double> T2;
        const T1 t1(1);
        const T2 t2(0.9);
        assert(!(t1 <  t2));
        assert(!(t1 <= t2));
        assert( (t1 >  t2));
        assert( (t1 >= t2));
    }
    {
        typedef hip::std::tuple<long> T1;
        typedef hip::std::tuple<double> T2;
        const T1 t1(1);
        const T2 t2(1.1);
        assert( (t1 <  t2));
        assert( (t1 <= t2));
        assert(!(t1 >  t2));
        assert(!(t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int> T1;
        typedef hip::std::tuple<double, long> T2;
        const T1 t1(1, 2);
        const T2 t2(1, 2);
        assert(!(t1 <  t2));
        assert( (t1 <= t2));
        assert(!(t1 >  t2));
        assert( (t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int> T1;
        typedef hip::std::tuple<double, long> T2;
        const T1 t1(1, 2);
        const T2 t2(0.9, 2);
        assert(!(t1 <  t2));
        assert(!(t1 <= t2));
        assert( (t1 >  t2));
        assert( (t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int> T1;
        typedef hip::std::tuple<double, long> T2;
        const T1 t1(1, 2);
        const T2 t2(1.1, 2);
        assert( (t1 <  t2));
        assert( (t1 <= t2));
        assert(!(t1 >  t2));
        assert(!(t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int> T1;
        typedef hip::std::tuple<double, long> T2;
        const T1 t1(1, 2);
        const T2 t2(1, 1);
        assert(!(t1 <  t2));
        assert(!(t1 <= t2));
        assert( (t1 >  t2));
        assert( (t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int> T1;
        typedef hip::std::tuple<double, long> T2;
        const T1 t1(1, 2);
        const T2 t2(1, 3);
        assert( (t1 <  t2));
        assert( (t1 <= t2));
        assert(!(t1 >  t2));
        assert(!(t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int, double> T1;
        typedef hip::std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1, 2, 3);
        assert(!(t1 <  t2));
        assert( (t1 <= t2));
        assert(!(t1 >  t2));
        assert( (t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int, double> T1;
        typedef hip::std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(0.9, 2, 3);
        assert(!(t1 <  t2));
        assert(!(t1 <= t2));
        assert( (t1 >  t2));
        assert( (t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int, double> T1;
        typedef hip::std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1.1, 2, 3);
        assert( (t1 <  t2));
        assert( (t1 <= t2));
        assert(!(t1 >  t2));
        assert(!(t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int, double> T1;
        typedef hip::std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1, 1, 3);
        assert(!(t1 <  t2));
        assert(!(t1 <= t2));
        assert( (t1 >  t2));
        assert( (t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int, double> T1;
        typedef hip::std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1, 3, 3);
        assert( (t1 <  t2));
        assert( (t1 <= t2));
        assert(!(t1 >  t2));
        assert(!(t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int, double> T1;
        typedef hip::std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1, 2, 2);
        assert(!(t1 <  t2));
        assert(!(t1 <= t2));
        assert( (t1 >  t2));
        assert( (t1 >= t2));
    }
    {
        typedef hip::std::tuple<long, int, double> T1;
        typedef hip::std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1, 2, 4);
        assert( (t1 <  t2));
        assert( (t1 <= t2));
        assert(!(t1 >  t2));
        assert(!(t1 >= t2));
    }
#if TEST_STD_VER > 11
    {
        typedef hip::std::tuple<long, int, double> T1;
        typedef hip::std::tuple<double, long, int> T2;
        constexpr T1 t1(1, 2, 3);
        constexpr T2 t2(1, 2, 4);
        static_assert( (t1 <  t2), "");
        static_assert( (t1 <= t2), "");
        static_assert(!(t1 >  t2), "");
        static_assert(!(t1 >= t2), "");
    }
#endif

  return 0;
}
