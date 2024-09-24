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

// template<class... Types>
//   tuple<VTypes...> make_tuple(Types&&... t);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>
#include <hip/std/functional>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        int i = 0;
        float j = 0;
        hip::std::tuple<int, int&, float&> t = hip::std::make_tuple(1, hip::std::ref(i),
                                                          hip::std::ref(j));
        assert(hip::std::get<0>(t) == 1);
        assert(hip::std::get<1>(t) == 0);
        assert(hip::std::get<2>(t) == 0);
        i = 2;
        j = 3.5;
        assert(hip::std::get<0>(t) == 1);
        assert(hip::std::get<1>(t) == 2);
        assert(hip::std::get<2>(t) == 3.5);
        hip::std::get<1>(t) = 0;
        hip::std::get<2>(t) = 0;
        assert(i == 0);
        assert(j == 0);
    }
#if TEST_STD_VER > 11
    {
        constexpr auto t1 = hip::std::make_tuple(0, 1, 3.14);
        constexpr int i1 = hip::std::get<1>(t1);
        constexpr double d1 = hip::std::get<2>(t1);
        static_assert (i1 == 1, "" );
        static_assert (d1 == 3.14, "" );
    }
#endif

  return 0;
}
