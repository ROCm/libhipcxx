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
//   tuple<Types&...> tie(Types&... t);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <hip/std/tuple>

// hip::std::string not supported
// #include <hip/std/string>
#include <hip/std/cassert>

#include "test_macros.h"

#if TEST_STD_VER > 11
__host__ __device__ constexpr bool test_tie_constexpr() {
    {
        int i = 42;
        double f = 1.1;
        using ExpectT = hip::std::tuple<int&, decltype(hip::std::ignore)&, double&>;
        auto res = hip::std::tie(i, hip::std::ignore, f);
        static_assert(hip::std::is_same<ExpectT, decltype(res)>::value, "");
        assert(&hip::std::get<0>(res) == &i);
        assert(&hip::std::get<1>(res) == &hip::std::ignore);
        assert(&hip::std::get<2>(res) == &f);
        // FIXME: If/when tuple gets constexpr assignment
        //res = hip::std::make_tuple(101, nullptr, -1.0);
    }
    return true;
}
#endif

int main(int, char**)
{
    {
        int i = 0;
        const char *_s = "C++";
        // hip::std::string not supported
        // hip::std::string s;
        const char *s;
        hip::std::tie(i, hip::std::ignore, s) = hip::std::make_tuple(42, 3.14, _s);
        assert(i == 42);
        assert(s == _s);
    }
#if TEST_STD_VER > 11
    {
        static constexpr int i = 42;
        static constexpr double f = 1.1;
        constexpr hip::std::tuple<const int &, const double &> t = hip::std::tie(i, f);
        static_assert ( hip::std::get<0>(t) == 42, "" );
        static_assert ( hip::std::get<1>(t) == 1.1, "" );
    }
    {
        static_assert(test_tie_constexpr(), "");
    }
#endif

  return 0;
}
