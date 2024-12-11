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
// UNSUPPORTED: c++98, c++03, c++11, nvrtc, nvrtc

// <chrono>
// class weekday;

// constexpr weekday operator-(const weekday& x, const days& y) noexcept;
//   Returns: x + -y.
//
// constexpr days operator-(const weekday& x, const weekday& y) noexcept;
// Returns: If x.ok() == true and y.ok() == true, returns a value d in the range
//    [days{0}, days{6}] satisfying y + d == x.
// Otherwise the value returned is unspecified.
// [Example: Sunday - Monday == days{6}. —end example]


extern "C" int printf(const char *, ...);

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "../../euclidian.h"

template <typename WD, typename Ds>
__host__ __device__
constexpr bool testConstexpr()
{
    {
    WD wd{5};
    Ds offset{3};
    if (wd - offset != WD{2}) return false;
    if (wd - WD{2} != offset) return false;
    }

//  Check the example
    if (WD{0} - WD{1} != Ds{6}) return false;
    return true;
}

int main(int, char**)
{
    using weekday  = hip::std::chrono::weekday;
    using days     = hip::std::chrono::days;

    ASSERT_NOEXCEPT(                   std::declval<weekday>() - std::declval<days>());
    ASSERT_SAME_TYPE(weekday, decltype(hip::std::declval<weekday>() - std::declval<days>()));

    ASSERT_NOEXCEPT(                   std::declval<weekday>() - std::declval<weekday>());
    ASSERT_SAME_TYPE(days,    decltype(hip::std::declval<weekday>() - std::declval<weekday>()));

    static_assert(testConstexpr<weekday, days>(), "");

    for (unsigned i = 0; i <= 6; ++i)
        for (unsigned j = 0; j <= 6; ++j)
        {
            weekday wd = weekday{i} - days{j};
            assert(wd + days{j} == weekday{i});
            assert((wd.c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, j)));
        }

    for (unsigned i = 0; i <= 6; ++i)
        for (unsigned j = 0; j <= 6; ++j)
        {
            days d = weekday{j} - weekday{i};
            assert(weekday{i} + d == weekday{j});
        }


  return 0;
}
