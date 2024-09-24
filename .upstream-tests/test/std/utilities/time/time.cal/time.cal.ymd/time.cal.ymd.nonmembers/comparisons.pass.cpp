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
// UNSUPPORTED: c++98, c++03, c++11, nvrtc

// <chrono>
// class year_month_day;

// constexpr bool operator==(const year_month_day& x, const year_month_day& y) noexcept;
//   Returns: x.year() == y.year() && x.month() == y.month().
//
// constexpr bool operator< (const year_month_day& x, const year_month_day& y) noexcept;
//   Returns:
//      If x.year() < y.year() returns true.
//      Otherwise, if x.year() > y.year() returns false.
//      Otherwise, if x.month() < y.month() returns true.
//      Otherwise, if x.month() > y.month() returns false.
//      Otherwise, returns x.day() < y.day()


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using day        = hip::std::chrono::day;
    using year       = hip::std::chrono::year;
    using month      = hip::std::chrono::month;
    using year_month_day = hip::std::chrono::year_month_day;

    AssertComparisons6AreNoexcept<year_month_day>();
    AssertComparisons6ReturnBool<year_month_day>();

    constexpr month January = hip::std::chrono::January;
    constexpr month February = hip::std::chrono::February;

    static_assert( testComparisons6(
        year_month_day{year{1234}, January, day{1}},
        year_month_day{year{1234}, January, day{1}},
        true, false), "");

//  different day
    static_assert( testComparisons6(
        year_month_day{year{1234}, January, day{1}},
        year_month_day{year{1234}, January, day{2}},
        false, true), "");

//  different month
    static_assert( testComparisons6(
        year_month_day{year{1234}, January, day{1}},
        year_month_day{year{1234}, February, day{1}},
        false, true), "");

//  different year
    static_assert( testComparisons6(
        year_month_day{year{1234}, January, day{1}},
        year_month_day{year{1235}, January, day{1}},
        false, true), "");


//  different month and day
    static_assert( testComparisons6(
        year_month_day{year{1234}, January, day{2}},
        year_month_day{year{1234}, February, day{1}},
        false, true), "");

//  different year and month
    static_assert( testComparisons6(
        year_month_day{year{1234}, February, day{1}},
        year_month_day{year{1235}, January, day{1}},
        false, true), "");

//  different year and day
    static_assert( testComparisons6(
        year_month_day{year{1234}, January, day{2}},
        year_month_day{year{1235}, January, day{1}},
        false, true), "");

//  different year, month and day
    static_assert( testComparisons6(
        year_month_day{year{1234}, February, day{2}},
        year_month_day{year{1235}, January, day{1}},
        false, true), "");


//  same year, different days
    for (unsigned i = 1; i < 28; ++i)
        for (unsigned j = 1; j < 28; ++j)
            assert((testComparisons6(
                year_month_day{year{1234}, January, day{i}},
                year_month_day{year{1234}, January, day{j}},
                i == j, i < j )));

//  same year, different months
    for (unsigned i = 1; i < 12; ++i)
        for (unsigned j = 1; j < 12; ++j)
            assert((testComparisons6(
                year_month_day{year{1234}, month{i}, day{12}},
                year_month_day{year{1234}, month{j}, day{12}},
                i == j, i < j )));

//  same month, different years
    for (int i = 1000; i < 20; ++i)
        for (int j = 1000; j < 20; ++j)
        assert((testComparisons6(
            year_month_day{year{i}, January, day{12}},
            year_month_day{year{j}, January, day{12}},
            i == j, i < j )));

  return 0;
}
