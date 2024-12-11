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
// class year_month_weekday;

// constexpr bool operator==(const year_month_weekday& x, const year_month_weekday& y) noexcept;
//   Returns: x.year() == y.year() && x.month() == y.month() && x.weekday_indexed() == y.weekday_indexed()
//


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using year               = hip::std::chrono::year;
    using month              = hip::std::chrono::month;
    using weekday_indexed    = hip::std::chrono::weekday_indexed;
    using weekday            = hip::std::chrono::weekday;
    using year_month_weekday = hip::std::chrono::year_month_weekday;

    AssertComparisons2AreNoexcept<year_month_weekday>();
    AssertComparisons2ReturnBool<year_month_weekday>();

    constexpr month January   = hip::std::chrono::January;
    constexpr month February  = hip::std::chrono::February;
    constexpr weekday Tuesday = hip::std::chrono::Tuesday;

    static_assert( testComparisons2(
        year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 1}},
        year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 1}},
        true), "");

//  different day
    static_assert( testComparisons2(
        year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 1}},
        year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 2}},
        false), "");

//  different month
    static_assert( testComparisons2(
        year_month_weekday{year{1234}, January,  weekday_indexed{Tuesday, 1}},
        year_month_weekday{year{1234}, February, weekday_indexed{Tuesday, 1}},
        false), "");

//  different year
    static_assert( testComparisons2(
        year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 1}},
        year_month_weekday{year{1235}, January, weekday_indexed{Tuesday, 1}},
        false), "");


//  different month and day
    static_assert( testComparisons2(
        year_month_weekday{year{1234}, January,  weekday_indexed{Tuesday, 1}},
        year_month_weekday{year{1234}, February, weekday_indexed{Tuesday, 2}},
        false), "");

//  different year and month
    static_assert( testComparisons2(
        year_month_weekday{year{1234}, February, weekday_indexed{Tuesday, 1}},
        year_month_weekday{year{1235}, January,  weekday_indexed{Tuesday, 1}},
        false), "");

//  different year and day
    static_assert( testComparisons2(
        year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 2}},
        year_month_weekday{year{1235}, January, weekday_indexed{Tuesday, 1}},
        false), "");

//  different year, month and day
    static_assert( testComparisons2(
        year_month_weekday{year{1234}, February, weekday_indexed{Tuesday, 2}},
        year_month_weekday{year{1235}, January,  weekday_indexed{Tuesday, 1}},
        false), "");


//  same year, different days
    for (unsigned i = 1; i < 28; ++i)
        for (unsigned j = 1; j < 28; ++j)
            assert((testComparisons2(
                year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, i}},
                year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, j}},
                i == j)));

//  same year, different months
    for (unsigned i = 1; i < 12; ++i)
        for (unsigned j = 1; j < 12; ++j)
            assert((testComparisons2(
                year_month_weekday{year{1234}, month{i}, weekday_indexed{Tuesday, 1}},
                year_month_weekday{year{1234}, month{j}, weekday_indexed{Tuesday, 1}},
                i == j)));

//  same month, different years
    for (int i = 1000; i < 20; ++i)
        for (int j = 1000; j < 20; ++j)
        assert((testComparisons2(
            year_month_weekday{year{i}, January, weekday_indexed{Tuesday, 1}},
            year_month_weekday{year{j}, January, weekday_indexed{Tuesday, 1}},
            i == j)));

  return 0;
}
