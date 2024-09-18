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
// class year_month_day_last;

// constexpr bool operator==(const year_month_day_last& x, const year_month_day_last& y) noexcept;
//   Returns: x.year() == y.year() && x.month_day_last() == y.month_day_last().
//
// constexpr bool operator< (const year_month_day_last& x, const year_month_day_last& y) noexcept;
//   Returns:
//      If x.year() < y.year(), returns true.
//      Otherwise, if x.year() > y.year(), returns false.
//      Otherwise, returns x.month_day_last() < y.month_day_last()

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using year                = hip::std::chrono::year;
    using month               = hip::std::chrono::month;
    using month_day_last      = hip::std::chrono::month_day_last;
    using year_month_day_last = hip::std::chrono::year_month_day_last;

    AssertComparisons6AreNoexcept<year_month_day_last>();
    AssertComparisons6ReturnBool<year_month_day_last>();

    constexpr month January = hip::std::chrono::January;
    constexpr month February = hip::std::chrono::February;

    static_assert( testComparisons6(
        year_month_day_last{year{1234}, month_day_last{January}},
        year_month_day_last{year{1234}, month_day_last{January}},
        true, false), "");

//  different month
    static_assert( testComparisons6(
        year_month_day_last{year{1234}, month_day_last{January}},
        year_month_day_last{year{1234}, month_day_last{February}},
        false, true), "");

//  different year
    static_assert( testComparisons6(
        year_month_day_last{year{1234}, month_day_last{January}},
        year_month_day_last{year{1235}, month_day_last{January}},
        false, true), "");

//  different month
    static_assert( testComparisons6(
        year_month_day_last{year{1234}, month_day_last{January}},
        year_month_day_last{year{1234}, month_day_last{February}},
        false, true), "");

//  different year and month
    static_assert( testComparisons6(
        year_month_day_last{year{1234}, month_day_last{February}},
        year_month_day_last{year{1235}, month_day_last{January}},
        false, true), "");

//  same year, different months
    for (unsigned i = 1; i < 12; ++i)
        for (unsigned j = 1; j < 12; ++j)
            assert((testComparisons6(
                year_month_day_last{year{1234}, month_day_last{month{i}}},
                year_month_day_last{year{1234}, month_day_last{month{j}}},
                i == j, i < j )));

//  same month, different years
    for (int i = 1000; i < 20; ++i)
        for (int j = 1000; j < 20; ++j)
        assert((testComparisons6(
            year_month_day_last{year{i}, month_day_last{January}},
            year_month_day_last{year{j}, month_day_last{January}},
            i == j, i < j )));

  return 0;
}
