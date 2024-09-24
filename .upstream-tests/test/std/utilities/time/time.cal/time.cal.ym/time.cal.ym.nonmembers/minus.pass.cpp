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
// UNSUPPORTED: c++98, c++03, c++11
// XFAIL: *

// <chrono>
// class year_month;

// constexpr year_month operator-(const year_month& ym, const years& dy) noexcept;
// Returns: ym + -dy.
//
// constexpr year_month operator-(const year_month& ym, const months& dm) noexcept;
// Returns: ym + -dm.
//
// constexpr months operator-(const year_month& x, const year_month& y) noexcept;
// Returns: x.year() - y.year() + months{static_cast<int>(unsigned{x.month()}) -
//                                       static_cast<int>(unsigned{y.month()})}


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

#include <iostream>

int main(int, char**)
{
    using year       = hip::std::chrono::year;
    using years      = hip::std::chrono::years;
    using month      = hip::std::chrono::month;
    using months     = hip::std::chrono::months;
    using year_month = hip::std::chrono::year_month;

    auto constexpr January = hip::std::chrono::January;

    {   // year_month - years
    ASSERT_NOEXCEPT(                      hip::std::declval<year_month>() - hip::std::declval<years>());
    ASSERT_SAME_TYPE(year_month, decltype(hip::std::declval<year_month>() - hip::std::declval<years>()));

//  static_assert(testConstexprYears (year_month{year{1}, month{1}}), "");

    year_month ym{year{1234}, January};
    for (int i = 0; i <= 10; ++i)
    {
        year_month ym1 = ym - years{i};
        assert(static_cast<int>(ym1.year()) == 1234 - i);
        assert(ym1.month() == hip::std::chrono::January);
    }
    }

    {   // year_month - months
    ASSERT_NOEXCEPT(                      hip::std::declval<year_month>() - hip::std::declval<months>());
    ASSERT_SAME_TYPE(year_month, decltype(hip::std::declval<year_month>() - hip::std::declval<months>()));

//  static_assert(testConstexprMonths(year_month{year{1}, month{1}}), "");

    auto constexpr November = hip::std::chrono::November;
    year_month ym{year{1234}, November};
    for (int i = 0; i <= 10; ++i)  // TODO test wrap-around
    {
        year_month ym1 = ym - months{i};
        assert(static_cast<int>(ym1.year()) == 1234);
        assert(ym1.month() == month(11 - i));
    }
    }

    {   // year_month - year_month
    ASSERT_NOEXCEPT(                  hip::std::declval<year_month>() - hip::std::declval<year_month>());
    ASSERT_SAME_TYPE(months, decltype(hip::std::declval<year_month>() - hip::std::declval<year_month>()));

//  static_assert(testConstexprMonths(year_month{year{1}, month{1}}), "");

//  Same year
    year y{2345};
    for (int i = 1; i <= 12; ++i)
        for (int j = 1; j <= 12; ++j)
    {
        months diff = year_month{y, month(i)} - year_month{y, month(j)};
        std::cout << "i: " << i << " j: " << j << " -> " << diff.count() << std::endl;
        assert(diff.count() == i - j);
    }

//  TODO: different year

    }

  return 0;
}
