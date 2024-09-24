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

// constexpr year_month_day operator+(const year_month_day& ymd, const months& dm) noexcept;
//   Returns: (ymd.year() / ymd.month() + dm) / ymd.day().
//
// constexpr year_month_day operator+(const months& dm, const year_month_day& ymd) noexcept;
//   Returns: ymd + dm.
//
//
// constexpr year_month_day operator+(const year_month_day& ymd, const years& dy) noexcept;
//   Returns: (ymd.year() + dy) / ymd.month() / ymd.day().
//
// constexpr year_month_day operator+(const years& dy, const year_month_day& ymd) noexcept;
//   Returns: ym + dm.



#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

__host__ __device__
constexpr bool testConstexprYears(hip::std::chrono::year_month_day ym)
{
    hip::std::chrono::years offset{23};
    if (static_cast<int>((ym         ).year()) !=  1) return false;
    if (static_cast<int>((ym + offset).year()) != 24) return false;
    if (static_cast<int>((offset + ym).year()) != 24) return false;
    return true;
}


__host__ __device__
constexpr bool testConstexprMonths(hip::std::chrono::year_month_day ym)
{
    hip::std::chrono::months offset{6};
    if (static_cast<unsigned>((ym         ).month()) !=  1) return false;
    if (static_cast<unsigned>((ym + offset).month()) !=  7) return false;
    if (static_cast<unsigned>((offset + ym).month()) !=  7) return false;
    return true;
}


int main(int, char**)
{
    using day        = hip::std::chrono::day;
    using year       = hip::std::chrono::year;
    using years      = hip::std::chrono::years;
    using month      = hip::std::chrono::month;
    using months     = hip::std::chrono::months;
    using year_month_day = hip::std::chrono::year_month_day;

    {   // year_month_day + months
    ASSERT_NOEXCEPT(hip::std::declval<year_month_day>() + std::declval<months>());
    ASSERT_NOEXCEPT(hip::std::declval<months>() + std::declval<year_month_day>());

    ASSERT_SAME_TYPE(year_month_day, decltype(hip::std::declval<year_month_day>() + std::declval<months>()));
    ASSERT_SAME_TYPE(year_month_day, decltype(hip::std::declval<months>() + std::declval<year_month_day>()));

    static_assert(testConstexprMonths(year_month_day{year{1}, month{1}, day{1}}), "");

    year_month_day ym{year{1234}, hip::std::chrono::January, day{12}};
    for (int i = 0; i <= 10; ++i)  // TODO test wrap-around
    {
        year_month_day ym1 = ym + months{i};
        year_month_day ym2 = months{i} + ym;
        assert(static_cast<int>(ym1.year()) == 1234);
        assert(static_cast<int>(ym2.year()) == 1234);
        assert(ym1.month() == month(1 + i));
        assert(ym2.month() == month(1 + i));
        assert(ym1.day()   == day{12});
        assert(ym2.day()   == day{12});
        assert(ym1 == ym2);
    }
    }

    {   // year_month_day + years
    ASSERT_NOEXCEPT(hip::std::declval<year_month_day>() + std::declval<years>());
    ASSERT_NOEXCEPT(hip::std::declval<years>() + std::declval<year_month_day>());

    ASSERT_SAME_TYPE(year_month_day, decltype(hip::std::declval<year_month_day>() + std::declval<years>()));
    ASSERT_SAME_TYPE(year_month_day, decltype(hip::std::declval<years>() + std::declval<year_month_day>()));

    static_assert(testConstexprYears (year_month_day{year{1}, month{1}, day{1}}), "");

    auto constexpr January = hip::std::chrono::January;

    year_month_day ym{year{1234}, January, day{12}};
    for (int i = 0; i <= 10; ++i)
    {
        year_month_day ym1 = ym + years{i};
        year_month_day ym2 = years{i} + ym;
        assert(static_cast<int>(ym1.year()) == i + 1234);
        assert(static_cast<int>(ym2.year()) == i + 1234);
        assert(ym1.month() == January);
        assert(ym2.month() == January);
        assert(ym1.day()   == day{12});
        assert(ym2.day()   == day{12});
        assert(ym1 == ym2);
    }
    }


  return 0;
}
