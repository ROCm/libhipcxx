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

//  constexpr year_month_day(const year_month_day_last& ymdl) noexcept;
//
//  Effects:  Constructs an object of type year_month_day by initializing
//              y_ with ymdl.year(), m_ with ymdl.month(), and d_ with ymdl.day().
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr chrono::day     day() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year                = hip::std::chrono::year;
    using month               = hip::std::chrono::month;
    using day                 = hip::std::chrono::day;
    using month_day_last = hip::std::chrono::month_day_last;
    using year_month_day_last = hip::std::chrono::year_month_day_last;
    using year_month_day      = hip::std::chrono::year_month_day;

    ASSERT_NOEXCEPT(year_month_day{hip::std::declval<const year_month_day_last>()});

    {
    constexpr year_month_day_last ymdl{year{2019}, month_day_last{month{1}}};
    constexpr year_month_day ymd{ymdl};

    static_assert( ymd.year()  == year{2019}, "");
    static_assert( ymd.month() == month{1},   "");
    static_assert( ymd.day()   == day{31},    "");
    static_assert( ymd.ok(),                  "");
    }

    {
    constexpr year_month_day_last ymdl{year{1970}, month_day_last{month{4}}};
    constexpr year_month_day ymd{ymdl};

    static_assert( ymd.year()  == year{1970}, "");
    static_assert( ymd.month() == month{4},   "");
    static_assert( ymd.day()   == day{30},    "");
    static_assert( ymd.ok(),                  "");
    }

    {
    constexpr year_month_day_last ymdl{year{2000}, month_day_last{month{2}}};
    constexpr year_month_day ymd{ymdl};

    static_assert( ymd.year()  == year{2000}, "");
    static_assert( ymd.month() == month{2},   "");
    static_assert( ymd.day()   == day{29},    "");
    static_assert( ymd.ok(),                  "");
    }

    { // Feb 1900 was NOT a leap year.
    year_month_day_last ymdl{year{1900}, month_day_last{month{2}}};
    year_month_day ymd{ymdl};

    assert( ymd.year()  == year{1900});
    assert( ymd.month() == month{2});
    assert( ymd.day()   == day{28});
    assert( ymd.ok());
    }

  return 0;
}
