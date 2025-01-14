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

// constexpr year_month_day
//   operator/(const year_month& ym, const day& d) noexcept;
// Returns: {ym.year(), ym.month(), d}.
//
// constexpr year_month_day
//   operator/(const year_month& ym, int d) noexcept;
// Returns: ym / day(d).
//
// constexpr year_month_day
//   operator/(const year& y, const month_day& md) noexcept;
// Returns: y / md.month() / md.day().
//
// constexpr year_month_day
//   operator/(int y, const month_day& md) noexcept;
// Returns: year(y) / md.
//
// constexpr year_month_day
//   operator/(const month_day& md, const year& y) noexcept;
// Returns: y / md.
//
// constexpr year_month_day
//   operator/(const month_day& md, int y) noexcept;
// Returns: year(y) / md.


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using year           = hip::std::chrono::year;
    using month          = hip::std::chrono::month;
    using day            = hip::std::chrono::day;
    using year_month     = hip::std::chrono::year_month;
    using month_day      = hip::std::chrono::month_day;
    using year_month_day = hip::std::chrono::year_month_day;

    constexpr month February = hip::std::chrono::February;
    constexpr year_month Feb2018{year{2018}, February};

    { // operator/(const year_month& ym, const day& d)
        ASSERT_NOEXCEPT (                         Feb2018/day{2});
        ASSERT_SAME_TYPE(year_month_day, decltype(Feb2018/day{2}));

        static_assert((Feb2018/day{2}).month() == February, "");
        static_assert((Feb2018/day{2}).day()   == day{2},   "");

        for (int i = 1000; i < 1010; ++i)
            for (int j = 1; j <= 12; ++j)
                for (unsigned k = 0; k <= 28; ++k)
                {
                    year y(i);
                    month m(j);
                    day d(k);
                    year_month ym(y, m);
                    year_month_day ymd = ym/d;
                    assert(ymd.year()  == y);
                    assert(ymd.month() == m);
                    assert(ymd.day()   == d);
                }
    }


    { // operator/(const year_month& ym, int d)
        ASSERT_NOEXCEPT (                         Feb2018/2);
        ASSERT_SAME_TYPE(year_month_day, decltype(Feb2018/2));

        static_assert((Feb2018/2).month() == February, "");
        static_assert((Feb2018/2).day()   == day{2},   "");

        for (int i = 1000; i < 1010; ++i)
            for (int j = 1; j <= 12; ++j)
                for (unsigned k = 0; k <= 28; ++k)
                {
                    year y(i);
                    month m(j);
                    day d(k);
                    year_month ym(y, m);
                    year_month_day ymd = ym/k;
                    assert(ymd.year()  == y);
                    assert(ymd.month() == m);
                    assert(ymd.day()   == d);
                }
    }


    { // operator/(const year_month& ym, int d)
        ASSERT_NOEXCEPT (                         Feb2018/2);
        ASSERT_SAME_TYPE(year_month_day, decltype(Feb2018/2));

        static_assert((Feb2018/2).month() == February, "");
        static_assert((Feb2018/2).day()   == day{2},   "");

        for (int i = 1000; i < 1010; ++i)
            for (int j = 1; j <= 12; ++j)
                for (unsigned k = 0; k <= 28; ++k)
                {
                    year y(i);
                    month m(j);
                    day d(k);
                    year_month ym(y, m);
                    year_month_day ymd = ym/k;
                    assert(ymd.year()  == y);
                    assert(ymd.month() == m);
                    assert(ymd.day()   == d);
                }
    }




    { // operator/(const year& y, const month_day& md) (and switched)
        ASSERT_NOEXCEPT (                         year{2018}/month_day{February, day{2}});
        ASSERT_SAME_TYPE(year_month_day, decltype(year{2018}/month_day{February, day{2}}));
        ASSERT_NOEXCEPT (                         month_day{February, day{2}}/year{2018});
        ASSERT_SAME_TYPE(year_month_day, decltype(month_day{February, day{2}}/year{2018}));

        static_assert((year{2018}/month_day{February, day{2}}).month() == February, "" );
        static_assert((year{2018}/month_day{February, day{2}}).day()   == day{2},   "" );
        static_assert((month_day{February, day{2}}/year{2018}).month() == February, "" );
        static_assert((month_day{February, day{2}}/year{2018}).day()   == day{2},   "" );

        for (int i = 1000; i < 1010; ++i)
            for (int j = 1; j <= 12; ++j)
                for (unsigned k = 0; k <= 28; ++k)
                {
                    year y(i);
                    month m(j);
                    day d(k);
                    month_day md(m, d);
                    year_month_day ymd1 = y/md;
                    year_month_day ymd2 = md/y;
                    assert(ymd1.year()  == y);
                    assert(ymd2.year()  == y);
                    assert(ymd1.month() == m);
                    assert(ymd2.month() == m);
                    assert(ymd1.day()   == d);
                    assert(ymd2.day()   == d);
                    assert(ymd1 == ymd2);
                }
    }

    { // operator/(const month_day& md, int y) (and switched)
        ASSERT_NOEXCEPT (                         2018/month_day{February, day{2}});
        ASSERT_SAME_TYPE(year_month_day, decltype(2018/month_day{February, day{2}}));
        ASSERT_NOEXCEPT (                         month_day{February, day{2}}/2018);
        ASSERT_SAME_TYPE(year_month_day, decltype(month_day{February, day{2}}/2018));

        static_assert((2018/month_day{February, day{2}}).month() == February, "" );
        static_assert((2018/month_day{February, day{2}}).day()   == day{2},   "" );
        static_assert((month_day{February, day{2}}/2018).month() == February, "" );
        static_assert((month_day{February, day{2}}/2018).day()   == day{2},   "" );

        for (int i = 1000; i < 1010; ++i)
            for (int j = 1; j <= 12; ++j)
                for (unsigned k = 0; k <= 28; ++k)
                {
                    year y(i);
                    month m(j);
                    day d(k);
                    month_day md(m, d);
                    year_month_day ymd1 = i/md;
                    year_month_day ymd2 = md/i;
                    assert(ymd1.year()  == y);
                    assert(ymd2.year()  == y);
                    assert(ymd1.month() == m);
                    assert(ymd2.month() == m);
                    assert(ymd1.day()   == d);
                    assert(ymd2.day()   == d);
                    assert(ymd1 == ymd2);
                }
    }


  return 0;
}
