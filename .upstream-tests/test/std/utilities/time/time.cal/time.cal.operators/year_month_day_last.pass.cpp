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

// constexpr year_month_day_last
//   operator/(const year_month& ym, last_spec) noexcept;
// Returns: {ym.year(), month_day_last{ym.month()}}.


// constexpr year_month_day_last
//   operator/(const year& y, const month_day_last& mdl) noexcept;
// Returns: {y, mdl}.
//
// constexpr year_month_day_last
//   operator/(int y, const month_day_last& mdl) noexcept;
// Returns: year(y) / mdl.
//
// constexpr year_month_day_last
//   operator/(const month_day_last& mdl, const year& y) noexcept;
// Returns: y / mdl.
//
// constexpr year_month_day_last
//   operator/(const month_day_last& mdl, int y) noexcept;
// Returns: year(y) / mdl.


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using month               = hip::std::chrono::month;
    using year_month          = hip::std::chrono::year_month;
    using year                = hip::std::chrono::year;
    using month_day_last      = hip::std::chrono::month_day_last;
    using year_month_day_last = hip::std::chrono::year_month_day_last;

    constexpr month February = hip::std::chrono::February;
    constexpr hip::std::chrono::last_spec last = hip::std::chrono::last;

    { // operator/(const year_month& ym, last_spec)
        constexpr year_month Feb2018{year{2018}, February};

        ASSERT_NOEXCEPT (                              Feb2018/last);
        ASSERT_SAME_TYPE(year_month_day_last, decltype(Feb2018/last));

        static_assert((Feb2018/last).year()  == year{2018}, "");
        static_assert((Feb2018/last).month() == February,   "");

        for (int i = 1000; i < 1010; ++i)
            for (unsigned j = 1; j <= 12; ++j)
            {
                year y{i};
                month m{j};
                year_month_day_last ymdl = year_month{y,m}/last;
                assert(ymdl.year()  == y);
                assert(ymdl.month() == m);
            }
    }


    { // operator/(const year& y, const month_day_last& mdl) (and switched)
        ASSERT_NOEXCEPT (                              year{2018}/month_day_last{February});
        ASSERT_SAME_TYPE(year_month_day_last, decltype(year{2018}/month_day_last{February}));
        ASSERT_NOEXCEPT (                              month_day_last{February}/year{2018});
        ASSERT_SAME_TYPE(year_month_day_last, decltype(month_day_last{February}/year{2018}));

        static_assert((year{2018}/month_day_last{February}).month() == February,   "");
        static_assert((year{2018}/month_day_last{February}).year()  == year{2018}, "");
        static_assert((month_day_last{February}/year{2018}).month() == February,   "");
        static_assert((month_day_last{February}/year{2018}).year()  == year{2018}, "");

        for (int i = 1000; i < 1010; ++i)
            for (unsigned j = 1; j <= 12; ++j)
            {
                year y{i};
                month m{j};
                year_month_day_last ymdl1 = y/month_day_last{m};
                year_month_day_last ymdl2 = month_day_last{m}/y;
                assert(ymdl1.month() == m);
                assert(ymdl2.month() == m);
                assert(ymdl2.year()  == y);
                assert(ymdl1.year()  == y);
                assert(ymdl1 == ymdl2);
            }
    }

    { // operator/(int y, const month_day_last& mdl) (and switched)
        ASSERT_NOEXCEPT (                              2018/month_day_last{February});
        ASSERT_SAME_TYPE(year_month_day_last, decltype(2018/month_day_last{February}));
        ASSERT_NOEXCEPT (                              month_day_last{February}/2018);
        ASSERT_SAME_TYPE(year_month_day_last, decltype(month_day_last{February}/2018));

        static_assert((2018/month_day_last{February}).month() == February,   "");
        static_assert((2018/month_day_last{February}).year()  == year{2018}, "");
        static_assert((month_day_last{February}/2018).month() == February,   "");
        static_assert((month_day_last{February}/2018).year()  == year{2018}, "");

        for (int i = 1000; i < 1010; ++i)
            for (unsigned j = 1; j <= 12; ++j)
            {
                year y{i};
                month m{j};
                year_month_day_last ymdl1 = i/month_day_last{m};
                year_month_day_last ymdl2 = month_day_last{m}/i;
                assert(ymdl1.month() == m);
                assert(ymdl2.month() == m);
                assert(ymdl2.year()  == y);
                assert(ymdl1.year()  == y);
                assert(ymdl1 == ymdl2);
            }
    }

  return 0;
}
