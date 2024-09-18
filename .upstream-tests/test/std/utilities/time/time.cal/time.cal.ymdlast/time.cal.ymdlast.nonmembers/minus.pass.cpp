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
//   operator-(const year_month_day_last& ymdl, const months& dm) noexcept;
//
//   Returns: ymdl + (-dm).
//
// constexpr year_month_day_last
//   operator-(const year_month_day_last& ymdl, const years& dy) noexcept;
//
//   Returns: ymdl + (-dy).


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

__host__ __device__
constexpr bool testConstexprYears (hip::std::chrono::year_month_day_last ymdl)
{
    hip::std::chrono::year_month_day_last ym1 = ymdl - hip::std::chrono::years{10};
    return
        ym1.year()  == hip::std::chrono::year{static_cast<int>(ymdl.year()) - 10}
     && ym1.month() == ymdl.month()
        ;
}

__host__ __device__
constexpr bool testConstexprMonths (hip::std::chrono::year_month_day_last ymdl)
{
    hip::std::chrono::year_month_day_last ym1 = ymdl - hip::std::chrono::months{6};
    return
        ym1.year()  == ymdl.year()
     && ym1.month() == hip::std::chrono::month{static_cast<unsigned>(ymdl.month()) - 6}
        ;
}

int main(int, char**)
{
    using year                = hip::std::chrono::year;
    using month               = hip::std::chrono::month;
    using month_day_last      = hip::std::chrono::month_day_last;
    using year_month_day_last = hip::std::chrono::year_month_day_last;
    using months              = hip::std::chrono::months;
    using years               = hip::std::chrono::years;

    constexpr month December = hip::std::chrono::December;

    { // year_month_day_last - years
    ASSERT_NOEXCEPT(                               std::declval<year_month_day_last>() - std::declval<years>());
    ASSERT_SAME_TYPE(year_month_day_last, decltype(hip::std::declval<year_month_day_last>() - std::declval<years>()));

    static_assert(testConstexprYears(year_month_day_last{year{1234}, month_day_last{December}}), "");
    year_month_day_last ym{year{1234}, month_day_last{December}};
    for (int i = 0; i <= 10; ++i)
    {
        year_month_day_last ym1 = ym - years{i};
        assert(static_cast<int>(ym1.year()) == 1234 - i);
        assert(ym1.month() == December);
    }
    }

    { // year_month_day_last - months
    ASSERT_NOEXCEPT(                               std::declval<year_month_day_last>() - std::declval<months>());
    ASSERT_SAME_TYPE(year_month_day_last, decltype(hip::std::declval<year_month_day_last>() - std::declval<months>()));

    static_assert(testConstexprMonths(year_month_day_last{year{1234}, month_day_last{December}}), "");
//  TODO test wrapping
    year_month_day_last ym{year{1234}, month_day_last{December}};
    for (unsigned i = 0; i <= 10; ++i)
    {
        year_month_day_last ym1 = ym - months{i};
        assert(static_cast<int>(ym1.year()) == 1234);
        assert(static_cast<unsigned>(ym1.month()) == 12U-i);
    }
    }


  return 0;
}
