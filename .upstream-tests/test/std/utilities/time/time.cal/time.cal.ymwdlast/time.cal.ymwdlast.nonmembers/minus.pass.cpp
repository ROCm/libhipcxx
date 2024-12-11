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
// class year_month_weekday_last;

// constexpr year_month_weekday_last operator-(const year_month_weekday_last& ymwdl, const months& dm) noexcept;
//   Returns: ymwdl + (-dm).
//
// constexpr year_month_weekday_last operator-(const year_month_weekday_last& ymwdl, const years& dy) noexcept;
//   Returns: ymwdl + (-dy).

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"


__host__ __device__
constexpr bool testConstexprYears(hip::std::chrono::year_month_weekday_last ym)
{
    hip::std::chrono::years offset{14};
    if (static_cast<int>((ym         ).year()) != 66) return false;
    if (static_cast<int>((ym - offset).year()) != 52) return false;
    return true;
}

__host__ __device__
constexpr bool testConstexprMonths(hip::std::chrono::year_month_weekday_last ym)
{
    hip::std::chrono::months offset{6};
    if (static_cast<unsigned>((ym         ).month()) != 10) return false;
    if (static_cast<unsigned>((ym - offset).month()) !=  4) return false;
    return true;
}

int main(int, char**)
{
    using year                    = hip::std::chrono::year;
    using month                   = hip::std::chrono::month;
    using weekday                 = hip::std::chrono::weekday;
    using weekday_last            = hip::std::chrono::weekday_last;
    using year_month_weekday_last = hip::std::chrono::year_month_weekday_last;
    using years                   = hip::std::chrono::years;
    using months                  = hip::std::chrono::months;

    constexpr month October = hip::std::chrono::October;
    constexpr weekday Tuesday = hip::std::chrono::Tuesday;

    { // year_month_weekday_last - years

    ASSERT_NOEXCEPT(                                   std::declval<year_month_weekday_last>() - std::declval<years>());
    ASSERT_SAME_TYPE(year_month_weekday_last, decltype(hip::std::declval<year_month_weekday_last>() - std::declval<years>()));

    static_assert(testConstexprYears(year_month_weekday_last{year{66}, October, weekday_last{Tuesday}}), "");

    year_month_weekday_last ym{year{1234}, October, weekday_last{Tuesday}};
    for (int i = 0; i <= 10; ++i)
    {
        year_month_weekday_last ym1 = ym - years{i};
        assert(ym1.year()         == year{1234 - i});
        assert(ym1.month()        == October);
        assert(ym1.weekday()      == Tuesday);
        assert(ym1.weekday_last() == weekday_last{Tuesday});
    }
    }

    { // year_month_weekday_last - months

    ASSERT_NOEXCEPT(                                   std::declval<year_month_weekday_last>() - std::declval<months>());
    ASSERT_SAME_TYPE(year_month_weekday_last, decltype(hip::std::declval<year_month_weekday_last>() - std::declval<months>()));

    static_assert(testConstexprMonths(year_month_weekday_last{year{66}, October, weekday_last{Tuesday}}), "");

    year_month_weekday_last ym{year{1234}, October, weekday_last{Tuesday}};
    for (unsigned i = 0; i < 10; ++i)
    {
        year_month_weekday_last ym1 = ym - months{i};
        assert(ym1.year()         == year{1234});
        assert(ym1.month()        == month{10 - i});
        assert(ym1.weekday()      == Tuesday);
        assert(ym1.weekday_last() == weekday_last{Tuesday});
    }
    }


  return 0;
}
