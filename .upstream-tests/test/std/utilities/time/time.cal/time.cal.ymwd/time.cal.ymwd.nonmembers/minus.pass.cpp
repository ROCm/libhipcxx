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

// constexpr year_month_weekday operator-(const year_month_weekday& ymwd, const months& dm) noexcept;
//   Returns: ymwd + (-dm).
//
// constexpr year_month_weekday operator-(const year_month_weekday& ymwd, const years& dy) noexcept;
//   Returns: ymwd + (-dy).


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"


__host__ __device__
constexpr bool testConstexprYears ()
{
    hip::std::chrono::year_month_weekday ym0{hip::std::chrono::year{1234}, hip::std::chrono::January, hip::std::chrono::weekday_indexed{hip::std::chrono::Tuesday, 1}};
    hip::std::chrono::year_month_weekday ym1 = ym0 - hip::std::chrono::years{10};
    return
        ym1.year()    == hip::std::chrono::year{1234-10}
     && ym1.month()   == hip::std::chrono::January
     && ym1.weekday() == hip::std::chrono::Tuesday
     && ym1.index()   == 1
        ;
}

__host__ __device__
constexpr bool testConstexprMonths ()
{
    hip::std::chrono::year_month_weekday ym0{hip::std::chrono::year{1234}, hip::std::chrono::November, hip::std::chrono::weekday_indexed{hip::std::chrono::Tuesday, 1}};
    hip::std::chrono::year_month_weekday ym1 = ym0 - hip::std::chrono::months{6};
    return
        ym1.year()    == hip::std::chrono::year{1234}
     && ym1.month()   == hip::std::chrono::May
     && ym1.weekday() == hip::std::chrono::Tuesday
     && ym1.index()   == 1
        ;
}


int main(int, char**)
{
    using year               = hip::std::chrono::year;
    using month              = hip::std::chrono::month;
    using weekday            = hip::std::chrono::weekday;
    using weekday_indexed    = hip::std::chrono::weekday_indexed;
    using year_month_weekday = hip::std::chrono::year_month_weekday;
    using years              = hip::std::chrono::years;
    using months             = hip::std::chrono::months;

    constexpr month November  = hip::std::chrono::November;
    constexpr weekday Tuesday = hip::std::chrono::Tuesday;

    {  // year_month_weekday - years
    ASSERT_NOEXCEPT(                              std::declval<year_month_weekday>() - std::declval<years>());
    ASSERT_SAME_TYPE(year_month_weekday, decltype(hip::std::declval<year_month_weekday>() - std::declval<years>()));

    static_assert(testConstexprYears(), "");

    year_month_weekday ym{year{1234}, November, weekday_indexed{Tuesday, 1}};
    for (int i = 0; i <= 10; ++i)
    {
        year_month_weekday ym1 = ym - years{i};
        assert(static_cast<int>(ym1.year()) == 1234 - i);
        assert(ym1.month()   == November);
        assert(ym1.weekday() == Tuesday);
        assert(ym1.index()   == 1);
    }
    }

    {  // year_month_weekday - months
    ASSERT_NOEXCEPT(                              std::declval<year_month_weekday>() - std::declval<months>());
    ASSERT_SAME_TYPE(year_month_weekday, decltype(hip::std::declval<year_month_weekday>() - std::declval<months>()));

    static_assert(testConstexprMonths(), "");

    year_month_weekday ym{year{1234}, November, weekday_indexed{Tuesday, 2}};
    for (unsigned i = 1; i <= 10; ++i)
    {
        year_month_weekday ym1 = ym - months{i};
        assert(ym1.year()    == year{1234});
        assert(ym1.month()   == month{11-i});
        assert(ym1.weekday() == Tuesday);
        assert(ym1.index()   == 2);
    }
    }

  return 0;
}
