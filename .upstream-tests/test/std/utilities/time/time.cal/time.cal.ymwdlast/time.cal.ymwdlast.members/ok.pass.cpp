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

// constexpr bool ok() const noexcept;
//  Returns: y_.ok() && m_.ok() && wdl_.ok().

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year                    = hip::std::chrono::year;
    using month                   = hip::std::chrono::month;
    using weekday                 = hip::std::chrono::weekday;
    using weekday_last            = hip::std::chrono::weekday_last;
    using year_month_weekday_last = hip::std::chrono::year_month_weekday_last;

    constexpr month January   = hip::std::chrono::January;
    constexpr weekday Tuesday = hip::std::chrono::Tuesday;

    ASSERT_NOEXCEPT(                std::declval<const year_month_weekday_last>().ok());
    ASSERT_SAME_TYPE(bool, decltype(hip::std::declval<const year_month_weekday_last>().ok()));

    static_assert(!year_month_weekday_last{year{-32768}, month{}, weekday_last{weekday{}}}.ok(),  ""); // All three bad

    static_assert(!year_month_weekday_last{year{-32768}, January, weekday_last{Tuesday}}.ok(),    ""); // Bad year
    static_assert(!year_month_weekday_last{year{2019},   month{}, weekday_last{Tuesday}}.ok(),    ""); // Bad month
    static_assert(!year_month_weekday_last{year{2019},   January, weekday_last{weekday{8}}}.ok(), ""); // Bad day

    static_assert(!year_month_weekday_last{year{-32768}, month{}, weekday_last{Tuesday}}.ok(),    ""); // Bad year & month
    static_assert(!year_month_weekday_last{year{2019},   month{}, weekday_last{weekday{8}}}.ok(), ""); // Bad month & day
    static_assert(!year_month_weekday_last{year{-32768}, January, weekday_last{weekday{8}}}.ok(), ""); // Bad year & day

    static_assert( year_month_weekday_last{year{2019},   January, weekday_last{Tuesday}}.ok(),    ""); // All OK

    for (unsigned i = 0; i <= 50; ++i)
    {
        year_month_weekday_last ym{year{2019}, January, weekday_last{Tuesday}};
        assert((ym.ok() == weekday_last{Tuesday}.ok()));
    }

    for (unsigned i = 0; i <= 50; ++i)
    {
        year_month_weekday_last ym{year{2019}, January, weekday_last{weekday{i}}};
        assert((ym.ok() == weekday_last{weekday{i}}.ok()));
    }

    for (unsigned i = 0; i <= 50; ++i)
    {
        year_month_weekday_last ym{year{2019}, month{i}, weekday_last{Tuesday}};
        assert((ym.ok() == month{i}.ok()));
    }

    const int ymax = static_cast<int>(year::max());
    for (int i = ymax - 100; i <= ymax + 100; ++i)
    {
        year_month_weekday_last ym{year{i}, January, weekday_last{Tuesday}};
        assert((ym.ok() == year{i}.ok()));
    }

  return 0;
}
