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

//  constexpr year_month_day(const sys_days& dp) noexcept;
//
//  Effects:  Constructs an object of type year_month_day that corresponds
//                to the date represented by dp.
//
//  Remarks: For any value ymd of type year_month_day for which ymd.ok() is true,
//                ymd == year_month_day{sys_days{ymd}} is true.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

// MSVC warns about unsigned/signed comparisons and addition/subtraction
// Silence these warnings, but not the ones within the header itself.
#if defined(_MSC_VER)
# pragma warning( disable: 4307 )
# pragma warning( disable: 4308 )
#endif

int main(int, char**)
{
    using year           = hip::std::chrono::year;
    using day            = hip::std::chrono::day;
    using sys_days       = hip::std::chrono::sys_days;
    using days           = hip::std::chrono::days;
    using year_month_day = hip::std::chrono::year_month_day;

    ASSERT_NOEXCEPT(year_month_day{hip::std::declval<sys_days>()});

    auto constexpr January = hip::std::chrono::January;

    {
    constexpr sys_days sd{};
    constexpr year_month_day ymd{sd};

    static_assert( ymd.ok(),                            "");
    static_assert( ymd.year()  == year{1970},           "");
    static_assert( ymd.month() == January, "");
    static_assert( ymd.day()   == day{1},               "");
    }

    {
    constexpr sys_days sd{days{10957+32}};
    constexpr year_month_day ymd{sd};

    auto constexpr February = hip::std::chrono::February;

    static_assert( ymd.ok(),                             "");
    static_assert( ymd.year()  == year{2000},            "");
    static_assert( ymd.month() == February, "");
    static_assert( ymd.day()   == day{2},                "");
    }


//  There's one more leap day between 1/1/40 and 1/1/70
//  when compared to 1/1/70 -> 1/1/2000
    {
    constexpr sys_days sd{days{-10957}};
    constexpr year_month_day ymd{sd};

    static_assert( ymd.ok(),                            "");
    static_assert( ymd.year()  == year{1940},           "");
    static_assert( ymd.month() == January, "");
    static_assert( ymd.day()   == day{2},               "");
    }

    {
    sys_days sd{days{-(10957+34)}};
    year_month_day ymd{sd};
    auto constexpr November = hip::std::chrono::November;

    assert( ymd.ok());
    assert( ymd.year()  == year{1939});
    assert( ymd.month() == November);
    assert( ymd.day()   == day{29});
    }

  return 0;
}
