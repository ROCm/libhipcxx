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

// constexpr operator sys_days() const noexcept;
//
// Returns: If ok(), returns a sys_days holding a count of days from the
//   sys_days epoch to *this (a negative value if *this represents a date
//   prior to the sys_days epoch). Otherwise, if y_.ok() && m_.ok() is true,
//   returns a sys_days which is offset from sys_days{y_/m_/last} by the
//   number of days d_ is offset from sys_days{y_/m_/last}.day(). Otherwise
//   the value returned is unspecified.
//
// Remarks: A sys_days in the range [days{-12687428}, days{11248737}] which
//   is converted to a year_month_day shall have the same value when
//   converted back to a sys_days.
//
// [Example:
//   static_assert(year_month_day{sys_days{2017y/January/0}}  == 2016y/December/31);
//   static_assert(year_month_day{sys_days{2017y/January/31}} == 2017y/January/31);
//   static_assert(year_month_day{sys_days{2017y/January/32}} == 2017y/February/1);
// —end example]

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

__host__ __device__
void RunTheExample()
{
    using namespace hip::std::chrono;

    static_assert(year_month_day{sys_days{year{2017}/January/0}}  == year{2016}/December/31,"");
    static_assert(year_month_day{sys_days{year{2017}/January/31}} == year{2017}/January/31,"");
    static_assert(year_month_day{sys_days{year{2017}/January/32}} == year{2017}/February/1,"");
}

int main(int, char**)
{
    using year           = hip::std::chrono::year;
    using month          = hip::std::chrono::month;
    using day            = hip::std::chrono::day;
    using sys_days       = hip::std::chrono::sys_days;
    using days           = hip::std::chrono::days;
    using year_month_day = hip::std::chrono::year_month_day;

    ASSERT_NOEXCEPT(sys_days(hip::std::declval<year_month_day>()));
    RunTheExample();

    {
    constexpr year_month_day ymd{year{1970}, month{1}, day{1}};
    constexpr sys_days sd{ymd};

    static_assert( sd.time_since_epoch() == days{0}, "");
    static_assert( year_month_day{sd} == ymd, ""); // and back
    }

    {
    constexpr year_month_day ymd{year{2000}, month{2}, day{2}};
    constexpr sys_days sd{ymd};

    static_assert( sd.time_since_epoch() == days{10957+32}, "");
    static_assert( year_month_day{sd} == ymd, ""); // and back
    }

//  There's one more leap day between 1/1/40 and 1/1/70
//  when compared to 1/1/70 -> 1/1/2000
    {
    constexpr year_month_day ymd{year{1940}, month{1}, day{2}};
    constexpr sys_days sd{ymd};

    static_assert( sd.time_since_epoch() == days{-10957}, "");
    static_assert( year_month_day{sd} == ymd, ""); // and back
    }

    {
    year_month_day ymd{year{1939}, month{11}, day{29}};
    sys_days sd{ymd};

    assert( sd.time_since_epoch() == days{-(10957+34)});
    assert( year_month_day{sd} == ymd); // and back
    }

//  These two tests check the wording for LWG 3206
    {
    constexpr year_month_day ymd{year{1971}, month{1}, day{0}}; // bad day
    static_assert(!ymd.ok(),         "");
    static_assert( ymd.year().ok(),  "");
    static_assert( ymd.month().ok(), "");
    static_assert(!ymd.day().ok(),   "");
    constexpr sys_days sd{ymd};
    static_assert(sd.time_since_epoch() == days{364}, "");
    static_assert(sd == sys_days{ymd.year()/ymd.month()/day{1}} + (ymd.day() - day{1}), "");
    }

    {
    constexpr year_month_day ymd{year{1970}, month{12}, day{32}}; // bad day
    static_assert(!ymd.ok(),         "");
    static_assert( ymd.year().ok(),  "");
    static_assert( ymd.month().ok(), "");
    static_assert(!ymd.day().ok(),   "");
    constexpr sys_days sd{ymd};
    static_assert(sd.time_since_epoch() == days{365}, "");
    static_assert(sd == sys_days{ymd.year()/ymd.month()/day{1}} + (ymd.day() - day{1}), "");
    }

    return 0;
}
