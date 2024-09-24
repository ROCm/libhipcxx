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
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month_weekday;

//  explicit constexpr year_month_weekday(const local_days& dp) noexcept;
//
//
//  Effects:  Constructs an object of type year_month_weekday that corresponds
//                to the date represented by dp
//
//  Remarks: Equivalent to constructing with sys_days{dp.time_since_epoch()}.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr chrono::day     day() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

// MSVC warns about unsigned/signed comparisons and addition/subtraction
// Silence these warnings, but not the ones within the header itself.
#if defined(_MSC_VER)
# pragma warning( disable: 4307 )
# pragma warning( disable: 4308 )
#endif

int main(int, char**)
{
    using year               = hip::std::chrono::year;
    using days               = hip::std::chrono::days;
    using local_days         = hip::std::chrono::local_days;
    using weekday_indexed    = hip::std::chrono::weekday_indexed;
    using year_month_weekday = hip::std::chrono::year_month_weekday;

    ASSERT_NOEXCEPT(year_month_weekday{hip::std::declval<const local_days>()});

    auto constexpr January = hip::std::chrono::January;

    {
    constexpr local_days sd{}; // 1-Jan-1970 was a Thursday
    constexpr year_month_weekday ymwd{sd};
    auto constexpr Thursday = hip::std::chrono::Thursday;

    static_assert( ymwd.ok(),                                                            "");
    static_assert( ymwd.year()            == year{1970},                                 "");
    static_assert( ymwd.month()           == January,                       "");
    static_assert( ymwd.weekday()         == Thursday,                      "");
    static_assert( ymwd.index()           == 1,                                          "");
    static_assert( ymwd.weekday_indexed() == weekday_indexed{Thursday, 1},  "");
    static_assert( ymwd                   == year_month_weekday{local_days{ymwd}},       ""); // round trip
    }

    {
    constexpr local_days sd{days{10957+32}}; // 2-Feb-2000 was a Wednesday
    constexpr year_month_weekday ymwd{sd};

    auto constexpr February = hip::std::chrono::February;
    auto constexpr Wednesday = hip::std::chrono::Wednesday;

    static_assert( ymwd.ok(),                                                            "");
    static_assert( ymwd.year()            == year{2000},                                 "");
    static_assert( ymwd.month()           == February,                      "");
    static_assert( ymwd.weekday()         == Wednesday,                     "");
    static_assert( ymwd.index()           == 1,                                          "");
    static_assert( ymwd.weekday_indexed() == weekday_indexed{Wednesday, 1}, "");
    static_assert( ymwd                   == year_month_weekday{local_days{ymwd}},       ""); // round trip
    }


    {
    constexpr local_days sd{days{-10957}}; // 2-Jan-1940 was a Tuesday
    constexpr year_month_weekday ymwd{sd};

    auto constexpr Tuesday = hip::std::chrono::Tuesday;

    static_assert( ymwd.ok(),                                                            "");
    static_assert( ymwd.year()            == year{1940},                                 "");
    static_assert( ymwd.month()           == January,                       "");
    static_assert( ymwd.weekday()         == Tuesday,                       "");
    static_assert( ymwd.index()           == 1,                                          "");
    static_assert( ymwd.weekday_indexed() == weekday_indexed{Tuesday, 1},   "");
    static_assert( ymwd                   == year_month_weekday{local_days{ymwd}},       ""); // round trip
    }

    {
    local_days sd{days{-(10957+34)}}; // 29-Nov-1939 was a Wednesday
    year_month_weekday ymwd{sd};
    auto constexpr November = hip::std::chrono::November;
    auto constexpr Wednesday = hip::std::chrono::Wednesday;

    assert( ymwd.ok());
    assert( ymwd.year()            == year{1939});
    assert( ymwd.month()           == November);
    assert( ymwd.weekday()         == Wednesday);
    assert( ymwd.index()           == 5);
    assert((ymwd.weekday_indexed() == weekday_indexed{Wednesday, 5}));
    assert( ymwd                   == year_month_weekday{local_days{ymwd}}); // round trip
    }

  return 0;
}
