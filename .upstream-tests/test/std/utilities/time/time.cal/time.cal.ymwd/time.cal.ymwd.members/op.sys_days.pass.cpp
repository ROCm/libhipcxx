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

// constexpr operator sys_days() const noexcept;
//
// Returns: If y_.ok() && m_.ok() && wdi_.weekday().ok(), returns a
//    sys_days that represents the date (index() - 1) * 7 days after the first
//    weekday() of year()/month(). If index() is 0 the returned sys_days
//    represents the date 7 days prior to the first weekday() of
//    year()/month(). Otherwise the returned value is unspecified.
//

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
    using year               = hip::std::chrono::year;
    using month              = hip::std::chrono::month;
    using weekday_indexed    = hip::std::chrono::weekday_indexed;
    using sys_days           = hip::std::chrono::sys_days;
    using days               = hip::std::chrono::days;
    using year_month_weekday = hip::std::chrono::year_month_weekday;

    ASSERT_NOEXCEPT(sys_days(hip::std::declval<year_month_weekday>()));

    {
    constexpr year_month_weekday ymwd{year{1970}, month{1}, weekday_indexed{hip::std::chrono::Thursday, 1}};
    constexpr sys_days sd{ymwd};

    static_assert( sd.time_since_epoch() == days{0}, "");
    static_assert( year_month_weekday{sd} == ymwd, ""); // and back
    }

    {
    constexpr year_month_weekday ymwd{year{2000}, month{2}, weekday_indexed{hip::std::chrono::Wednesday, 1}};
    constexpr sys_days sd{ymwd};

    static_assert( sd.time_since_epoch() == days{10957+32}, "");
    static_assert( year_month_weekday{sd} == ymwd, ""); // and back
    }

//  There's one more leap day between 1/1/40 and 1/1/70
//  when compared to 1/1/70 -> 1/1/2000
    {
    constexpr year_month_weekday ymwd{year{1940}, month{1},weekday_indexed{hip::std::chrono::Tuesday, 1}};
    constexpr sys_days sd{ymwd};

    static_assert( sd.time_since_epoch() == days{-10957}, "");
    static_assert( year_month_weekday{sd} == ymwd, ""); // and back
    }

    {
    year_month_weekday ymwd{year{1939}, month{11}, weekday_indexed{hip::std::chrono::Wednesday, 5}};
    sys_days sd{ymwd};

    assert( sd.time_since_epoch() == days{-(10957+34)});
    assert( year_month_weekday{sd} == ymwd); // and back
    }

    return 0;
}
