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
// class year_month_weekday_last;

// constexpr operator sys_days() const noexcept;
//  Returns: If ok() == true, returns a sys_days that represents the last weekday()
//             of year()/month(). Otherwise the returned value is unspecified.

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"


int main(int, char**)
{
    using year                    = hip::std::chrono::year;
    using month                   = hip::std::chrono::month;
    using year_month_weekday_last = hip::std::chrono::year_month_weekday_last;
    using sys_days                = hip::std::chrono::sys_days;
    using days                    = hip::std::chrono::days;
    using weekday                 = hip::std::chrono::weekday;
    using weekday_last            = hip::std::chrono::weekday_last;

    ASSERT_NOEXCEPT(                    static_cast<sys_days>(hip::std::declval<const year_month_weekday_last>()));
    ASSERT_SAME_TYPE(sys_days, decltype(static_cast<sys_days>(hip::std::declval<const year_month_weekday_last>())));

    auto constexpr January = hip::std::chrono::January;
    auto constexpr Tuesday = hip::std::chrono::Tuesday;

    { // Last Tuesday in Jan 1970 was the 27th
    constexpr year_month_weekday_last ymwdl{year{1970}, January, weekday_last{Tuesday}};
    constexpr sys_days sd{ymwdl};

    static_assert(sd.time_since_epoch() == days{26}, "");
    }

    { // Last Tuesday in Jan 2000 was the 25th
    constexpr year_month_weekday_last ymwdl{year{2000}, January, weekday_last{Tuesday}};
    constexpr sys_days sd{ymwdl};

    static_assert(sd.time_since_epoch() == days{10957+24}, "");
    }

    { // Last Tuesday in Jan 1940 was the 30th
    constexpr year_month_weekday_last ymwdl{year{1940}, January, weekday_last{Tuesday}};
    constexpr sys_days sd{ymwdl};

    static_assert(sd.time_since_epoch() == days{-10958+29}, "");
    }

    { // Last Tuesday in Nov 1939 was the 28th
    year_month_weekday_last ymdl{year{1939}, hip::std::chrono::November, weekday_last{Tuesday}};
    sys_days sd{ymdl};

    assert(sd.time_since_epoch() == days{-(10957+35)});
    }

  return 0;
}
