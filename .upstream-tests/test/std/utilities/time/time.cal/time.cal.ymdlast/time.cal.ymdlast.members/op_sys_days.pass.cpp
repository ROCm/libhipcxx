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
// class year_month_day_last;

// constexpr operator sys_days() const noexcept;
//  Returns: sys_days{year()/month()/day()}.

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year                = hip::std::chrono::year;
    using month_day_last      = hip::std::chrono::month_day_last;
    using year_month_day_last = hip::std::chrono::year_month_day_last;
    using sys_days            = hip::std::chrono::sys_days;
    using days                = hip::std::chrono::days;

    ASSERT_NOEXCEPT(                    static_cast<sys_days>(hip::std::declval<const year_month_day_last>()));
    ASSERT_SAME_TYPE(sys_days, decltype(static_cast<sys_days>(hip::std::declval<const year_month_day_last>())));

    auto constexpr January = hip::std::chrono::January;
    auto constexpr November = hip::std::chrono::November;

    { // Last day in Jan 1970 was the 31st
    constexpr year_month_day_last ymdl{year{1970}, month_day_last{January}};
    constexpr sys_days sd{ymdl};
    
    static_assert(sd.time_since_epoch() == days{30}, "");
    }

    {
    constexpr year_month_day_last ymdl{year{2000}, month_day_last{January}};
    constexpr sys_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{10957+30}, "");
    }

    {
    constexpr year_month_day_last ymdl{year{1940}, month_day_last{January}};
    constexpr sys_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{-10957+29}, "");
    }

    {
    year_month_day_last ymdl{year{1939}, month_day_last{November}};
    sys_days sd{ymdl};

    assert(sd.time_since_epoch() == days{-(10957+33)});
    }

  return 0;
}
