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

// constexpr year_month_day operator-(const year_month_day& ymd, const years& dy) noexcept;
//    Returns: ymd + (-dy)


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

__host__ __device__
constexpr bool test_constexpr ()
{
    hip::std::chrono::year_month_day ym0{hip::std::chrono::year{1234}, hip::std::chrono::January, hip::std::chrono::day{12}};
    hip::std::chrono::year_month_day ym1 = ym0 - hip::std::chrono::years{10};
    return
        ym1.year()  == hip::std::chrono::year{1234-10}
     && ym1.month() == hip::std::chrono::January
     && ym1.day()   == hip::std::chrono::day{12}
        ;
}

int main(int, char**)
{
    using year           = hip::std::chrono::year;
    using month          = hip::std::chrono::month;
    using day            = hip::std::chrono::day;
    using year_month_day = hip::std::chrono::year_month_day;
    using years          = hip::std::chrono::years;

    ASSERT_NOEXCEPT(                          std::declval<year_month_day>() - std::declval<years>());
    ASSERT_SAME_TYPE(year_month_day, decltype(hip::std::declval<year_month_day>() - std::declval<years>()));

    constexpr month January = hip::std::chrono::January;

    static_assert(test_constexpr(), "");

    year_month_day ym{year{1234}, January, day{10}};
    for (int i = 0; i <= 10; ++i)
    {
        year_month_day ym1 = ym - years{i};
        assert(static_cast<int>(ym1.year()) == 1234 - i);
        assert(ym1.month() == January);
        assert(ym1.day() == day{10});
    }

  return 0;
}
