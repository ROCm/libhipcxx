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
// class month_day;

// constexpr bool ok() const noexcept;
//  Returns: true if m_.ok() is true, 1d <= d_, and d_ is less than or equal to the
//    number of days in month m_; otherwise returns false.
//  When m_ == February, the number of days is considered to be 29.

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using day       = hip::std::chrono::day;
    using month     = hip::std::chrono::month;
    using month_day = hip::std::chrono::month_day;

    ASSERT_NOEXCEPT(                std::declval<const month_day>().ok());
    ASSERT_SAME_TYPE(bool, decltype(hip::std::declval<const month_day>().ok()));

    static_assert(!month_day{}.ok(),                         "");
    static_assert( month_day{hip::std::chrono::May, day{2}}.ok(), "");

    assert(!(month_day(hip::std::chrono::April, day{0}).ok()));

    assert( (month_day{hip::std::chrono::March, day{1}}.ok()));
    for (unsigned i = 1; i <= 12; ++i)
    {
        const bool is31 = i == 1 || i == 3 || i == 5 || i == 7 || i == 8 || i == 10 || i == 12;
        assert(!(month_day{month{i}, day{ 0}}.ok()));
        assert( (month_day{month{i}, day{ 1}}.ok()));
        assert( (month_day{month{i}, day{10}}.ok()));
        assert( (month_day{month{i}, day{29}}.ok()));
        assert( (month_day{month{i}, day{30}}.ok()) == (i != 2));
        assert( (month_day{month{i}, day{31}}.ok()) == is31);
        assert(!(month_day{month{i}, day{32}}.ok()));
    }

//  If the month is not ok, all the days are bad
    for (unsigned i = 1; i <= 35; ++i)
        assert(!(month_day{month{13}, day{i}}.ok()));

  return 0;
}
