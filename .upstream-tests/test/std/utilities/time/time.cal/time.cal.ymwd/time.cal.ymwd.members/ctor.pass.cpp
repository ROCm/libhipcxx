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

//  year_month_weekday() = default;
//  constexpr year_month_weekday(const chrono::year& y, const chrono::month& m,
//                               const chrono::weekday_indexed& wdi) noexcept;
//
//  Effects:  Constructs an object of type year_month_weekday by initializing
//                y_ with y, m_ with m, and wdi_ with wdi.
//
//  constexpr chrono::year                       year() const noexcept;
//  constexpr chrono::month                     month() const noexcept;
//  constexpr chrono::weekday                 weekday() const noexcept;
//  constexpr unsigned                          index() const noexcept;
//  constexpr chrono::weekday_indexed weekday_indexed() const noexcept;
//  constexpr bool                                 ok() const noexcept;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year               = hip::std::chrono::year;
    using month              = hip::std::chrono::month;
    using weekday            = hip::std::chrono::weekday;
    using weekday_indexed    = hip::std::chrono::weekday_indexed;
    using year_month_weekday = hip::std::chrono::year_month_weekday;

    constexpr month January = hip::std::chrono::January;
    constexpr weekday Tuesday = hip::std::chrono::Tuesday;

    ASSERT_NOEXCEPT(year_month_weekday{});
    ASSERT_NOEXCEPT(year_month_weekday{year{1}, month{1}, weekday_indexed{Tuesday, 1}});

    constexpr year_month_weekday ym0{};
    static_assert( ym0.year()            == year{},            "");
    static_assert( ym0.month()           == month{},           "");
    static_assert( ym0.weekday()         == weekday{},         "");
    static_assert( ym0.index()           == 0,                 "");
    static_assert( ym0.weekday_indexed() == weekday_indexed{}, "");
    static_assert(!ym0.ok(),                                   "");

    constexpr year_month_weekday ym1{year{2019}, January, weekday_indexed{Tuesday, 1}};
    static_assert( ym1.year()            == year{2019},                  "");
    static_assert( ym1.month()           == January,                     "");
    static_assert( ym1.weekday()         == Tuesday,                     "");
    static_assert( ym1.index()           == 1,                           "");
    static_assert( ym1.weekday_indexed() == weekday_indexed{Tuesday, 1}, "");
    static_assert( ym1.ok(),                                             "");


  return 0;
}
