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

//  constexpr year_month_weekday_last(const chrono::year& y, const chrono::month& m,
//                               const chrono::weekday_last& wdl) noexcept;
//
//  Effects:  Constructs an object of type year_month_weekday_last by initializing
//                y_ with y, m_ with m, and wdl_ with wdl.
//
//  constexpr chrono::year                 year() const noexcept;
//  constexpr chrono::month               month() const noexcept;
//  constexpr chrono::weekday           weekday() const noexcept;
//  constexpr chrono::weekday_last weekday_last() const noexcept;
//  constexpr bool                           ok() const noexcept;

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

    constexpr month January = hip::std::chrono::January;
    constexpr weekday Tuesday = hip::std::chrono::Tuesday;

    ASSERT_NOEXCEPT(year_month_weekday_last{year{1}, month{1}, weekday_last{Tuesday}});

    constexpr year_month_weekday_last ym1{year{2019}, January, weekday_last{Tuesday}};
    static_assert( ym1.year()         == year{2019},            "");
    static_assert( ym1.month()        == January,               "");
    static_assert( ym1.weekday()      == Tuesday,               "");
    static_assert( ym1.weekday_last() == weekday_last{Tuesday}, "");
    static_assert( ym1.ok(),                                    "");


  return 0;
}
