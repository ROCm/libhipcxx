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
// class year_month_day_last;

// constexpr chrono::day day() const noexcept;
//  Returns: wd_

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year                = hip::std::chrono::year;
    using month               = hip::std::chrono::month;
    using day                 = hip::std::chrono::day;
    using month_day_last      = hip::std::chrono::month_day_last;
    using year_month_day_last = hip::std::chrono::year_month_day_last;

    ASSERT_NOEXCEPT(               std::declval<const year_month_day_last>().day());
    ASSERT_SAME_TYPE(day, decltype(hip::std::declval<const year_month_day_last>().day()));

//  Some months have a 31st
    static_assert( year_month_day_last{year{2020}, month_day_last{month{ 1}}}.day() == day{31}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{ 2}}}.day() == day{29}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{ 3}}}.day() == day{31}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{ 4}}}.day() == day{30}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{ 5}}}.day() == day{31}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{ 6}}}.day() == day{30}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{ 7}}}.day() == day{31}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{ 8}}}.day() == day{31}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{ 9}}}.day() == day{30}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{10}}}.day() == day{31}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{11}}}.day() == day{30}, "");
    static_assert( year_month_day_last{year{2020}, month_day_last{month{12}}}.day() == day{31}, "");

    assert((year_month_day_last{year{2019}, month_day_last{month{ 2}}}.day() == day{28}));
    assert((year_month_day_last{year{2020}, month_day_last{month{ 2}}}.day() == day{29}));
    assert((year_month_day_last{year{2021}, month_day_last{month{ 2}}}.day() == day{28}));

  return 0;
}