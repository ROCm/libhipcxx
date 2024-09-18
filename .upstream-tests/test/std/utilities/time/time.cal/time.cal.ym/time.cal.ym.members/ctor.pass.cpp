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
// class year_month;

//            year_month() = default;
//  constexpr year_month(const chrono::year& y, const chrono::month& m) noexcept;
//
//  Effects:  Constructs an object of type year_month by initializing y_ with y, and m_ with m.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year       = hip::std::chrono::year;
    using month      = hip::std::chrono::month;
    using year_month = hip::std::chrono::year_month;

    ASSERT_NOEXCEPT(year_month{});
    ASSERT_NOEXCEPT(year_month{year{1}, month{1}});

    constexpr year_month ym0{};
    static_assert( ym0.year()  == year{},  "");
    static_assert( ym0.month() == month{}, "");
    static_assert(!ym0.ok(),               "");

    constexpr year_month ym1{year{2018}, hip::std::chrono::January};
    static_assert( ym1.year()  == year{2018},           "");
    static_assert( ym1.month() == hip::std::chrono::January, "");
    static_assert( ym1.ok(),                            "");

    constexpr year_month ym2{year{2018}, month{}};
    static_assert( ym2.year()  == year{2018}, "");
    static_assert( ym2.month() == month{},    "");
    static_assert(!ym2.ok(),                  "");

  return 0;
}
