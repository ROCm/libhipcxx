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
// class month_weekday;

// constexpr bool ok() const noexcept;
//  Returns: m_.ok() && wdi_.ok().

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using month_weekday   = hip::std::chrono::month_weekday;
    using month           = hip::std::chrono::month;
    using weekday         = hip::std::chrono::weekday;
    using weekday_indexed = hip::std::chrono::weekday_indexed;

    constexpr weekday Sunday = hip::std::chrono::Sunday;

    ASSERT_NOEXCEPT(                std::declval<const month_weekday>().ok());
    ASSERT_SAME_TYPE(bool, decltype(hip::std::declval<const month_weekday>().ok()));

    static_assert(!month_weekday{month{}, weekday_indexed{}}.ok(),                   "");
    static_assert( month_weekday{hip::std::chrono::May, weekday_indexed{Sunday, 2}}.ok(), "");

    assert(!(month_weekday(hip::std::chrono::April, weekday_indexed{Sunday, 0}).ok()));
    assert( (month_weekday{hip::std::chrono::March, weekday_indexed{Sunday, 1}}.ok()));

    for (unsigned i = 1; i <= 12; ++i)
        for (unsigned j = 0; j <= 6; ++j)
        {
            month_weekday mwd{month{i}, weekday_indexed{Sunday, j}};
            assert(mwd.ok() == (j >= 1 && j <= 5));
        }

//  If the month is not ok, all the weekday_indexed are bad
    for (unsigned i = 1; i <= 10; ++i)
        assert(!(month_weekday{month{13}, weekday_indexed{Sunday, i}}.ok()));

  return 0;
}
