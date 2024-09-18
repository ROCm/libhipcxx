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
// class month_weekday_last;

// constexpr bool operator==(const month_weekday_last& x, const month_weekday_last& y) noexcept;
//   Returns: x.month() == y.month()
//
// constexpr bool operator< (const month_weekday_last& x, const month_weekday_last& y) noexcept;
//   Returns: x.month() < y.month()


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using month              = hip::std::chrono::month;
    using weekday_last       = hip::std::chrono::weekday_last;
    using weekday            = hip::std::chrono::weekday;
    using month_weekday_last = hip::std::chrono::month_weekday_last;

    constexpr month January = hip::std::chrono::January;
    constexpr weekday Tuesday = hip::std::chrono::Tuesday;
    constexpr weekday Wednesday = hip::std::chrono::Wednesday;

    AssertComparisons2AreNoexcept<month_weekday_last>();
    AssertComparisons2ReturnBool<month_weekday_last>();

    static_assert( testComparisons2(
        month_weekday_last{hip::std::chrono::January, weekday_last{Tuesday}},
        month_weekday_last{hip::std::chrono::January, weekday_last{Tuesday}},
        true), "");

    static_assert( testComparisons2(
        month_weekday_last{hip::std::chrono::January, weekday_last{Tuesday}},
        month_weekday_last{hip::std::chrono::January, weekday_last{Wednesday}},
        false), "");

//  vary the months
    for (unsigned i = 1; i < 12; ++i)
        for (unsigned j = 1; j < 12; ++j)
            assert((testComparisons2(
                month_weekday_last{month{i}, weekday_last{Tuesday}},
                month_weekday_last{month{j}, weekday_last{Tuesday}},
            i == j)));

//  vary the weekday
    for (unsigned i = 0; i < 6; ++i)
        for (unsigned j = 0; j < 6; ++j)
            assert((testComparisons2(
                month_weekday_last{January, weekday_last{weekday{i}}},
                month_weekday_last{January, weekday_last{weekday{j}}},
            i == j)));

//  both different
        assert((testComparisons2(
            month_weekday_last{month{1}, weekday_last{weekday{1}}},
            month_weekday_last{month{2}, weekday_last{weekday{2}}},
        false)));

  return 0;
}
