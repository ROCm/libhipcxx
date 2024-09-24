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

// constexpr month_day
//   operator/(const month& m, const day& d) noexcept;
// Returns: {m, d}.
//
// constexpr month_day
//   operator/(const day& d, const month& m) noexcept;
// Returns: m / d.

// constexpr month_day
//   operator/(const month& m, int d) noexcept;
// Returns: m / day(d).
//
// constexpr month_day
//   operator/(int m, const day& d) noexcept;
// Returns: month(m) / d.
//
// constexpr month_day
//   operator/(const day& d, int m) noexcept;
// Returns: month(m) / d.


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using month_day = hip::std::chrono::month_day;
    using month     = hip::std::chrono::month;
    using day       = hip::std::chrono::day;

    constexpr month February = hip::std::chrono::February;

    { // operator/(const month& m, const day& d) (and switched)
        ASSERT_NOEXCEPT (                    February/day{1});
        ASSERT_SAME_TYPE(month_day, decltype(February/day{1}));
        ASSERT_NOEXCEPT (                    day{1}/February);
        ASSERT_SAME_TYPE(month_day, decltype(day{1}/February));

        for (int i = 1; i <= 12; ++i)
            for (unsigned j = 0; j <= 30; ++j)
            {
                month m(i);
                day   d{j};
                month_day md1 = m/d;
                month_day md2 = d/m;
                assert(md1.month() == m);
                assert(md1.day()   == d);
                assert(md2.month() == m);
                assert(md2.day()   == d);
                assert(md1 == md2);
            }
    }


    { // operator/(const month& m, int d) (NOT switched)
        ASSERT_NOEXCEPT (                    February/2);
        ASSERT_SAME_TYPE(month_day, decltype(February/2));

        for (int i = 1; i <= 12; ++i)
            for (unsigned j = 0; j <= 30; ++j)
            {
                month m(i);
                day   d(j);
                month_day md1 = m/j;
                assert(md1.month() == m);
                assert(md1.day()   == d);
            }
    }


    { // operator/(const day& d, int m) (and switched)
        ASSERT_NOEXCEPT (                    day{2}/2);
        ASSERT_SAME_TYPE(month_day, decltype(day{2}/2));
        ASSERT_NOEXCEPT (                    2/day{2});
        ASSERT_SAME_TYPE(month_day, decltype(2/day{2}));

        for (int i = 1; i <= 12; ++i)
            for (unsigned j = 0; j <= 30; ++j)
            {
                month m(i);
                day   d(j);
                month_day md1 = d/i;
                month_day md2 = i/d;
                assert(md1.month() == m);
                assert(md1.day()   == d);
                assert(md2.month() == m);
                assert(md2.day()   == d);
                assert(md1 == md2);
            }
    }

  return 0;
}
