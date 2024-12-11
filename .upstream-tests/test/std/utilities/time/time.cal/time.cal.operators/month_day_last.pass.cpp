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
// class month_day_last;

// constexpr month_day_last
//   operator/(const month& m, last_spec) noexcept;
// Returns: month_day_last{m}.
//
// constexpr month_day_last
//   operator/(int m, last_spec) noexcept;
// Returns: month(m) / last.
//
// constexpr month_day_last
//   operator/(last_spec, const month& m) noexcept;
// Returns: m / last.
//
// constexpr month_day_last
//   operator/(last_spec, int m) noexcept;
// Returns: month(m) / last.
//
//
// [Note: A month_day_last object can be constructed using the expression m/last or last/m,
//     where m is an expression of type month. — end note]
// [Example:
//     constexpr auto mdl = February/last; // mdl is the last day of February of an as yet unspecified year
//     static_assert(mdl.month() == February);
// --end example]

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using month          = hip::std::chrono::month;
    using month_day_last = hip::std::chrono::month_day_last;

    constexpr month February = hip::std::chrono::February;
    constexpr hip::std::chrono::last_spec last = hip::std::chrono::last;

    ASSERT_SAME_TYPE(month_day_last, decltype(last/February));
    ASSERT_SAME_TYPE(month_day_last, decltype(February/last));

//  Run the example
    {
    constexpr auto mdl = February/hip::std::chrono::last;
    static_assert(mdl.month() == February, "");
    }

    { // operator/(const month& m, last_spec) and switched
        ASSERT_NOEXCEPT (                         last/February);
        ASSERT_SAME_TYPE(month_day_last, decltype(last/February));
        ASSERT_NOEXCEPT (                         February/last);
        ASSERT_SAME_TYPE(month_day_last, decltype(February/last));

        static_assert((last/February).month() == February, "");
        static_assert((February/last).month() == February, "");

        for (unsigned i = 1; i < 12; ++i)
        {
            month m{i};
            month_day_last mdl1 = last/m;
            month_day_last mdl2 = m/last;
            assert(mdl1.month() == m);
            assert(mdl2.month() == m);
            assert(mdl1 == mdl2);
        }
    }

    { // operator/(int, last_spec) and switched
        ASSERT_NOEXCEPT (                         last/2);
        ASSERT_SAME_TYPE(month_day_last, decltype(last/2));
        ASSERT_NOEXCEPT (                         2/last);
        ASSERT_SAME_TYPE(month_day_last, decltype(2/last));

        static_assert((last/2).month() == February, "");
        static_assert((2/last).month() == February, "");

        for (unsigned i = 1; i < 12; ++i)
        {
            month m{i};
            month_day_last mdl1 = last/i;
            month_day_last mdl2 = i/last;
            assert(mdl1.month() == m);
            assert(mdl2.month() == m);
            assert(mdl1 == mdl2);
        }
    }


  return 0;
}
