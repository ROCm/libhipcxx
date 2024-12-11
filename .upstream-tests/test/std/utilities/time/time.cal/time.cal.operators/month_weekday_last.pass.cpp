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

// constexpr month_weekday_last
//   operator/(const month& m, const weekday_last& wdl) noexcept;
// Returns: {m, wdl}.
//
// constexpr month_weekday_last
//   operator/(int m, const weekday_last& wdl) noexcept;
// Returns: month(m) / wdl.
//
// constexpr month_weekday_last
//   operator/(const weekday_last& wdl, const month& m) noexcept;
// Returns: m / wdl.
//
// constexpr month_weekday_last
//   operator/(const weekday_last& wdl, int m) noexcept;
// Returns: month(m) / wdl.




#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using month_weekday      = hip::std::chrono::month_weekday;
    using month              = hip::std::chrono::month;
    using weekday            = hip::std::chrono::weekday;
    using weekday_last       = hip::std::chrono::weekday_last;
    using month_weekday_last = hip::std::chrono::month_weekday_last;

    constexpr weekday Tuesday = hip::std::chrono::Tuesday;
    constexpr month February = hip::std::chrono::February;
    constexpr hip::std::chrono::last_spec last = hip::std::chrono::last;

    { // operator/(const month& m, const weekday_last& wdi) (and switched)
        ASSERT_NOEXCEPT (February/Tuesday[last]);
        ASSERT_SAME_TYPE(month_weekday_last, decltype(February/Tuesday[last]));
        ASSERT_NOEXCEPT (Tuesday[last]/February);
        ASSERT_SAME_TYPE(month_weekday_last, decltype(Tuesday[last]/February));

    //  Run the example
        {
        constexpr month_weekday_last wdi = February/Tuesday[last];
        static_assert(wdi.month()        == February,      "");
        static_assert(wdi.weekday_last() == Tuesday[last], "");
        }

        for (int i = 1; i <= 12; ++i)
            for (unsigned j = 0; j <= 6; ++j)
            {
                month m(i);
                weekday_last wdi = weekday{j}[last];
                month_weekday_last mwd1 = m/wdi;
                month_weekday_last mwd2 = wdi/m;
                assert(mwd1.month() == m);
                assert(mwd1.weekday_last() == wdi);
                assert(mwd2.month() == m);
                assert(mwd2.weekday_last() == wdi);
                assert(mwd1 == mwd2);
            }
    }


    { // operator/(int m, const weekday_last& wdi) (and switched)
        ASSERT_NOEXCEPT (2/Tuesday[2]);
        ASSERT_SAME_TYPE(month_weekday_last, decltype(2/Tuesday[last]));
        ASSERT_NOEXCEPT (Tuesday[2]/2);
        ASSERT_SAME_TYPE(month_weekday_last, decltype(Tuesday[last]/2));

    //  Run the example
        {
        constexpr month_weekday wdi = 2/Tuesday[3];
        static_assert(wdi.month()           == February,   "");
        static_assert(wdi.weekday_indexed() == Tuesday[3], "");
        }

        for (int i = 1; i <= 12; ++i)
            for (unsigned j = 0; j <= 6; ++j)
            {
                weekday_last wdi = weekday{j}[last];
                month_weekday_last mwd1 = i/wdi;
                month_weekday_last mwd2 = wdi/i;
                assert(mwd1.month() == month(i));
                assert(mwd1.weekday_last() == wdi);
                assert(mwd2.month() == month(i));
                assert(mwd2.weekday_last() == wdi);
                assert(mwd1 == mwd2);
            }
    }

  return 0;
}
