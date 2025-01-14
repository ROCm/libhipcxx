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

// constexpr year_month operator/(const year& y, const month& m) noexcept;
//   Returns: {y, m}.
//
// constexpr year_month operator/(const year& y, int m) noexcept;
//   Returns: y / month(m).



#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using month      = hip::std::chrono::month;
    using year       = hip::std::chrono::year;
    using year_month = hip::std::chrono::year_month;

    constexpr month February = hip::std::chrono::February;

    { // operator/(const year& y, const month& m)
        ASSERT_NOEXCEPT (                     year{2018}/February);
        ASSERT_SAME_TYPE(year_month, decltype(year{2018}/February));

        static_assert((year{2018}/February).year()  == year{2018}, "");
        static_assert((year{2018}/February).month() == month{2},   "");
        for (int i = 1000; i <= 1030; ++i)
            for (unsigned j = 1; j <= 12; ++j)
            {
                year_month ym = year{i}/month{j};
                assert(static_cast<int>(ym.year())       == i);
                assert(static_cast<unsigned>(ym.month()) == j);
            }
    }


    { // operator/(const year& y, const int m)
        ASSERT_NOEXCEPT (                     year{2018}/4);
        ASSERT_SAME_TYPE(year_month, decltype(year{2018}/4));

        static_assert((year{2018}/2).year()  == year{2018}, "");
        static_assert((year{2018}/2).month() == month{2},   "");

        for (int i = 1000; i <= 1030; ++i)
            for (unsigned j = 1; j <= 12; ++j)
            {
                year_month ym = year{i}/j;
                assert(static_cast<int>(ym.year())       == i);
                assert(static_cast<unsigned>(ym.month()) == j);
            }
    }

  return 0;
}
