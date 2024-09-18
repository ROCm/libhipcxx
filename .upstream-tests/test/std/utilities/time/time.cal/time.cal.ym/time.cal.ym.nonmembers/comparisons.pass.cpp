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
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month;

// constexpr bool operator==(const year_month& x, const year_month& y) noexcept;
//   Returns: x.year() == y.year() && x.month() == y.month().
//
// constexpr bool operator< (const year_month& x, const year_month& y) noexcept;
//   Returns:
//      If x.year() < y.year() returns true.
//      Otherwise, if x.year() > y.year() returns false.
//      Otherwise, returns x.month() < y.month().

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using year       = hip::std::chrono::year;
    using month      = hip::std::chrono::month;
    using year_month = hip::std::chrono::year_month;

    AssertComparisons6AreNoexcept<year_month>();
    AssertComparisons6ReturnBool<year_month>();

    auto constexpr January = hip::std::chrono::January;
    auto constexpr February = hip::std::chrono::February;

    static_assert( testComparisons6(
        year_month{year{1234}, January},
        year_month{year{1234}, January},
        true, false), "");

    static_assert( testComparisons6(
        year_month{year{1234}, January},
        year_month{year{1234}, February},
        false, true), "");

    static_assert( testComparisons6(
        year_month{year{1234}, January},
        year_month{year{1235}, January},
        false, true), "");

//  same year, different months
    for (unsigned i = 1; i < 12; ++i)
        for (unsigned j = 1; j < 12; ++j)
            assert((testComparisons6(
                year_month{year{1234}, month{i}},
                year_month{year{1234}, month{j}},
                i == j, i < j )));

//  same month, different years
    for (int i = 1000; i < 20; ++i)
        for (int j = 1000; j < 20; ++j)
        assert((testComparisons6(
            year_month{year{i}, January},
            year_month{year{j}, January},
            i == j, i < j )));

  return 0;
}
