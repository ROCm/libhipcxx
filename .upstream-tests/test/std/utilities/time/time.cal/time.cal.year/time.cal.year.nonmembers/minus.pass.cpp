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
// class year;

// constexpr year operator-(const year& x, const years& y) noexcept;
//   Returns: x + -y.
//
// constexpr years operator-(const year& x, const year& y) noexcept;
//   Returns: If x.ok() == true and y.ok() == true, returns a value m in the range
//   [years{0}, years{11}] satisfying y + m == x.
//   Otherwise the value returned is unspecified.
//   [Example: January - February == years{11}. —end example]

extern "C" int printf(const char *, ...);

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

template <typename Y, typename Ys>
__host__ __device__
constexpr bool testConstexpr()
{
    Y y{2313};
    Ys offset{1006};
    if (y - offset != Y{1307}) return false;
    if (y - Y{1307} != offset) return false;
    return true;
}

int main(int, char**)
{
    using year  = hip::std::chrono::year;
    using years = hip::std::chrono::years;

    ASSERT_NOEXCEPT(                 std::declval<year>() - std::declval<years>());
    ASSERT_SAME_TYPE(year , decltype(hip::std::declval<year>() - std::declval<years>()));

    ASSERT_NOEXCEPT(                 std::declval<year>() - std::declval<year>());
    ASSERT_SAME_TYPE(years, decltype(hip::std::declval<year>() - std::declval<year>()));

    static_assert(testConstexpr<year, years>(), "");

    year y{1223};
    for (int i = 1100; i <= 1110; ++i)
    {
        year  y1 = y - years{i};
        years ys1 = y - year{i};
        assert(static_cast<int>(y1) == 1223 - i);
        assert(ys1.count()          == 1223 - i);
    }

  return 0;
}
