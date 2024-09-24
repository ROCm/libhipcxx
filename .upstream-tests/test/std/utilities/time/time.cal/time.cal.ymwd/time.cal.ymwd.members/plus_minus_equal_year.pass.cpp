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
// XFAIL: gcc-4.8, gcc-5, gcc-6
// gcc before gcc-7 fails with an internal compiler error

// <chrono>
// class year_month_weekday;

// constexpr year_month_weekday& operator+=(const years& d) noexcept;
// constexpr year_month_weekday& operator-=(const years& d) noexcept;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

template <typename D, typename Ds>
__host__ __device__
constexpr bool testConstexpr(D d1)
{
    if (static_cast<int>((d1          ).year()) !=  1) return false;
    if (static_cast<int>((d1 += Ds{ 1}).year()) !=  2) return false;
    if (static_cast<int>((d1 += Ds{ 2}).year()) !=  4) return false;
    if (static_cast<int>((d1 += Ds{12}).year()) != 16) return false;
    if (static_cast<int>((d1 -= Ds{ 1}).year()) != 15) return false;
    if (static_cast<int>((d1 -= Ds{ 2}).year()) != 13) return false;
    if (static_cast<int>((d1 -= Ds{12}).year()) !=  1) return false;
    return true;
}

int main(int, char**)
{
    using year               = hip::std::chrono::year;
    using month              = hip::std::chrono::month;
    using weekday            = hip::std::chrono::weekday;
    using weekday_indexed    = hip::std::chrono::weekday_indexed;
    using year_month_weekday = hip::std::chrono::year_month_weekday;
    using years              = hip::std::chrono::years;

    ASSERT_NOEXCEPT(                               std::declval<year_month_weekday&>() += std::declval<years>());
    ASSERT_SAME_TYPE(year_month_weekday&, decltype(hip::std::declval<year_month_weekday&>() += std::declval<years>()));

    ASSERT_NOEXCEPT(                               std::declval<year_month_weekday&>() -= std::declval<years>());
    ASSERT_SAME_TYPE(year_month_weekday&, decltype(hip::std::declval<year_month_weekday&>() -= std::declval<years>()));

    auto constexpr Tuesday = hip::std::chrono::Tuesday;
    auto constexpr January = hip::std::chrono::January;

    static_assert(testConstexpr<year_month_weekday, years>(year_month_weekday{year{1}, January, weekday_indexed{Tuesday, 2}}), "");

    for (int i = 1000; i <= 1010; ++i)
    {
        year_month_weekday ymwd(year{i}, January, weekday_indexed{Tuesday, 2});

        assert(static_cast<int>((ymwd += years{2}).year()) == i + 2);
        assert(ymwd.month()    == January);
        assert(ymwd.weekday()  == Tuesday);
        assert(ymwd.index()    == 2);

        assert(static_cast<int>((ymwd            ).year()) == i + 2);
        assert(ymwd.month()    == January);
        assert(ymwd.weekday()  == Tuesday);
        assert(ymwd.index()    == 2);

        assert(static_cast<int>((ymwd -= years{1}).year()) == i + 1);
        assert(ymwd.month()    == January);
        assert(ymwd.weekday()  == Tuesday);
        assert(ymwd.index()    == 2);

        assert(static_cast<int>((ymwd            ).year()) == i + 1);
        assert(ymwd.month()    == January);
        assert(ymwd.weekday()  == Tuesday);
        assert(ymwd.index()    == 2);
    }

  return 0;
}
