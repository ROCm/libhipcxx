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
// class month;

// constexpr month operator-(const month& x, const months& y) noexcept;
//   Returns: x + -y.
//
// constexpr months operator-(const month& x, const month& y) noexcept;
//   Returns: If x.ok() == true and y.ok() == true, returns a value m in the range
//   [months{0}, months{11}] satisfying y + m == x.
//   Otherwise the value returned is unspecified.
//   [Example: January - February == months{11}. —end example]

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

// MSVC warns about unsigned/signed comparisons and addition/subtraction
// Silence these warnings, but not the ones within the header itself.
#if defined(_MSC_VER)
# pragma warning( disable: 4307 )
# pragma warning( disable: 4308 )
#endif

template <typename M, typename Ms>
__host__ __device__
constexpr bool testConstexpr()
{
    {
    M m{5};
    Ms offset{3};
    if (m - offset != M{2}) return false;
    if (m - M{2} != offset) return false;
    }

//  Check the example
    if (M{1} - M{2} != Ms{11}) return false;
    return true;
}

int main(int, char**)
{
    using month  = hip::std::chrono::month;
    using months = hip::std::chrono::months;

    ASSERT_NOEXCEPT(hip::std::declval<month>() - std::declval<months>());
    ASSERT_NOEXCEPT(hip::std::declval<month>() - std::declval<month>());

    ASSERT_SAME_TYPE(month , decltype(hip::std::declval<month>() - std::declval<months>()));
    ASSERT_SAME_TYPE(months, decltype(hip::std::declval<month>() - std::declval<month> ()));

static_assert(testConstexpr<month, months>(), "");

    month m{6};
    for (unsigned i = 1; i <= 12; ++i)
    {
        month m1   = m - months{i};
//      months off = m - month {i};
        int exp = 6 - i;
        if (exp < 1)
            exp += 12;
        assert(static_cast<unsigned>(m1) == static_cast<unsigned>(exp));
//          assert(off.count()               == static_cast<unsigned>(exp));
    }

  return 0;
}
