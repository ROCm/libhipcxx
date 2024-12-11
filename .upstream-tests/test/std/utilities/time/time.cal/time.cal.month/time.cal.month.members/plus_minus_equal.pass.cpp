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

// constexpr month& operator+=(const month& d) noexcept;
// constexpr month& operator-=(const month& d) noexcept;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

template <typename M, typename Ms>
__host__ __device__
constexpr bool testConstexpr()
{
    M m1{1};
    if (static_cast<unsigned>(m1 += Ms{ 1}) !=  2) return false;
    if (static_cast<unsigned>(m1 += Ms{ 2}) !=  4) return false;
    if (static_cast<unsigned>(m1 += Ms{ 8}) != 12) return false;
    if (static_cast<unsigned>(m1 -= Ms{ 1}) != 11) return false;
    if (static_cast<unsigned>(m1 -= Ms{ 2}) !=  9) return false;
    if (static_cast<unsigned>(m1 -= Ms{ 8}) !=  1) return false;
    return true;
}

int main(int, char**)
{
    using month  = hip::std::chrono::month;
    using months = hip::std::chrono::months;

    ASSERT_NOEXCEPT(hip::std::declval<month&>() += std::declval<months&>());
    ASSERT_NOEXCEPT(hip::std::declval<month&>() -= std::declval<months&>());
    ASSERT_SAME_TYPE(month&, decltype(hip::std::declval<month&>() += std::declval<months&>()));
    ASSERT_SAME_TYPE(month&, decltype(hip::std::declval<month&>() -= std::declval<months&>()));

    static_assert(testConstexpr<month, months>(), "");

    for (unsigned i = 1; i <= 10; ++i)
    {
        month month(i);
        int exp = i + 10;
        while (exp > 12)
            exp -= 12;
        assert(static_cast<unsigned>(month += months{10}) == static_cast<unsigned>(exp));
        assert(static_cast<unsigned>(month)               == static_cast<unsigned>(exp));
    }

    for (unsigned i = 1; i <= 10; ++i)
    {
        month month(i);
        int exp = i - 9;
        while (exp < 1)
            exp += 12;
        assert(static_cast<unsigned>(month -= months{ 9}) == static_cast<unsigned>(exp));
        assert(static_cast<unsigned>(month)               == static_cast<unsigned>(exp));
    }

  return 0;
}
