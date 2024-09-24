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
// class day;

// constexpr day& operator+=(const days& d) noexcept;
// constexpr day& operator-=(const days& d) noexcept;

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

template <typename D, typename Ds>
__host__ __device__
constexpr bool testConstexpr()
{
    D d1{1};
    if (static_cast<unsigned>(d1 += Ds{ 1}) !=  2) return false;
    if (static_cast<unsigned>(d1 += Ds{ 2}) !=  4) return false;
    if (static_cast<unsigned>(d1 += Ds{22}) != 26) return false;
    if (static_cast<unsigned>(d1 -= Ds{ 1}) != 25) return false;
    if (static_cast<unsigned>(d1 -= Ds{ 2}) != 23) return false;
    if (static_cast<unsigned>(d1 -= Ds{22}) !=  1) return false;
    return true;
}

int main(int, char**)
{
    using day  = hip::std::chrono::day;
    using days = hip::std::chrono::days;

    ASSERT_NOEXCEPT(hip::std::declval<day&>() += std::declval<days>());
    ASSERT_NOEXCEPT(hip::std::declval<day&>() -= std::declval<days>());

    ASSERT_SAME_TYPE(day&, decltype(hip::std::declval<day&>() += std::declval<days>()));
    ASSERT_SAME_TYPE(day&, decltype(hip::std::declval<day&>() -= std::declval<days>()));

    static_assert(testConstexpr<day, days>(), "");

    for (unsigned i = 0; i <= 10; ++i)
    {
        day day(i);
        assert(static_cast<unsigned>(day += days{22}) == i + 22);
        assert(static_cast<unsigned>(day)             == i + 22);
        assert(static_cast<unsigned>(day -= days{12}) == i + 10);
        assert(static_cast<unsigned>(day)             == i + 10);
    }

  return 0;
}
