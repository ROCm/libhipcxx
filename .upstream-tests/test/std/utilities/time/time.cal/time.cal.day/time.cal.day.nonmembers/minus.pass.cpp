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

// constexpr day operator-(const day& x, const days& y) noexcept;
//   Returns: x + -y.
//
// constexpr days operator-(const day& x, const day& y) noexcept;
//   Returns: days{int(unsigned{x}) - int(unsigned{y}).


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
    D d{23};
    Ds offset{6};
    if (d - offset != D{17}) return false;
    if (d - D{17} != offset) return false;
    return true;
}

int main(int, char**)
{
    using day  = hip::std::chrono::day;
    using days = hip::std::chrono::days;

    ASSERT_NOEXCEPT(hip::std::declval<day>() - std::declval<days>());
    ASSERT_NOEXCEPT(hip::std::declval<day>() - std::declval<day>());

    ASSERT_SAME_TYPE(day,  decltype(hip::std::declval<day>() - std::declval<days>()));
    ASSERT_SAME_TYPE(days, decltype(hip::std::declval<day>() - std::declval<day>()));

    static_assert(testConstexpr<day, days>(), "");

    day dy{12};
    for (unsigned i = 0; i <= 10; ++i)
    {
        day d1   = dy - days{i};
        days off = dy - day {i};
        assert(static_cast<unsigned>(d1) == 12 - i);
        assert(off.count() == static_cast<int>(12 - i)); // days is signed
    }

  return 0;
}
