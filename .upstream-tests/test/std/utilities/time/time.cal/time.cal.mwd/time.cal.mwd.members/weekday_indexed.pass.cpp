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
// class month_weekday;

// constexpr chrono::weekday_indexed weekday_indexed() const noexcept;
//  Returns: wdi_

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using month_weekday   = hip::std::chrono::month_weekday;
    using month           = hip::std::chrono::month;
    using weekday         = hip::std::chrono::weekday;
    using weekday_indexed = hip::std::chrono::weekday_indexed;

    constexpr weekday Sunday = hip::std::chrono::Sunday;

    ASSERT_NOEXCEPT(                           std::declval<const month_weekday>().weekday_indexed());
    ASSERT_SAME_TYPE(weekday_indexed, decltype(hip::std::declval<const month_weekday>().weekday_indexed()));

    static_assert( month_weekday{month{}, weekday_indexed{}}.weekday_indexed() == weekday_indexed{}, "");

    for (unsigned i = 1; i <= 10; ++i)
    {
        constexpr month March = hip::std::chrono::March;
        month_weekday md(March, weekday_indexed{Sunday, i});
        assert( static_cast<unsigned>(md.weekday_indexed().weekday() == Sunday));
        assert( static_cast<unsigned>(md.weekday_indexed().index() == i));
    }

  return 0;
}
