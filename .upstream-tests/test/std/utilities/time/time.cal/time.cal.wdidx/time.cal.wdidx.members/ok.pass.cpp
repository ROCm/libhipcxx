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
// class weekday_indexed;

// constexpr bool ok() const noexcept;
//  Returns: wd_.ok() && 1 <= index_ && index_ <= 5

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using weekday         = hip::std::chrono::weekday;
    using weekday_indexed = hip::std::chrono::weekday_indexed;

    ASSERT_NOEXCEPT(                std::declval<const weekday_indexed>().ok());
    ASSERT_SAME_TYPE(bool, decltype(hip::std::declval<const weekday_indexed>().ok()));

    static_assert(!weekday_indexed{}.ok(),                       "");
    static_assert( weekday_indexed{hip::std::chrono::Sunday, 2}.ok(), "");

    assert(!(weekday_indexed(hip::std::chrono::Tuesday, 0).ok()));
    auto constexpr Tuesday = hip::std::chrono::Tuesday;
    for (unsigned i = 1; i <= 5; ++i)
    {
        weekday_indexed wdi(Tuesday, i);
        assert( wdi.ok());
    }

    for (unsigned i = 6; i <= 20; ++i)
    {
        weekday_indexed wdi(Tuesday, i);
        assert(!wdi.ok());
    }

//  Not a valid weekday
    assert(!(weekday_indexed(weekday{9U}, 1).ok()));

  return 0;
}
