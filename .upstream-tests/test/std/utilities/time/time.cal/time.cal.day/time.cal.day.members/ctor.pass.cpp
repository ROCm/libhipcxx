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

//                     day() = default;
//  explicit constexpr day(unsigned d) noexcept;
//  explicit constexpr operator unsigned() const noexcept;

//  Effects: Constructs an object of type day by initializing d_ with d.
//    The value held is unspecified if d is not in the range [0, 255].

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using day = hip::std::chrono::day;

    ASSERT_NOEXCEPT(day{});
    ASSERT_NOEXCEPT(day(0U));
    ASSERT_NOEXCEPT(static_cast<unsigned>(day(0U)));

    constexpr day d0{};
    static_assert(static_cast<unsigned>(d0) == 0, "");

    constexpr day d1{1};
    static_assert(static_cast<unsigned>(d1) == 1, "");

    for (unsigned i = 0; i <= 255; ++i)
    {
        day day(i);
        assert(static_cast<unsigned>(day) == i);
    }

  return 0;
}
