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
// class weekday;

//                     weekday() = default;
//  explicit constexpr weekday(unsigned wd) noexcept;
//  constexpr weekday(const sys_days& dp) noexcept;
//  explicit constexpr weekday(const local_days& dp) noexcept;
//
//  unsigned c_encoding() const noexcept;

//  Effects: Constructs an object of type weekday by initializing wd_ with wd == 7 ? 0 : wd
//    The value held is unspecified if wd is not in the range [0, 255].

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using weekday = hip::std::chrono::weekday;

    ASSERT_NOEXCEPT(weekday{});
    ASSERT_NOEXCEPT(weekday(1));
    ASSERT_NOEXCEPT(weekday(1).c_encoding());

    constexpr weekday m0{};
    static_assert(m0.c_encoding() == 0, "");

    constexpr weekday m1{1};
    static_assert(m1.c_encoding() == 1, "");

    for (unsigned i = 0; i <= 255; ++i)
    {
        weekday m(i);
        assert(m.c_encoding() == (i == 7 ? 0 : i));
    }

// TODO - sys_days and local_days ctor tests

  return 0;
}
