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
// class weekday_last;

//  explicit constexpr weekday_last(const chrono::weekday& wd) noexcept;
//
//  Effects: Constructs an object of type weekday_last by initializing wd_ with wd.
//
//  constexpr chrono::weekday weekday() const noexcept;
//  constexpr bool ok() const noexcept;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using weekday      = hip::std::chrono::weekday;
    using weekday_last = hip::std::chrono::weekday_last;

    ASSERT_NOEXCEPT(weekday_last{weekday{}});

    constexpr weekday_last wdl0{weekday{}};
    static_assert( wdl0.weekday() == weekday{}, "");
    static_assert( wdl0.ok(),                   "");

    constexpr weekday_last wdl1 {weekday{1}};
    static_assert( wdl1.weekday() == weekday{1}, "");
    static_assert( wdl1.ok(),                    "");

    for (unsigned i = 0; i <= 255; ++i)
    {
        weekday_last wdl{weekday{i}};
        assert(wdl.weekday() == weekday{i});
    }

  return 0;
}
