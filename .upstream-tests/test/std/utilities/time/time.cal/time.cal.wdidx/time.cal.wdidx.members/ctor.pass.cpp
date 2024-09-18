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

//                     weekday_indexed() = default;
//  constexpr weekday_indexed(const chrono::weekday& wd, unsigned index) noexcept;
//
//  Effects: Constructs an object of type weekday_indexed by initializing wd_ with wd and index_ with index.
//    The values held are unspecified if !wd.ok() or index is not in the range [1, 5].
//
//  constexpr chrono::weekday weekday() const noexcept;
//  constexpr unsigned        index()   const noexcept;
//  constexpr bool ok()                 const noexcept;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using weekday  = hip::std::chrono::weekday;
    using weekday_indexed = hip::std::chrono::weekday_indexed;

    ASSERT_NOEXCEPT(weekday_indexed{});
    ASSERT_NOEXCEPT(weekday_indexed(weekday{1}, 1));

    constexpr weekday_indexed wdi0{};
    static_assert( wdi0.weekday() == weekday{}, "");
    static_assert( wdi0.index() == 0,           "");
    static_assert(!wdi0.ok(),                   "");

    constexpr weekday_indexed wdi1{hip::std::chrono::Sunday, 2};
    static_assert( wdi1.weekday() == hip::std::chrono::Sunday, "");
    static_assert( wdi1.index() == 2,                     "");
    static_assert( wdi1.ok(),                             "");

    auto constexpr Tuesday = hip::std::chrono::Tuesday;

    for (unsigned i = 1; i <= 5; ++i)
    {
        weekday_indexed wdi(Tuesday, i);
        assert( wdi.weekday() == Tuesday);
        assert( wdi.index() == i);
        assert( wdi.ok());
    }

    for (unsigned i = 6; i <= 20; ++i)
    {
        weekday_indexed wdi(Tuesday, i);
        assert(!wdi.ok());
    }

  return 0;
}
