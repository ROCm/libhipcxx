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
// class month_weekday_last;

//  constexpr month_weekday_last(const chrono::month& m,
//                               const chrono::weekday_last& wdl) noexcept;
//
//  Effects:  Constructs an object of type month_weekday_last by
//            initializing m_ with m, and wdl_ with wdl.
//
//     constexpr chrono::month        month() const noexcept;
//     constexpr chrono::weekday_last weekday_last()  const noexcept;
//     constexpr bool                 ok()    const noexcept;


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using month              = hip::std::chrono::month;
    using weekday            = hip::std::chrono::weekday;
    using weekday_last       = hip::std::chrono::weekday_last;
    using month_weekday_last = hip::std::chrono::month_weekday_last;

    constexpr month January = hip::std::chrono::January;
    constexpr weekday Tuesday = hip::std::chrono::Tuesday;

    ASSERT_NOEXCEPT(month_weekday_last{January, weekday_last{Tuesday}});

//  bad month
    constexpr month_weekday_last mwdl1{month{}, weekday_last{Tuesday}};
    static_assert( mwdl1.month() == month{},                      "");
    static_assert( mwdl1.weekday_last() == weekday_last{Tuesday}, "");
    static_assert(!mwdl1.ok(),                                    "");

//  bad weekday_last
    constexpr month_weekday_last mwdl2{January, weekday_last{weekday{16}}};
    static_assert( mwdl2.month() == January,                          "");
    static_assert( mwdl2.weekday_last() == weekday_last{weekday{16}}, "");
    static_assert(!mwdl2.ok(),                                        "");

//  Good month and weekday_last
    constexpr month_weekday_last mwdl3{January, weekday_last{weekday{4}}};
    static_assert( mwdl3.month() == January,                         "");
    static_assert( mwdl3.weekday_last() == weekday_last{weekday{4}}, "");
    static_assert( mwdl3.ok(),                                       "");

  return 0;
}
