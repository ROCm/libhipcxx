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
// UNSUPPORTED: c++98, c++03, c++11
// <chrono>

// constexpr bool is_pm(const hours& h) noexcept;
//   Returns: 12h <= h && h <= 23

#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    using hours = hip::std::chrono::hours;
    ASSERT_SAME_TYPE(bool, decltype(hip::std::chrono::is_pm(hip::std::declval<hours>())));
    ASSERT_NOEXCEPT(                hip::std::chrono::is_pm(hip::std::declval<hours>()));

    static_assert(!hip::std::chrono::is_pm(hours( 0)), "");
    static_assert(!hip::std::chrono::is_pm(hours(11)), "");
    static_assert( hip::std::chrono::is_pm(hours(12)), "");
    static_assert( hip::std::chrono::is_pm(hours(23)), "");
    
    for (int i = 0; i < 12; ++i)
        assert(!hip::std::chrono::is_pm(hours(i)));
    for (int i = 12; i < 24; ++i)
        assert( hip::std::chrono::is_pm(hours(i)));

    return 0;
}
