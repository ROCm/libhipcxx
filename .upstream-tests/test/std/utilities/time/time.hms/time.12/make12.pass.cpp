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

// constexpr hours make12(const hours& h) noexcept;
//   Returns: The 12-hour equivalent of h in the range [1h, 12h].
//     If h is not in the range [0h, 23h], the value returned is unspecified.

#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    using hours = hip::std::chrono::hours;
    ASSERT_SAME_TYPE(hours, decltype(hip::std::chrono::make12(hip::std::declval<hours>())));
    ASSERT_NOEXCEPT(                 hip::std::chrono::make12(hip::std::declval<hours>()));

    static_assert( hip::std::chrono::make12(hours( 0)) == hours(12), "");
    static_assert( hip::std::chrono::make12(hours(11)) == hours(11), "");
    static_assert( hip::std::chrono::make12(hours(12)) == hours(12), "");
    static_assert( hip::std::chrono::make12(hours(23)) == hours(11), "");
    
    assert( hip::std::chrono::make12(hours(0)) == hours(12));
    for (int i = 1; i < 13; ++i)
        assert( hip::std::chrono::make12(hours(i)) == hours(i));
    for (int i = 13; i < 24; ++i)
        assert( hip::std::chrono::make12(hours(i)) == hours(i-12));

    return 0;
}
