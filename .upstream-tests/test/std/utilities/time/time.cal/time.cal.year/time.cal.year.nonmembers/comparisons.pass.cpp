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
// class year;

// constexpr bool operator==(const year& x, const year& y) noexcept;
// constexpr bool operator!=(const year& x, const year& y) noexcept;
// constexpr bool operator< (const year& x, const year& y) noexcept;
// constexpr bool operator> (const year& x, const year& y) noexcept;
// constexpr bool operator<=(const year& x, const year& y) noexcept;
// constexpr bool operator>=(const year& x, const year& y) noexcept;


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"


int main(int, char**)
{
    using year = hip::std::chrono::year;

    AssertComparisons6AreNoexcept<year>();
    AssertComparisons6ReturnBool<year>();

    static_assert(testComparisons6Values<year>(0,0), "");
    static_assert(testComparisons6Values<year>(0,1), "");

//  Some 'ok' values as well
    static_assert(testComparisons6Values<year>( 5, 5), "");
    static_assert(testComparisons6Values<year>( 5,10), "");

    for (int i = 1; i < 10; ++i)
        for (int j = 1; j < 10; ++j)
            assert(testComparisons6Values<year>(i, j));

  return 0;
}
