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

//   constexpr weekday_indexed operator[](unsigned index) const noexcept;
//   constexpr weekday_last    operator[](last_spec)      const noexcept;


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"
#include "../../euclidian.h"

int main(int, char**)
{
    using weekday         = hip::std::chrono::weekday;
    using weekday_last    = hip::std::chrono::weekday_last;
    using weekday_indexed = hip::std::chrono::weekday_indexed;

    constexpr weekday Sunday = hip::std::chrono::Sunday;

    ASSERT_NOEXCEPT(                           hip::std::declval<weekday>()[1U]);
    ASSERT_SAME_TYPE(weekday_indexed, decltype(hip::std::declval<weekday>()[1U]));

    ASSERT_NOEXCEPT(                           hip::std::declval<weekday>()[hip::std::chrono::last]);
    ASSERT_SAME_TYPE(weekday_last,    decltype(hip::std::declval<weekday>()[hip::std::chrono::last]));

    static_assert(Sunday[2].weekday() == Sunday, "");
    static_assert(Sunday[2].index  () == 2, "");

    for (unsigned i = 0; i <= 6; ++i)
    {
        weekday wd(i);
        weekday_last wdl = wd[hip::std::chrono::last];
        assert(wdl.weekday() == wd);
        assert(wdl.ok());
    }

    for (unsigned i = 0; i <= 6; ++i)
        for (unsigned j = 1; j <= 5; ++j)
    {
        weekday wd(i);
        weekday_indexed wdi = wd[j];
        assert(wdi.weekday() == wd);
        assert(wdi.index() == j);
        assert(wdi.ok());
    }

  return 0;
}
