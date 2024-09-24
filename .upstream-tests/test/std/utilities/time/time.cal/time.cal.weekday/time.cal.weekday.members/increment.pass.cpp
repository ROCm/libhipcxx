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

//  constexpr weekday& operator++() noexcept;
//  constexpr weekday operator++(int) noexcept;


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"
#include "../../euclidian.h"

template <typename WD>
__host__ __device__
constexpr bool testConstexpr()
{
    WD wd{5};
    if ((++wd).c_encoding() != 6) return false;
    if ((wd++).c_encoding() != 6) return false;
    if ((wd)  .c_encoding() != 0) return false;
    return true;
}

int main(int, char**)
{
    using weekday = hip::std::chrono::weekday;
    ASSERT_NOEXCEPT(++(hip::std::declval<weekday&>())  );
    ASSERT_NOEXCEPT(  (hip::std::declval<weekday&>())++);

    ASSERT_SAME_TYPE(weekday , decltype(  std::declval<weekday&>()++));
    ASSERT_SAME_TYPE(weekday&, decltype(++std::declval<weekday&>()  ));

    static_assert(testConstexpr<weekday>(), "");

    for (unsigned i = 0; i <= 6; ++i)
    {
        weekday wd(i);
        assert(((++wd).c_encoding() == euclidian_addition<unsigned, 0, 6>(i, 1)));
        assert(((wd++).c_encoding() == euclidian_addition<unsigned, 0, 6>(i, 1)));
        assert(((wd)  .c_encoding() == euclidian_addition<unsigned, 0, 6>(i, 2)));
    }

  return 0;
}
