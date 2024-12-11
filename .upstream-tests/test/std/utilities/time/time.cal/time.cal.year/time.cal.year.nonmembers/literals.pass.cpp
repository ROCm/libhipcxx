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
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-5, clang-6, clang-7
// UNSUPPORTED: apple-clang-6, apple-clang-7, apple-clang-8, apple-clang-9
// UNSUPPORTED: apple-clang-10.0.0

// <chrono>
// class year;

// constexpr year operator""y(unsigned long long y) noexcept;

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
#ifndef _LIBCUDACXX_HAS_NO_CXX20_CHRONO_LITERALS
    {
    using namespace hip::std::chrono;
    ASSERT_NOEXCEPT(4y);

    static_assert( 2017y == year(2017), "");
    year y1 = 2018y;
    assert (y1 == year(2018));
    }

    {
    using namespace hip::std::literals;
    ASSERT_NOEXCEPT(4d);

    static_assert( 2017y == hip::std::chrono::year(2017), "");

    hip::std::chrono::year y1 = 2020y;
    assert (y1 == hip::std::chrono::year(2020));
    }
#endif // !_LIBCUDACXX_HAS_NO_CXX20_CHRONO_LITERALS

  return 0;
}
