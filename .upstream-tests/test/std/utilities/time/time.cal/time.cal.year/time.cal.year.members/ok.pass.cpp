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

// constexpr bool ok() const noexcept;
//  Returns: min() <= y_ && y_ <= max().
//
//  static constexpr year min() noexcept;
//   Returns year{ 32767};
//  static constexpr year max() noexcept;
//   Returns year{-32767};

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year = hip::std::chrono::year;

    ASSERT_NOEXCEPT(                std::declval<const year>().ok());
    ASSERT_SAME_TYPE(bool, decltype(hip::std::declval<const year>().ok()));

    ASSERT_NOEXCEPT(                year::max());
    ASSERT_SAME_TYPE(year, decltype(year::max()));

    ASSERT_NOEXCEPT(                year::min());
    ASSERT_SAME_TYPE(year, decltype(year::min()));

    static_assert(static_cast<int>(year::min()) == -32767, "");
    static_assert(static_cast<int>(year::max()) ==  32767, "");

    assert(year{-20001}.ok());
    assert(year{ -2000}.ok());
    assert(year{    -1}.ok());
    assert(year{     0}.ok());
    assert(year{     1}.ok());
    assert(year{  2000}.ok());
    assert(year{ 20001}.ok());

    static_assert(!year{-32768}.ok(), "");

  return 0;
}