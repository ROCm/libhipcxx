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

// constexpr bool is_is_leap() const noexcept;
//  y_ % 4 == 0 && (y_ % 100 != 0 || y_ % 400 == 0)
//

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year = hip::std::chrono::year;

    ASSERT_NOEXCEPT(                year(1).is_leap());
    ASSERT_SAME_TYPE(bool, decltype(year(1).is_leap()));

    static_assert(!year{1}.is_leap(), "");
    static_assert(!year{2}.is_leap(), "");
    static_assert(!year{3}.is_leap(), "");
    static_assert( year{4}.is_leap(), "");

    assert( year{-2000}.is_leap());
    assert( year{ -400}.is_leap());
    assert(!year{ -300}.is_leap());
    assert(!year{ -200}.is_leap());

    assert(!year{  200}.is_leap());
    assert(!year{  300}.is_leap());
    assert( year{  400}.is_leap());
    assert(!year{ 1997}.is_leap());
    assert(!year{ 1998}.is_leap());
    assert(!year{ 1999}.is_leap());
    assert( year{ 2000}.is_leap());
    assert(!year{ 2001}.is_leap());
    assert(!year{ 2002}.is_leap());
    assert(!year{ 2003}.is_leap());
    assert( year{ 2004}.is_leap());
    assert(!year{ 2100}.is_leap());

  return 0;
}
