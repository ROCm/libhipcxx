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

// inline constexpr weekday   Sunday{0};
// inline constexpr weekday   Monday{1};
// inline constexpr weekday   Tuesday{2};
// inline constexpr weekday   Wednesday{3};
// inline constexpr weekday   Thursday{4};
// inline constexpr weekday   Friday{5};
// inline constexpr weekday   Saturday{6};


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{

    ASSERT_SAME_TYPE(const hip::std::chrono::weekday, decltype(hip::std::chrono::Sunday));
    ASSERT_SAME_TYPE(const hip::std::chrono::weekday, decltype(hip::std::chrono::Monday));
    ASSERT_SAME_TYPE(const hip::std::chrono::weekday, decltype(hip::std::chrono::Tuesday));
    ASSERT_SAME_TYPE(const hip::std::chrono::weekday, decltype(hip::std::chrono::Wednesday));
    ASSERT_SAME_TYPE(const hip::std::chrono::weekday, decltype(hip::std::chrono::Thursday));
    ASSERT_SAME_TYPE(const hip::std::chrono::weekday, decltype(hip::std::chrono::Friday));
    ASSERT_SAME_TYPE(const hip::std::chrono::weekday, decltype(hip::std::chrono::Saturday));

    static_assert( hip::std::chrono::Sunday    == hip::std::chrono::weekday(0),  "");
    static_assert( hip::std::chrono::Monday    == hip::std::chrono::weekday(1),  "");
    static_assert( hip::std::chrono::Tuesday   == hip::std::chrono::weekday(2),  "");
    static_assert( hip::std::chrono::Wednesday == hip::std::chrono::weekday(3),  "");
    static_assert( hip::std::chrono::Thursday  == hip::std::chrono::weekday(4),  "");
    static_assert( hip::std::chrono::Friday    == hip::std::chrono::weekday(5),  "");
    static_assert( hip::std::chrono::Saturday  == hip::std::chrono::weekday(6),  "");

    assert(hip::std::chrono::Sunday    == hip::std::chrono::weekday(0));
    assert(hip::std::chrono::Monday    == hip::std::chrono::weekday(1));
    assert(hip::std::chrono::Tuesday   == hip::std::chrono::weekday(2));
    assert(hip::std::chrono::Wednesday == hip::std::chrono::weekday(3));
    assert(hip::std::chrono::Thursday  == hip::std::chrono::weekday(4));
    assert(hip::std::chrono::Friday    == hip::std::chrono::weekday(5));
    assert(hip::std::chrono::Saturday  == hip::std::chrono::weekday(6));

    assert(hip::std::chrono::Sunday.c_encoding()    ==  0);
    assert(hip::std::chrono::Monday.c_encoding()    ==  1);
    assert(hip::std::chrono::Tuesday.c_encoding()   ==  2);
    assert(hip::std::chrono::Wednesday.c_encoding() ==  3);
    assert(hip::std::chrono::Thursday.c_encoding()  ==  4);
    assert(hip::std::chrono::Friday.c_encoding()    ==  5);
    assert(hip::std::chrono::Saturday.c_encoding()  ==  6);

  return 0;
}
