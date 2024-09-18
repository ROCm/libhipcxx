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

// inline constexpr month January{1};
// inline constexpr month February{2};
// inline constexpr month March{3};
// inline constexpr month April{4};
// inline constexpr month May{5};
// inline constexpr month June{6};
// inline constexpr month July{7};
// inline constexpr month August{8};
// inline constexpr month September{9};
// inline constexpr month October{10};
// inline constexpr month November{11};
// inline constexpr month December{12};


#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{

    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::January));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::February));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::March));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::April));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::May));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::June));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::July));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::August));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::September));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::October));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::November));
    ASSERT_SAME_TYPE(const hip::std::chrono::month, decltype(hip::std::chrono::December));

    static_assert( hip::std::chrono::January   == hip::std::chrono::month(1),  "");
    static_assert( hip::std::chrono::February  == hip::std::chrono::month(2),  "");
    static_assert( hip::std::chrono::March     == hip::std::chrono::month(3),  "");
    static_assert( hip::std::chrono::April     == hip::std::chrono::month(4),  "");
    static_assert( hip::std::chrono::May       == hip::std::chrono::month(5),  "");
    static_assert( hip::std::chrono::June      == hip::std::chrono::month(6),  "");
    static_assert( hip::std::chrono::July      == hip::std::chrono::month(7),  "");
    static_assert( hip::std::chrono::August    == hip::std::chrono::month(8),  "");
    static_assert( hip::std::chrono::September == hip::std::chrono::month(9),  "");
    static_assert( hip::std::chrono::October   == hip::std::chrono::month(10), "");
    static_assert( hip::std::chrono::November  == hip::std::chrono::month(11), "");
    static_assert( hip::std::chrono::December  == hip::std::chrono::month(12), "");

    assert(hip::std::chrono::January   == hip::std::chrono::month(1));
    assert(hip::std::chrono::February  == hip::std::chrono::month(2));
    assert(hip::std::chrono::March     == hip::std::chrono::month(3));
    assert(hip::std::chrono::April     == hip::std::chrono::month(4));
    assert(hip::std::chrono::May       == hip::std::chrono::month(5));
    assert(hip::std::chrono::June      == hip::std::chrono::month(6));
    assert(hip::std::chrono::July      == hip::std::chrono::month(7));
    assert(hip::std::chrono::August    == hip::std::chrono::month(8));
    assert(hip::std::chrono::September == hip::std::chrono::month(9));
    assert(hip::std::chrono::October   == hip::std::chrono::month(10));
    assert(hip::std::chrono::November  == hip::std::chrono::month(11));
    assert(hip::std::chrono::December  == hip::std::chrono::month(12));

    assert(static_cast<unsigned>(hip::std::chrono::January)   ==  1);
    assert(static_cast<unsigned>(hip::std::chrono::February)  ==  2);
    assert(static_cast<unsigned>(hip::std::chrono::March)     ==  3);
    assert(static_cast<unsigned>(hip::std::chrono::April)     ==  4);
    assert(static_cast<unsigned>(hip::std::chrono::May)       ==  5);
    assert(static_cast<unsigned>(hip::std::chrono::June)      ==  6);
    assert(static_cast<unsigned>(hip::std::chrono::July)      ==  7);
    assert(static_cast<unsigned>(hip::std::chrono::August)    ==  8);
    assert(static_cast<unsigned>(hip::std::chrono::September) ==  9);
    assert(static_cast<unsigned>(hip::std::chrono::October)   == 10);
    assert(static_cast<unsigned>(hip::std::chrono::November)  == 11);
    assert(static_cast<unsigned>(hip::std::chrono::December)  == 12);

  return 0;
}
