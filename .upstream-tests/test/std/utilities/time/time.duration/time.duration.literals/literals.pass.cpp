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
// <cuda/std/chrono>

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    using namespace hip::std::literals::chrono_literals;

//    Make sure the types are right
    static_assert ( hip::std::is_same<decltype( 3h   ), hip::std::chrono::hours>::value, "" );
    static_assert ( hip::std::is_same<decltype( 3min ), hip::std::chrono::minutes>::value, "" );
    static_assert ( hip::std::is_same<decltype( 3s   ), hip::std::chrono::seconds>::value, "" );
    static_assert ( hip::std::is_same<decltype( 3ms  ), hip::std::chrono::milliseconds>::value, "" );
    static_assert ( hip::std::is_same<decltype( 3us  ), hip::std::chrono::microseconds>::value, "" );
    static_assert ( hip::std::is_same<decltype( 3ns  ), hip::std::chrono::nanoseconds>::value, "" );

    hip::std::chrono::hours h = 4h;
    assert ( h == hip::std::chrono::hours(4));
    auto h2 = 4.0h;
    assert ( h == h2 );

    hip::std::chrono::minutes min = 36min;
    assert ( min == hip::std::chrono::minutes(36));
    auto min2 = 36.0min;
    assert ( min == min2 );

    hip::std::chrono::seconds s = 24s;
    assert ( s == hip::std::chrono::seconds(24));
    auto s2 = 24.0s;
    assert ( s == s2 );

    hip::std::chrono::milliseconds ms = 247ms;
    assert ( ms == hip::std::chrono::milliseconds(247));
    auto ms2 = 247.0ms;
    assert ( ms == ms2 );

    hip::std::chrono::microseconds us = 867us;
    assert ( us == hip::std::chrono::microseconds(867));
    auto us2 = 867.0us;
    assert ( us == us2 );

    hip::std::chrono::nanoseconds ns = 645ns;
    assert ( ns == hip::std::chrono::nanoseconds(645));
    auto ns2 = 645.ns;
    assert ( ns == ns2 );


  return 0;
}
