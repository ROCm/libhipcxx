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

// <cuda/std/chrono>

// time_point

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator==(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator!=(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef hip::std::chrono::system_clock Clock;
    typedef hip::std::chrono::milliseconds Duration1;
    typedef hip::std::chrono::microseconds Duration2;
    typedef hip::std::chrono::time_point<Clock, Duration1> T1;
    typedef hip::std::chrono::time_point<Clock, Duration2> T2;

    {
    T1 t1(Duration1(3));
    T1 t2(Duration1(3));
    assert( (t1 == t2));
    assert(!(t1 != t2));
    }
    {
    T1 t1(Duration1(3));
    T1 t2(Duration1(4));
    assert(!(t1 == t2));
    assert( (t1 != t2));
    }
    {
    T1 t1(Duration1(3));
    T2 t2(Duration2(3000));
    assert( (t1 == t2));
    assert(!(t1 != t2));
    }
    {
    T1 t1(Duration1(3));
    T2 t2(Duration2(3001));
    assert(!(t1 == t2));
    assert( (t1 != t2));
    }

#if TEST_STD_VER > 11
    {
    constexpr T1 t1(Duration1(3));
    constexpr T1 t2(Duration1(3));
    static_assert( (t1 == t2), "");
    static_assert(!(t1 != t2), "");
    }
    {
    constexpr T1 t1(Duration1(3));
    constexpr T1 t2(Duration1(4));
    static_assert(!(t1 == t2), "");
    static_assert( (t1 != t2), "");
    }
    {
    constexpr T1 t1(Duration1(3));
    constexpr T2 t2(Duration2(3000));
    static_assert( (t1 == t2), "");
    static_assert(!(t1 != t2), "");
    }
    {
    constexpr T1 t1(Duration1(3));
    constexpr T2 t2(Duration2(3001));
    static_assert(!(t1 == t2), "");
    static_assert( (t1 != t2), "");
    }
#endif

  return 0;
}
