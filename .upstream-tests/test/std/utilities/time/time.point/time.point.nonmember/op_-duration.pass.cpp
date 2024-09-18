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

// template <class Clock, class Duration1, class Rep2, class Period2>
//   time_point<Clock, typename common_type<Duration1, duration<Rep2, Period2>>::type>
//   operator-(const time_point<Clock, Duration1>& lhs, const duration<Rep2, Period2>& rhs);

#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"

template <class D>
__host__ __device__
void test2739()  // LWG2739
{
    typedef hip::std::chrono::time_point<hip::std::chrono::system_clock> TimePoint;
    typedef hip::std::chrono::duration<D> Dur;
    const Dur d(5);
    TimePoint t0 = hip::std::chrono::system_clock::from_time_t(200);
    TimePoint t1 = t0 - d;
    assert(t1 < t0);
}

int main(int, char**)
{
    typedef hip::std::chrono::system_clock Clock;
    typedef hip::std::chrono::milliseconds Duration1;
    typedef hip::std::chrono::microseconds Duration2;
    {
    hip::std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    hip::std::chrono::time_point<Clock, Duration2> t2 = t1 - Duration2(5);
    assert(t2.time_since_epoch() == Duration2(2995));
    }
#if TEST_STD_VER > 11
    {
    constexpr hip::std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    constexpr hip::std::chrono::time_point<Clock, Duration2> t2 = t1 - Duration2(5);
    static_assert(t2.time_since_epoch() == Duration2(2995), "");
    }
#endif
    test2739<int32_t>();
    test2739<uint32_t>();

  return 0;
}
