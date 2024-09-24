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

// template <class ToDuration, class Clock, class Duration>
//   time_point<Clock, ToDuration>
//   time_point_cast(const time_point<Clock, Duration>& t);

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

#include "test_macros.h"

template <class FromDuration, class ToDuration>
__host__ __device__
void
test(const FromDuration& df, const ToDuration& d)
{
    typedef hip::std::chrono::system_clock Clock;
    typedef hip::std::chrono::time_point<Clock, FromDuration> FromTimePoint;
    typedef hip::std::chrono::time_point<Clock, ToDuration> ToTimePoint;
    {
    FromTimePoint f(df);
    ToTimePoint t(d);
    typedef decltype(hip::std::chrono::time_point_cast<ToDuration>(f)) R;
    static_assert((hip::std::is_same<R, ToTimePoint>::value), "");
    assert(hip::std::chrono::time_point_cast<ToDuration>(f) == t);
    }
}

#if TEST_STD_VER > 11

template<class FromDuration, long long From, class ToDuration, long long To>
__host__ __device__
void test_constexpr ()
{
    typedef hip::std::chrono::system_clock Clock;
    typedef hip::std::chrono::time_point<Clock, FromDuration> FromTimePoint;
    typedef hip::std::chrono::time_point<Clock, ToDuration> ToTimePoint;
    {
    constexpr FromTimePoint f{FromDuration{From}};
    constexpr ToTimePoint   t{ToDuration{To}};
    static_assert(hip::std::chrono::time_point_cast<ToDuration>(f) == t, "");
    }

}

#endif

int main(int, char**)
{
    test(hip::std::chrono::milliseconds(7265000), hip::std::chrono::hours(2));
    test(hip::std::chrono::milliseconds(7265000), hip::std::chrono::minutes(121));
    test(hip::std::chrono::milliseconds(7265000), hip::std::chrono::seconds(7265));
    test(hip::std::chrono::milliseconds(7265000), hip::std::chrono::milliseconds(7265000));
    test(hip::std::chrono::milliseconds(7265000), hip::std::chrono::microseconds(7265000000LL));
    test(hip::std::chrono::milliseconds(7265000), hip::std::chrono::nanoseconds(7265000000000LL));
    test(hip::std::chrono::milliseconds(7265000),
         hip::std::chrono::duration<double, hip::std::ratio<3600> >(7265./3600));
    test(hip::std::chrono::duration<int, hip::std::ratio<2, 3> >(9),
         hip::std::chrono::duration<int, hip::std::ratio<3, 5> >(10));
#if TEST_STD_VER > 11
    {
    test_constexpr<hip::std::chrono::milliseconds, 7265000, hip::std::chrono::hours,    2> ();
    test_constexpr<hip::std::chrono::milliseconds, 7265000, hip::std::chrono::minutes,121> ();
    test_constexpr<hip::std::chrono::milliseconds, 7265000, hip::std::chrono::seconds,7265> ();
    test_constexpr<hip::std::chrono::milliseconds, 7265000, hip::std::chrono::milliseconds,7265000> ();
    test_constexpr<hip::std::chrono::milliseconds, 7265000, hip::std::chrono::microseconds,7265000000LL> ();
    test_constexpr<hip::std::chrono::milliseconds, 7265000, hip::std::chrono::nanoseconds,7265000000000LL> ();
    typedef hip::std::chrono::duration<int, hip::std::ratio<3, 5>> T1;
    test_constexpr<hip::std::chrono::duration<int, hip::std::ratio<2, 3>>, 9, T1, 10> ();
    }
#endif

  return 0;
}
