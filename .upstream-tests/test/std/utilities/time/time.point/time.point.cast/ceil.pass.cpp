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

// ceil

// template <class ToDuration, class Clock, class Duration>
//   time_point<Clock, ToDuration>
//   ceil(const time_point<Clock, Duration>& t);

#include <hip/std/chrono>
#include <hip/std/type_traits>
#include <hip/std/cassert>

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
    typedef decltype(hip::std::chrono::ceil<ToDuration>(f)) R;
    static_assert((hip::std::is_same<R, ToTimePoint>::value), "");
    assert(hip::std::chrono::ceil<ToDuration>(f) == t);
    }
}

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
    static_assert(hip::std::chrono::ceil<ToDuration>(f) == t, "");
    }
}


int main(int, char**)
{
//  7290000ms is 2 hours, 1 minute, and 30 seconds
    test(hip::std::chrono::milliseconds( 7290000), hip::std::chrono::hours( 3));
    test(hip::std::chrono::milliseconds(-7290000), hip::std::chrono::hours(-2));
    test(hip::std::chrono::milliseconds( 7290000), hip::std::chrono::minutes( 122));
    test(hip::std::chrono::milliseconds(-7290000), hip::std::chrono::minutes(-121));

//  9000000ms is 2 hours and 30 minutes
    test_constexpr<hip::std::chrono::milliseconds, 9000000, hip::std::chrono::hours,    3> ();
    test_constexpr<hip::std::chrono::milliseconds,-9000000, hip::std::chrono::hours,   -2> ();
    test_constexpr<hip::std::chrono::milliseconds, 9000001, hip::std::chrono::minutes, 151> ();
    test_constexpr<hip::std::chrono::milliseconds,-9000001, hip::std::chrono::minutes,-150> ();

    test_constexpr<hip::std::chrono::milliseconds, 9000000, hip::std::chrono::seconds, 9000> ();
    test_constexpr<hip::std::chrono::milliseconds,-9000000, hip::std::chrono::seconds,-9000> ();

  return 0;
}
