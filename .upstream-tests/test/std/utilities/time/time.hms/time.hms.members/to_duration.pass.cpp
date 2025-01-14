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
// <chrono>

// template <class Duration>
// class hh_mm_ss
// 
// constexpr precision to_duration() const noexcept;
//
// See the table in hours.pass.cpp for correspondence between the magic values used below

#include <hip/std/chrono>
#include <hip/std/cassert>

#include "test_macros.h"

template <typename Duration>
__host__ __device__
constexpr long long check_duration(Duration d)
{
    using HMS = hip::std::chrono::hh_mm_ss<Duration>;
    ASSERT_SAME_TYPE(typename HMS::precision, decltype(hip::std::declval<HMS>().to_duration()));
    ASSERT_NOEXCEPT(                                   hip::std::declval<HMS>().to_duration());
    
    return HMS(d).to_duration().count();
}

int main(int, char**)
{
    using microfortnights = hip::std::chrono::duration<int, hip::std::ratio<756, 625>>;
    
    static_assert( check_duration(hip::std::chrono::minutes( 1)) ==  60, "");
    static_assert( check_duration(hip::std::chrono::minutes(-1)) == -60, "");

    assert( check_duration(hip::std::chrono::seconds( 5000)) ==    5000LL);
    assert( check_duration(hip::std::chrono::seconds(-5000)) ==   -5000LL);
    assert( check_duration(hip::std::chrono::minutes( 5000)) ==  300000LL);
    assert( check_duration(hip::std::chrono::minutes(-5000)) == -300000LL);
    assert( check_duration(hip::std::chrono::hours( 11))     ==   39600LL);
    assert( check_duration(hip::std::chrono::hours(-11))     ==  -39600LL);

    assert( check_duration(hip::std::chrono::milliseconds( 123456789LL)) ==  123456789LL);
    assert( check_duration(hip::std::chrono::milliseconds(-123456789LL)) == -123456789LL);
    assert( check_duration(hip::std::chrono::microseconds( 123456789LL)) ==  123456789LL);
    assert( check_duration(hip::std::chrono::microseconds(-123456789LL)) == -123456789LL);
    assert( check_duration(hip::std::chrono::nanoseconds( 123456789LL))  ==  123456789LL);
    assert( check_duration(hip::std::chrono::nanoseconds(-123456789LL))  == -123456789LL);

    assert( check_duration(microfortnights(  1000)) ==   12096000);
    assert( check_duration(microfortnights( -1000)) ==  -12096000);
    assert( check_duration(microfortnights( 10000)) ==  120960000);
    assert( check_duration(microfortnights(-10000)) == -120960000);

    return 0;
}
